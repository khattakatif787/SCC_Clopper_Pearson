from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import time
import numpy as np
import pandas as pd
PROJECT_ROOT = Path.home() / 'SAC_Extension_Clopper_Upper_Bound_2026-09-04'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
CONFIDENCE_LEVELS = ['90', '95', '99', '99.9']
INDIVIDUAL_ALRS = ['0.01', '0.001', '0.0001']
ENSEMBLE_SIZES = [2, 3]
TRUE_LABEL_COL = 'true_label'
REJECT_TAG = -1

def as_text(values) -> pd.Series:
    return pd.Series(values, dtype='object').astype(str).str.strip()

def rejection_mask(values) -> np.ndarray:
    numeric = pd.to_numeric(as_text(values), errors='coerce')
    return numeric.to_numpy() == float(REJECT_TAG)

def label_equal(true_values, predicted_values) -> np.ndarray:
    true_text = as_text(true_values)
    pred_text = as_text(predicted_values)
    true_numeric = pd.to_numeric(true_text, errors='coerce')
    pred_numeric = pd.to_numeric(pred_text, errors='coerce')
    both_numeric = (true_numeric.notna() & pred_numeric.notna()).to_numpy()
    equal = np.zeros(len(true_text), dtype=bool)
    if np.any(both_numeric):
        equal[both_numeric] = true_numeric.to_numpy()[both_numeric] == pred_numeric.to_numpy()[both_numeric]
    non_numeric = ~both_numeric
    if np.any(non_numeric):
        equal[non_numeric] = true_text.to_numpy()[non_numeric] == pred_text.to_numpy()[non_numeric]
    return equal

def compute_metrics(true_labels, predictions) -> dict:
    true_labels = np.asarray(true_labels, dtype=object)
    predictions = np.asarray(predictions, dtype=object)
    if len(true_labels) != len(predictions):
        raise ValueError('true_labels and predictions have different lengths.')
    rejected = rejection_mask(predictions)
    accepted = ~rejected
    equal = label_equal(true_labels, predictions)
    correct = int(np.count_nonzero(accepted & equal))
    misclassified = int(np.count_nonzero(accepted & ~equal))
    rejected_count = int(np.count_nonzero(rejected))
    total = len(true_labels)
    if correct + misclassified + rejected_count != total:
        raise RuntimeError('Correct + Misclassified + Rejected != Total')
    return {'Correct': correct, 'Misclassified': misclassified, 'Rejected': rejected_count, 'Total': total}

def mask_to_int(mask: np.ndarray) -> int:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder='little')
    return int.from_bytes(packed.tobytes(), byteorder='little', signed=False)

def build_validation_bitsets(df_val: pd.DataFrame, scc_columns: list[str]):
    true_labels = df_val[TRUE_LABEL_COL].to_numpy(dtype=object)
    correct_bits = {}
    reject_bits = {}
    for scc in scc_columns:
        predictions = df_val[scc].to_numpy(dtype=object)
        rejected = rejection_mask(predictions)
        correct = ~rejected & label_equal(true_labels, predictions)
        correct_bits[scc] = mask_to_int(correct)
        reject_bits[scc] = mask_to_int(rejected)
    return (correct_bits, reject_bits)

def number_of_ordered_sequences(number_of_sccs: int, k: int) -> int:
    if number_of_sccs < k:
        return 0
    if k == 2:
        return number_of_sccs * (number_of_sccs - 1)
    if k == 3:
        return number_of_sccs * (number_of_sccs - 1) * (number_of_sccs - 2)
    raise ValueError('Only k=2 and k=3 are supported.')

def find_best_exact_order(scc_columns: list[str], correct_bits: dict[str, int], reject_bits: dict[str, int], k: int):
    if k not in (2, 3):
        raise ValueError('Only ensemble sizes 2 and 3 are supported.')
    if len(scc_columns) < k:
        return ([], -1)
    best_order = []
    best_score = -1
    if k == 2:
        for scc_a in scc_columns:
            correct_a = correct_bits[scc_a].bit_count()
            reject_a = reject_bits[scc_a]
            for scc_b in scc_columns:
                if scc_b == scc_a:
                    continue
                score = correct_a + (reject_a & correct_bits[scc_b]).bit_count()
                if score > best_score:
                    best_score = score
                    best_order = [scc_a, scc_b]
        return (best_order, best_score)
    for scc_a in scc_columns:
        correct_a = correct_bits[scc_a].bit_count()
        reject_a = reject_bits[scc_a]
        for scc_b in scc_columns:
            if scc_b == scc_a:
                continue
            correct_ab = correct_a + (reject_a & correct_bits[scc_b]).bit_count()
            reject_ab = reject_a & reject_bits[scc_b]
            for scc_c in scc_columns:
                if scc_c == scc_a or scc_c == scc_b:
                    continue
                score = correct_ab + (reject_ab & correct_bits[scc_c]).bit_count()
                if score > best_score:
                    best_score = score
                    best_order = [scc_a, scc_b, scc_c]
    return (best_order, best_score)

def apply_ordered_ensemble(df: pd.DataFrame, ensemble_order: list[str]) -> np.ndarray:
    total = len(df)
    output = np.full(total, REJECT_TAG, dtype=object)
    unresolved = np.ones(total, dtype=bool)
    for scc in ensemble_order:
        predictions = df[scc].to_numpy(dtype=object)
        rejected = rejection_mask(predictions)
        take = unresolved & ~rejected
        output[take] = predictions[take]
        unresolved[take] = False
        if not np.any(unresolved):
            break
    return output

def read_scc_pair(validation_path: Path, test_path: Path):
    df_val = pd.read_csv(validation_path, keep_default_na=False)
    df_test = pd.read_csv(test_path, keep_default_na=False)
    df_val.columns = [str(column).strip() for column in df_val.columns]
    df_test.columns = [str(column).strip() for column in df_test.columns]
    if TRUE_LABEL_COL not in df_val.columns:
        raise ValueError(f"Missing '{TRUE_LABEL_COL}' in:\n{validation_path}")
    if TRUE_LABEL_COL not in df_test.columns:
        raise ValueError(f"Missing '{TRUE_LABEL_COL}' in:\n{test_path}")
    validation_sccs = [column for column in df_val.columns if column != TRUE_LABEL_COL]
    test_sccs = [column for column in df_test.columns if column != TRUE_LABEL_COL]
    if validation_sccs != test_sccs:
        raise RuntimeError(f'VALIDATION and TEST SCC columns differ.\nVAL : {validation_path}\nTEST: {test_path}')
    return (df_val, df_test, validation_sccs)

def get_system_alr(individual_alr: str, k: int) -> str:
    return format(Decimal(individual_alr) * Decimal(k), 'f')

def save_result(dataset_dir: Path, confidence: str, individual_alr: str, k: int, candidate_count: int, best_order: list[str], best_validation_correct: int, df_val: pd.DataFrame, df_test: pd.DataFrame):
    target_system_alr = get_system_alr(individual_alr, k)
    if Decimal(individual_alr) * Decimal(k) > Decimal(target_system_alr):
        raise RuntimeError('k * alr > ALR_t')
    validation_predictions = apply_ordered_ensemble(df_val, best_order)
    test_predictions = apply_ordered_ensemble(df_test, best_order)
    true_validation = df_val[TRUE_LABEL_COL].to_numpy(dtype=object)
    true_test = df_test[TRUE_LABEL_COL].to_numpy(dtype=object)
    validation_metrics = compute_metrics(true_validation, validation_predictions)
    test_metrics = compute_metrics(true_test, test_predictions)
    if validation_metrics['Correct'] != best_validation_correct:
        raise RuntimeError('Search score does not match recomputed VALIDATION Correct.')
    output_dir = dataset_dir / 'SCCE_exact_ordered' / f'k_{k}' / f'individual_ALR_{individual_alr}'
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TRUE_LABEL_COL: df_val[TRUE_LABEL_COL], 'SCCE_prediction': validation_predictions}).to_csv(output_dir / 'SCCE_VALIDATION.csv', index=False)
    pd.DataFrame({TRUE_LABEL_COL: df_test[TRUE_LABEL_COL], 'SCCE_prediction': test_predictions}).to_csv(output_dir / 'SCCE_TEST.csv', index=False)
    pd.DataFrame({'Position': list(range(1, k + 1)), 'SCC': best_order}).to_csv(output_dir / 'SELECTED_ORDER.csv', index=False)
    common_information = {'Confidence': confidence, 'Individual_SCC_ALR': individual_alr, 'System_ALR': target_system_alr, 'Ensemble_Size': k, 'Candidate_Ordered_Sequences': candidate_count}
    pd.DataFrame([{**common_information, **validation_metrics}]).to_csv(output_dir / 'SCCE_METRICS_VALIDATION.csv', index=False)
    pd.DataFrame([{**common_information, **test_metrics}]).to_csv(output_dir / 'SCCE_METRICS_TEST.csv', index=False)

def discover_dataset_dirs(confidence_root: Path):
    if not confidence_root.exists():
        return []
    return sorted((path.parent for path in confidence_root.rglob('SCC_outputs') if path.is_dir()))

def process_dataset(dataset_dir: Path, confidence: str):
    relative_dataset = dataset_dir.relative_to(OUTPUT_ROOT / confidence)
    scc_folder = dataset_dir / 'SCC_outputs'
    print('\n' + '=' * 100)
    print(f'CONFIDENCE: {confidence}%')
    print(f'DATASET   : {relative_dataset.as_posix()}')
    print('=' * 100)
    for individual_alr in INDIVIDUAL_ALRS:
        validation_path = scc_folder / f'SCC_VALIDATION_ALR_{individual_alr}.csv'
        test_path = scc_folder / f'SCC_TEST_ALR_{individual_alr}.csv'
        if not validation_path.exists() and (not test_path.exists()):
            print(f'[SKIP ALR={individual_alr}] No qualified SCC files.')
            continue
        if validation_path.exists() != test_path.exists():
            raise FileNotFoundError(f'VALIDATION/TEST SCC file mismatch:\n{validation_path}\n{test_path}')
        df_val, df_test, scc_columns = read_scc_pair(validation_path, test_path)
        print(f'\nIndividual SCC ALR={individual_alr} | qualified SCCs={len(scc_columns)}')
        correct_bits, reject_bits = build_validation_bitsets(df_val, scc_columns)
        for k in ENSEMBLE_SIZES:
            if len(scc_columns) < k:
                print(f'[SKIP k={k}] Only {len(scc_columns)} SCCs available; exactly {k} required.')
                continue
            target_system_alr = get_system_alr(individual_alr, k)
            if Decimal(individual_alr) * Decimal(k) > Decimal(target_system_alr):
                print(f'[SKIP k={k}] {k} * {individual_alr} > ALR_t={target_system_alr}')
                continue
            candidate_count = number_of_ordered_sequences(len(scc_columns), k)
            print(f'[SEARCH k={k}] ordered candidates={candidate_count:,} | ALR_t={target_system_alr}')
            start = time.perf_counter()
            best_order, best_score = find_best_exact_order(scc_columns, correct_bits, reject_bits, k)
            elapsed = time.perf_counter() - start
            if not best_order:
                print(f'[NO VALID ENSEMBLE k={k}]')
                continue
            save_result(dataset_dir=dataset_dir, confidence=confidence, individual_alr=individual_alr, k=k, candidate_count=candidate_count, best_order=best_order, best_validation_correct=best_score, df_val=df_val, df_test=df_test)
            print(f'[DONE k={k}] best VALIDATION Correct={best_score} | time={elapsed:.3f}s')
            for position, scc in enumerate(best_order, start=1):
                print(f'  {position}: {scc}')

def main():
    print('PROJECT_ROOT    :', PROJECT_ROOT)
    print('OUTPUT_ROOT     :', OUTPUT_ROOT)
    print('CONFIDENCE      :', CONFIDENCE_LEVELS)
    print('INDIVIDUAL ALRS :', INDIVIDUAL_ALRS)
    print('ENSEMBLE SIZES  :', ENSEMBLE_SIZES)
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f'Missing output root:\n{OUTPUT_ROOT}')
    processed = 0
    for confidence in CONFIDENCE_LEVELS:
        confidence_root = OUTPUT_ROOT / confidence
        dataset_dirs = discover_dataset_dirs(confidence_root)
        print('\n' + '#' * 100)
        print(f'CONFIDENCE LEVEL: {confidence}%')
        print(f'Datasets found: {len(dataset_dirs)}')
        print('#' * 100)
        for dataset_dir in dataset_dirs:
            process_dataset(dataset_dir, confidence)
            processed += 1
    print('\n' + '=' * 100)
    print('ALL EXACT ORDERED SCC ENSEMBLE SEARCHES FINISHED')
    print('Dataset-confidence combinations processed:', processed)
    print('=' * 100)
if __name__ == '__main__':
    main()
