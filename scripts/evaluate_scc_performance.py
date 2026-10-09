from __future__ import annotations
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path.home() / 'SAC_Extension_Clopper_Upper_Bound_2026-09-04'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
CONFIDENCE_LEVELS = ['90', '95', '99', '99.9']
TARGET_ALRS = ['0.01', '0.001', '0.0001']
TRUE_LABEL_COL = 'true_label'
REJECT_TAG = -1

def normalize_label_for_comparison(value):
    try:
        if pd.isna(value):
            return ('missing', None)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = value.strip()
        try:
            numeric_value = Decimal(text)
            if numeric_value.is_finite():
                return ('numeric', numeric_value.normalize())
        except (InvalidOperation, ValueError):
            pass
        return ('string', text)
    if isinstance(value, (int, np.integer)):
        return ('numeric', Decimal(int(value)))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return ('numeric', Decimal(str(float(value))).normalize())
    return ('string', str(value).strip())

def labels_equal_single(a, b) -> bool:
    return normalize_label_for_comparison(a) == normalize_label_for_comparison(b)

def label_equality_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    if len(a) != len(b):
        raise ValueError(f'Cannot compare arrays with different lengths: {len(a)} vs {len(b)}')
    return np.fromiter((labels_equal_single(x, y) for x, y in zip(a, b)), dtype=bool, count=len(a))

def is_rejection_value(value) -> bool:
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value):
                return False
            return float(value) == float(REJECT_TAG)
        if isinstance(value, str):
            text = value.strip()
            try:
                numeric_value = Decimal(text)
                return numeric_value.is_finite() and numeric_value == Decimal(REJECT_TAG)
            except (InvalidOperation, ValueError):
                return False
    except (TypeError, ValueError):
        return False
    return False

def rejection_mask(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=object)
    return np.fromiter((is_rejection_value(value) for value in predictions), dtype=bool, count=len(predictions))

def compute_scc_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> dict:
    true_labels = np.asarray(true_labels, dtype=object)
    predictions = np.asarray(predictions, dtype=object)
    if len(true_labels) != len(predictions):
        raise ValueError('True-label and prediction arrays have different lengths.')
    total = len(true_labels)
    rejected_mask = rejection_mask(predictions)
    accepted_mask = ~rejected_mask
    equal_mask = label_equality_mask(true_labels, predictions)
    correct_mask = accepted_mask & equal_mask
    misclassified_mask = accepted_mask & ~equal_mask
    correct = int(np.count_nonzero(correct_mask))
    misclassified = int(np.count_nonzero(misclassified_mask))
    rejected = int(np.count_nonzero(rejected_mask))
    if correct + misclassified + rejected != total:
        raise RuntimeError(f'SCC accounting error:\nCorrect       = {correct}\nMisclassified = {misclassified}\nRejected      = {rejected}\nTotal         = {total}')
    return {'Correct': correct, 'Misclassified': misclassified, 'Rejected': rejected, 'Total': total}

def read_scc_output(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df.columns = [str(col).strip() for col in df.columns]
    if TRUE_LABEL_COL not in df.columns:
        raise ValueError(f"'{TRUE_LABEL_COL}' column missing from:\n{path}")
    return df

def create_metrics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    true_labels = df[TRUE_LABEL_COL].to_numpy(dtype=object)
    scc_columns = [col for col in df.columns if col != TRUE_LABEL_COL]
    rows = []
    for scc_name in scc_columns:
        predictions = df[scc_name].to_numpy(dtype=object)
        metrics = compute_scc_metrics(true_labels=true_labels, predictions=predictions)
        rows.append({'SCC': scc_name, **metrics})
    return pd.DataFrame(rows, columns=['SCC', 'Correct', 'Misclassified', 'Rejected', 'Total'])

def process_scc_pair(validation_input: Path, test_input: Path, validation_output: Path, test_output: Path, confidence: str, alr: str, dataset_name: str) -> None:
    df_val = read_scc_output(validation_input)
    df_test = read_scc_output(test_input)
    val_scc_columns = [col for col in df_val.columns if col != TRUE_LABEL_COL]
    test_scc_columns = [col for col in df_test.columns if col != TRUE_LABEL_COL]
    if val_scc_columns != test_scc_columns:
        raise RuntimeError(f'VALIDATION and TEST SCC columns differ.\nDataset   : {dataset_name}\nConfidence: {confidence}%\nALR       : {alr}\nVAL SCCs  : {len(val_scc_columns)}\nTEST SCCs : {len(test_scc_columns)}')
    if len(val_scc_columns) == 0:
        print(f'[SKIP {confidence}% | ALR={alr}] No SCC columns.')
        return
    val_metrics = create_metrics_dataframe(df_val)
    test_metrics = create_metrics_dataframe(df_test)
    if not np.array_equal(val_metrics['SCC'].to_numpy(), test_metrics['SCC'].to_numpy()):
        raise RuntimeError('SCC ordering mismatch between VALIDATION and TEST metrics.')
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    val_metrics.to_csv(validation_output, index=False)
    test_metrics.to_csv(test_output, index=False)
    print(f'[DONE {confidence}% | ALR={alr}] SCCs={len(val_metrics)}')
    print(f'  VAL : {validation_output}')
    print(f'  TEST: {test_output}')

def discover_dataset_dirs(confidence_root: Path) -> list[Path]:
    if not confidence_root.exists():
        return []
    scc_output_dirs = sorted((path for path in confidence_root.rglob('SCC_outputs') if path.is_dir()))
    return [path.parent for path in scc_output_dirs]

def process_dataset(dataset_dir: Path, confidence: str) -> None:
    confidence_root = OUTPUT_ROOT / confidence
    relative_dataset = dataset_dir.relative_to(confidence_root)
    dataset_name = relative_dataset.as_posix()
    print('\n' + '=' * 100)
    print(f'CONFIDENCE: {confidence}%')
    print(f'DATASET   : {dataset_name}')
    print('=' * 100)
    scc_outputs_dir = dataset_dir / 'SCC_outputs'
    metrics_dir = dataset_dir / 'SCC_metrics'
    for alr in TARGET_ALRS:
        validation_input = scc_outputs_dir / f'SCC_VALIDATION_ALR_{alr}.csv'
        test_input = scc_outputs_dir / f'SCC_TEST_ALR_{alr}.csv'
        validation_output = metrics_dir / f'SCC_METRICS_VALIDATION_ALR_{alr}.csv'
        test_output = metrics_dir / f'SCC_METRICS_TEST_ALR_{alr}.csv'
        if not validation_input.exists() and (not test_input.exists()):
            print(f'[SKIP {confidence}% | ALR={alr}] No qualified SCC output files.')
            continue
        if validation_input.exists() != test_input.exists():
            raise FileNotFoundError(f'VALIDATION/TEST SCC file mismatch.\nDataset   : {dataset_name}\nConfidence: {confidence}%\nALR       : {alr}\nVAL exists : {validation_input.exists()}\nTEST exists: {test_input.exists()}')
        process_scc_pair(validation_input=validation_input, test_input=test_input, validation_output=validation_output, test_output=test_output, confidence=confidence, alr=alr, dataset_name=dataset_name)

def main() -> None:
    print('PROJECT_ROOT :', PROJECT_ROOT)
    print('OUTPUT_ROOT  :', OUTPUT_ROOT)
    print('CONFIDENCE   :', CONFIDENCE_LEVELS)
    print('TARGET_ALRS  :', TARGET_ALRS)
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f'Output root does not exist:\n{OUTPUT_ROOT}')
    dataset_confidence_combinations = 0
    for confidence in CONFIDENCE_LEVELS:
        confidence_root = OUTPUT_ROOT / confidence
        dataset_dirs = discover_dataset_dirs(confidence_root)
        print('\n' + '#' * 100)
        print(f'CONFIDENCE LEVEL: {confidence}%')
        print(f'Datasets found: {len(dataset_dirs)}')
        print('#' * 100)
        for dataset_dir in dataset_dirs:
            process_dataset(dataset_dir=dataset_dir, confidence=confidence)
            dataset_confidence_combinations += 1
    print('\n' + '=' * 100)
    print('ALL SCC METRICS FINISHED')
    print('Dataset-confidence combinations processed:', dataset_confidence_combinations)
    print('Metrics stored under:', OUTPUT_ROOT)
    print('=' * 100)
if __name__ == '__main__':
    main()
