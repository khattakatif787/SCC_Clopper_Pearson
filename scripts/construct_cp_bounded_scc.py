from __future__ import annotations
import math
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from scipy.stats import beta as beta_distribution
except ImportError as exc:
    raise ImportError('This script requires scipy. Install it with: pip install scipy') from exc
PROJECT_ROOT = Path.home() / 'SAC_Extension_Clopper_Upper_Bound_2026-09-04'
DATASETS_ROOT = PROJECT_ROOT / 'datasets'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
TARGET_ALRS = [0.01, 0.001, 0.0001]
CONFIDENCE_LEVELS = [(0.9, '90'), (0.95, '95'), (0.99, '99'), (0.999, '99.9')]
SAVE_SEARCH_TRACE = True
DATASET_FILTER: list[str] | None = None
REJECT_TAG = -1
NON_UM_COLS = {'true_label', 'predicted_label', 'is_misclassification', 'probabilities'}

def alr_tag(value: float) -> str:
    known = {0.01: '0.01', 0.001: '0.001', 0.0001: '0.0001'}
    for k, text in known.items():
        if np.isclose(value, k, rtol=0.0, atol=1e-15):
            return text
    return f'{value:.12g}'

def read_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if 'true_label' not in df.columns:
        raw = pd.read_csv(path, header=None)
        raw.columns = raw.iloc[0].astype(str).str.strip()
        df = raw.iloc[1:].reset_index(drop=True)
        df.columns = [str(c).strip() for c in df.columns]
    required = {'true_label', 'predicted_label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns {sorted(missing)} in: {path}')
    return df

def infer_um_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        name = str(col).strip()
        if name in NON_UM_COLS:
            continue
        if name.lower().startswith('unnamed:'):
            continue
        out.append(name)
    return out

def list_val_files(tmp_folder: Path) -> list[Path]:
    return sorted(tmp_folder.glob('*_VAL.csv'))

def corresponding_test_file(val_path: Path) -> Path:
    return val_path.with_name(val_path.name.replace('_VAL.csv', '_TEST.csv'))

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
        raise ValueError(f'Cannot compare label arrays of different lengths: {len(a)} vs {len(b)}')
    return np.fromiter((labels_equal_single(x, y) for x, y in zip(a, b)), dtype=bool, count=len(a))

def labels_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) != len(b):
        return False
    return bool(np.all(label_equality_mask(a, b)))

def is_reject_value(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and value == REJECT_TAG

def rejection_mask(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=object)
    return np.fromiter((is_reject_value(value) for value in predictions), dtype=bool, count=len(predictions))

def looks_like_combined(um_name: str) -> bool:
    return 'Combined Calculator' in um_name or 'Multiple Combined Calculator' in um_name

def infer_um_direction(um_name: str) -> bool | None:
    name = um_name.strip()
    if name == 'MaxProb Calculator':
        return False
    if name == 'Entropy Calculator':
        return False
    if looks_like_combined(name):
        return False
    if 'AutoEncoder Loss' in name:
        return True
    return None

def is_bounded_confidence_01(um_name: str) -> bool:
    return um_name in {'MaxProb Calculator', 'Entropy Calculator'}

def numeric_array(values: np.ndarray | pd.Series) -> np.ndarray:
    s = pd.to_numeric(pd.Series(values), errors='coerce')
    v = s.to_numpy(dtype=float)
    finite = np.isfinite(v)
    finite_max = np.nanmax(v[finite]) if np.any(finite) else 0.0
    return np.nan_to_num(v, nan=0.0, posinf=finite_max, neginf=0.0)

def to_risk_raw(values: np.ndarray, um_name: str, is_risk: bool) -> np.ndarray:
    v = numeric_array(values)
    if looks_like_combined(um_name) and (not is_risk):
        if np.min(v) < 0.0:
            v = (v + 1.0) / 2.0
    if not is_risk:
        v = 1.0 - v
    return v

def normalize_01_with_params(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    v = values.astype(float)
    denom = vmax - vmin
    if not np.isfinite(denom) or denom <= 0:
        out = np.zeros_like(v, dtype=float)
    else:
        out = (v - vmin) / denom
    out = np.clip(out, 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

def to_risk_scaled_valtest(val_values: np.ndarray, test_values: np.ndarray, um_name: str, is_risk: bool) -> tuple[np.ndarray, np.ndarray]:
    if looks_like_combined(um_name):
        val_conf = numeric_array(val_values)
        test_conf = numeric_array(test_values)
        use_minus1_plus1_scale = np.min(val_conf) < 0.0 or np.min(test_conf) < 0.0
        if use_minus1_plus1_scale:
            val_conf = (val_conf + 1.0) / 2.0
            test_conf = (test_conf + 1.0) / 2.0
        risk_val = 1.0 - val_conf
        risk_test = 1.0 - test_conf
        risk_val = np.clip(risk_val, 0.0, 1.0)
        risk_test = np.clip(risk_test, 0.0, 1.0)
        return (risk_val, risk_test)
    risk_val_raw = to_risk_raw(val_values, um_name, is_risk=is_risk)
    risk_test_raw = to_risk_raw(test_values, um_name, is_risk=is_risk)
    if is_bounded_confidence_01(um_name):
        risk_val = np.clip(risk_val_raw, 0.0, 1.0)
        risk_test = np.clip(risk_test_raw, 0.0, 1.0)
        return (risk_val, risk_test)
    vmin = float(np.min(risk_val_raw))
    vmax = float(np.max(risk_val_raw))
    risk_val = normalize_01_with_params(risk_val_raw, vmin, vmax)
    risk_test = normalize_01_with_params(risk_test_raw, vmin, vmax)
    return (risk_val, risk_test)

def apply_rejection(pred_labels: np.ndarray, risk_scores: np.ndarray, threshold: float) -> np.ndarray:
    reject_mask = risk_scores > threshold
    out = np.asarray(pred_labels, dtype=object).copy()
    out[reject_mask] = REJECT_TAG
    return out

def compute_scc_counts(y_true: np.ndarray, y_pred_with_reject: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred_with_reject = np.asarray(y_pred_with_reject, dtype=object)
    if len(y_true) != len(y_pred_with_reject):
        raise ValueError('y_true and y_pred_with_reject must have the same length.')
    n = len(y_true)
    rejected_mask = rejection_mask(y_pred_with_reject)
    accepted_mask = ~rejected_mask
    equal_mask = label_equality_mask(y_true, y_pred_with_reject)
    correct_mask = accepted_mask & equal_mask
    misclass_mask = accepted_mask & ~equal_mask
    correct = int(np.count_nonzero(correct_mask))
    misclass = int(np.count_nonzero(misclass_mask))
    rejected = int(np.count_nonzero(rejected_mask))
    if correct + misclass + rejected != n:
        raise RuntimeError(f'Internal SCC counting error: correct={correct}, misclass={misclass}, rejected={rejected}, total={n}')
    return {'correct': correct, 'misclass': misclass, 'rejected': rejected, 'total': int(n)}

def counts_to_metrics(counts: dict) -> dict:
    n = counts['total']
    correct = counts['correct']
    misclass = counts['misclass']
    rejected = counts['rejected']
    accepted = correct + misclass
    aw = correct / n if n > 0 else np.nan
    ew = misclass / n if n > 0 else np.nan
    phi = rejected / n if n > 0 else np.nan
    ew_ans = misclass / accepted if accepted > 0 else np.nan
    return {'aw': aw, 'ew': ew, 'phi': phi, 'ew_ans': ew_ans, 'coverage': accepted / n if n > 0 else np.nan}

def build_threshold_profile(y_true: np.ndarray, y_pred: np.ndarray, risk_scores: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    n = len(y_true)
    if n == 0:
        raise ValueError('Validation set is empty.')
    if len(y_pred) != n or len(risk_scores) != n:
        raise ValueError('y_true, y_pred, and risk_scores must have equal length.')
    risk = np.asarray(risk_scores, dtype=float)
    equal_mask = label_equality_mask(y_true, y_pred)
    misclassified = (~equal_mask).astype(np.int64)
    order = np.argsort(risk, kind='mergesort')
    risk_sorted = risk[order]
    mis_sorted = misclassified[order]
    cumulative_mis = np.cumsum(mis_sorted)
    group_ends = np.flatnonzero(np.r_[risk_sorted[1:] != risk_sorted[:-1], True])
    thresholds = risk_sorted[group_ends]
    accepted = group_ends + 1
    misclass = cumulative_mis[group_ends]
    correct = accepted - misclass
    rejected = n - accepted
    return {'n': n, 'thresholds': thresholds.astype(float), 'accepted': accepted.astype(np.int64), 'correct': correct.astype(np.int64), 'misclass': misclass.astype(np.int64), 'rejected': rejected.astype(np.int64)}

def state_for_misclass_budget(profile: dict, max_misclass: int) -> dict:
    mis = profile['misclass']
    idx = int(np.searchsorted(mis, max_misclass, side='right') - 1)
    if idx < 0:
        return {'all_rejected': True, 'threshold': np.nan, 'accepted': 0, 'correct': 0, 'misclass': 0, 'rejected': profile['n'], 'total': profile['n']}
    return {'all_rejected': False, 'threshold': float(profile['thresholds'][idx]), 'accepted': int(profile['accepted'][idx]), 'correct': int(profile['correct'][idx]), 'misclass': int(profile['misclass'][idx]), 'rejected': int(profile['rejected'][idx]), 'total': int(profile['n'])}

def clopper_pearson_upper(k: int, n: int, confidence: float) -> float:
    if n <= 0:
        raise ValueError('n must be > 0')
    if k < 0 or k > n:
        raise ValueError('k must satisfy 0 <= k <= n')
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must be in (0,1)')
    if k == n:
        return 1.0
    if k == 0:
        return float(-np.expm1(np.log1p(-confidence) / n))
    return float(beta_distribution.ppf(confidence, k + 1, n - k))

def search_cp_valid_scc(profile: dict, target_alr: float, confidence: float) -> tuple[dict, list[dict]]:
    n = int(profile['n'])
    if not 0.0 <= target_alr <= 1.0:
        raise ValueError('target_alr must be in [0,1]')
    start_budget = int(math.floor(target_alr * n + 1e-12))
    attempts: list[tuple[float, int]] = [(float(target_alr), start_budget)]
    for budget in range(start_budget - 1, -1, -1):
        attempts.append((budget / n, budget))
    trace: list[dict] = []
    for attempt_no, (x, budget) in enumerate(attempts, start=1):
        state = state_for_misclass_budget(profile, budget)
        if state['all_rejected']:
            trace.append({'attempt': attempt_no, 'x': x, 'allowed_misclass_budget': budget, 'threshold': np.nan, 'accepted': 0, 'correct': 0, 'misclass': 0, 'rejected': n, 'empirical_epsilon': 0.0, 'cp_upper': np.nan, 'decision': 'ALL_REJECTED_STOP'})
            result = {'status': 'ALL_REJECTED_STOP', 'selected_x': np.nan, 'allowed_misclass_budget': budget, 'threshold': np.nan, 'cp_upper': np.nan, 'attempts': attempt_no, 'state': state}
            return (result, trace)
        k = int(state['misclass'])
        cp_upper = clopper_pearson_upper(k, n, confidence)
        empirical_epsilon = k / n
        passed = cp_upper < target_alr
        trace.append({'attempt': attempt_no, 'x': x, 'allowed_misclass_budget': budget, 'threshold': state['threshold'], 'accepted': state['accepted'], 'correct': state['correct'], 'misclass': state['misclass'], 'rejected': state['rejected'], 'empirical_epsilon': empirical_epsilon, 'cp_upper': cp_upper, 'decision': 'PASS' if passed else 'FAIL_CP'})
        if passed:
            result = {'status': 'PASS', 'selected_x': float(x), 'allowed_misclass_budget': int(budget), 'threshold': float(state['threshold']), 'cp_upper': float(cp_upper), 'attempts': attempt_no, 'state': state}
            return (result, trace)
    last_state = state_for_misclass_budget(profile, 0)
    result = {'status': 'NO_CP_VALID_SCC', 'selected_x': np.nan, 'allowed_misclass_budget': 0, 'threshold': np.nan, 'cp_upper': np.nan if last_state['all_rejected'] else clopper_pearson_upper(int(last_state['misclass']), n, confidence), 'attempts': len(attempts), 'state': last_state}
    return (result, trace)

def discover_tmp_folders(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f'Datasets root does not exist: {root}')
    tmp_folders = sorted((p for p in root.rglob('tmp') if p.is_dir()))
    if DATASET_FILTER:
        wanted = [s.replace('\\', '/').strip('/') for s in DATASET_FILTER]
        filtered: list[Path] = []
        for tmp in tmp_folders:
            rel_dataset = tmp.parent.relative_to(root).as_posix()
            if any((w in rel_dataset for w in wanted)):
                filtered.append(tmp)
        tmp_folders = filtered
    return tmp_folders

def process_dataset(tmp_folder: Path) -> None:
    dataset_dir = tmp_folder.parent
    relative_dataset = dataset_dir.relative_to(DATASETS_ROOT)
    print('\n' + '=' * 100)
    print(f'DATASET: {relative_dataset.as_posix()}')
    print(f'INPUT  : {tmp_folder}')
    print('=' * 100)
    val_files = list_val_files(tmp_folder)
    if not val_files:
        print(f'[SKIP DATASET] No *_VAL.csv files found in {tmp_folder}')
        return
    keys = [(confidence, confidence_label, target_alr) for confidence, confidence_label in CONFIDENCE_LEVELS for target_alr in TARGET_ALRS]
    val_outputs: dict[tuple, pd.DataFrame] = {}
    test_outputs: dict[tuple, pd.DataFrame] = {}
    metadata: dict[tuple, list[dict]] = {key: [] for key in keys}
    traces: dict[tuple, list[dict]] = {key: [] for key in keys}
    reference_val_labels: np.ndarray | None = None
    reference_test_labels: np.ndarray | None = None
    valid_pair_count = 0
    for val_path in val_files:
        base = val_path.name.replace('_VAL.csv', '')
        test_path = corresponding_test_file(val_path)
        if not test_path.exists():
            print(f'[WARN] Missing TEST file for {base}: {test_path}')
            continue
        df_val = read_prediction_csv(val_path)
        df_test = read_prediction_csv(test_path)
        y_true_val = df_val['true_label'].to_numpy()
        y_pred_val = df_val['predicted_label'].to_numpy()
        y_true_test = df_test['true_label'].to_numpy()
        y_pred_test = df_test['predicted_label'].to_numpy()
        if len(y_true_val) == 0 or len(y_true_test) == 0:
            print(f'[WARN] Empty VAL/TEST data for {base}; skipping.')
            continue
        if reference_val_labels is None:
            reference_val_labels = y_true_val.copy()
            reference_test_labels = y_true_test.copy()
            for key in keys:
                val_outputs[key] = pd.DataFrame({'true_label': reference_val_labels})
                test_outputs[key] = pd.DataFrame({'true_label': reference_test_labels})
        else:
            if not labels_equal(reference_val_labels, y_true_val):
                raise ValueError(f'Validation row/label alignment differs for {val_path}. All classifier CSVs for a dataset must have the same VAL rows/order.')
            if not labels_equal(reference_test_labels, y_true_test):
                raise ValueError(f'Test row/label alignment differs for {test_path}. All classifier CSVs for a dataset must have the same TEST rows/order.')
        valid_pair_count += 1
        um_cols = infer_um_columns(df_val)
        print(f'\nClassifier: {base}')
        for um in um_cols:
            if um not in df_test.columns:
                print(f'  [SKIP] UM missing in TEST: {um}')
                continue
            is_risk = infer_um_direction(um)
            if is_risk is None:
                print(f'  [SKIP] Unknown UM direction: {um}')
                continue
            risk_val, risk_test = to_risk_scaled_valtest(df_val[um].to_numpy(), df_test[um].to_numpy(), um_name=um, is_risk=is_risk)
            profile = build_threshold_profile(y_true=y_true_val, y_pred=y_pred_val, risk_scores=risk_val)
            col_name = f'{base}_SCC_{um}'
            for confidence, confidence_label, target_alr in keys:
                key = (confidence, confidence_label, target_alr)
                result, trace = search_cp_valid_scc(profile=profile, target_alr=target_alr, confidence=confidence)
                if SAVE_SEARCH_TRACE:
                    for row in trace:
                        traces[key].append({'dataset': relative_dataset.as_posix(), 'classifier': base, 'uncertainty_measure': um, 'target_alr': target_alr, 'confidence': confidence, **row})
                common_meta = {'dataset': relative_dataset.as_posix(), 'classifier': base, 'uncertainty_measure': um, 'scc_column': col_name, 'target_alr': target_alr, 'confidence': confidence, 'status': result['status'], 'selected_x': result['selected_x'], 'allowed_misclass_budget': result['allowed_misclass_budget'], 'threshold': result['threshold'], 'cp_upper': result['cp_upper'], 'cp_margin_to_target': target_alr - result['cp_upper'] if np.isfinite(result['cp_upper']) else np.nan, 'search_attempts': result['attempts']}
                if result['status'] != 'PASS':
                    state = result['state']
                    metadata[key].append({**common_meta, 'val_total': len(y_true_val), 'val_correct': state.get('correct', np.nan), 'val_misclass': state.get('misclass', np.nan), 'val_rejected': state.get('rejected', np.nan), 'val_aw': np.nan, 'val_epsilon': np.nan, 'val_phi': np.nan, 'val_accepted_error': np.nan, 'test_total': len(y_true_test), 'test_correct': np.nan, 'test_misclass': np.nan, 'test_rejected': np.nan, 'test_aw': np.nan, 'test_epsilon': np.nan, 'test_phi': np.nan, 'test_accepted_error': np.nan})
                    print(f"  [{confidence_label}% | ALR={alr_tag(target_alr)}] {um} -> {result['status']}")
                    continue
                threshold = float(result['threshold'])
                val_pred_scc = apply_rejection(y_pred_val, risk_val, threshold)
                test_pred_scc = apply_rejection(y_pred_test, risk_test, threshold)
                val_reject_mask = rejection_mask(val_pred_scc)
                if np.all(val_reject_mask):
                    common_meta['status'] = 'ALL_REJECTED_STOP'
                    metadata[key].append({**common_meta, 'val_total': len(y_true_val), 'val_correct': 0, 'val_misclass': 0, 'val_rejected': len(y_true_val), 'val_aw': 0.0, 'val_epsilon': 0.0, 'val_phi': 1.0, 'val_accepted_error': np.nan, 'test_total': len(y_true_test), 'test_correct': np.nan, 'test_misclass': np.nan, 'test_rejected': np.nan, 'test_aw': np.nan, 'test_epsilon': np.nan, 'test_phi': np.nan, 'test_accepted_error': np.nan})
                    print(f'  [{confidence_label}% | ALR={alr_tag(target_alr)}] {um} -> ALL_REJECTED_STOP')
                    continue
                val_outputs[key][col_name] = val_pred_scc
                test_outputs[key][col_name] = test_pred_scc
                val_counts = compute_scc_counts(y_true_val, val_pred_scc)
                test_counts = compute_scc_counts(y_true_test, test_pred_scc)
                val_metrics = counts_to_metrics(val_counts)
                test_metrics = counts_to_metrics(test_counts)
                if val_counts['correct'] + val_counts['misclass'] + val_counts['rejected'] != val_counts['total']:
                    raise RuntimeError('VAL count consistency error.')
                if test_counts['correct'] + test_counts['misclass'] + test_counts['rejected'] != test_counts['total']:
                    raise RuntimeError('TEST count consistency error.')
                cp_check = clopper_pearson_upper(val_counts['misclass'], val_counts['total'], confidence)
                if not cp_check < target_alr:
                    raise RuntimeError(f'Internal consistency error: a selected SCC does not satisfy CP < target. Dataset={relative_dataset}, classifier={base}, UM={um}, target={target_alr}, confidence={confidence}, CP={cp_check}')
                metadata[key].append({**common_meta, 'cp_upper': cp_check, 'cp_margin_to_target': target_alr - cp_check, 'val_total': val_counts['total'], 'val_correct': val_counts['correct'], 'val_misclass': val_counts['misclass'], 'val_rejected': val_counts['rejected'], 'val_aw': val_metrics['aw'], 'val_epsilon': val_metrics['ew'], 'val_phi': val_metrics['phi'], 'val_accepted_error': val_metrics['ew_ans'], 'test_total': test_counts['total'], 'test_correct': test_counts['correct'], 'test_misclass': test_counts['misclass'], 'test_rejected': test_counts['rejected'], 'test_aw': test_metrics['aw'], 'test_epsilon': test_metrics['ew'], 'test_phi': test_metrics['phi'], 'test_accepted_error': test_metrics['ew_ans']})
                print(f"  [PASS {confidence_label}% | target={alr_tag(target_alr)}] {um} | x={result['selected_x']:.10g} | thr={threshold:.10g} | k={val_counts['misclass']}/{val_counts['total']} | CP={cp_check:.10g} | phi={val_metrics['phi']:.4f}")
    if valid_pair_count == 0 or reference_val_labels is None:
        print(f'[SKIP DATASET] No valid VAL/TEST pairs in {tmp_folder}')
        return
    for confidence, confidence_label, target_alr in keys:
        key = (confidence, confidence_label, target_alr)
        target_text = alr_tag(target_alr)
        base_out_dir = OUTPUT_ROOT / confidence_label / relative_dataset
        scc_dir = base_out_dir / 'SCC_outputs'
        details_dir = base_out_dir / 'SCC_metadata_traces'
        val_path = scc_dir / f'SCC_VALIDATION_ALR_{target_text}.csv'
        test_path = scc_dir / f'SCC_TEST_ALR_{target_text}.csv'
        metadata_path = details_dir / f'SCC_METADATA_ALR_{target_text}.csv'
        trace_path = details_dir / f'SCC_SEARCH_TRACE_ALR_{target_text}.csv'
        legacy_metadata_path = scc_dir / f'SCC_METADATA_ALR_{target_text}.csv'
        legacy_trace_path = scc_dir / f'SCC_SEARCH_TRACE_ALR_{target_text}.csv'
        pass_count = sum((row.get('status') == 'PASS' for row in metadata[key]))
        if pass_count == 0:
            stale_paths = [val_path, test_path, metadata_path, trace_path, legacy_metadata_path, legacy_trace_path]
            removed_any = False
            for stale_path in stale_paths:
                if stale_path.exists():
                    stale_path.unlink()
                    removed_any = True
            print(f'\n[NOT SAVED {confidence_label}% | ALR={target_text}] PASS SCCs=0')
            if removed_any:
                print('  Removed stale CSV files from a previous run.')
            continue
        scc_dir.mkdir(parents=True, exist_ok=True)
        details_dir.mkdir(parents=True, exist_ok=True)
        val_outputs[key].to_csv(val_path, index=False)
        test_outputs[key].to_csv(test_path, index=False)
        pd.DataFrame(metadata[key]).to_csv(metadata_path, index=False)
        if SAVE_SEARCH_TRACE:
            pd.DataFrame(traces[key]).to_csv(trace_path, index=False)
        elif trace_path.exists():
            trace_path.unlink()
        for legacy_path in [legacy_metadata_path, legacy_trace_path]:
            if legacy_path.exists():
                legacy_path.unlink()
        print(f'\n[SAVED {confidence_label}% | ALR={target_text}] PASS SCCs={pass_count}\n  SCC outputs:\n    {val_path}\n    {test_path}\n  Metadata/trace:\n    {metadata_path}')
        if SAVE_SEARCH_TRACE:
            print(f'    {trace_path}')

def main() -> None:
    print('PROJECT_ROOT :', PROJECT_ROOT)
    print('DATASETS_ROOT:', DATASETS_ROOT)
    print('OUTPUT_ROOT  :', OUTPUT_ROOT)
    print('TARGET_ALRS  :', TARGET_ALRS)
    print('CONFIDENCE   :', [label for _, label in CONFIDENCE_LEVELS])
    print('DATASET_FILTER:', DATASET_FILTER)
    tmp_folders = discover_tmp_folders(DATASETS_ROOT)
    if not tmp_folders:
        raise FileNotFoundError(f'No tmp/ dataset folders found under: {DATASETS_ROOT}')
    print(f'\nFound {len(tmp_folders)} dataset tmp/ folders:')
    for tmp in tmp_folders:
        print('  -', tmp.parent.relative_to(DATASETS_ROOT).as_posix())
    for tmp_folder in tmp_folders:
        process_dataset(tmp_folder)
    print('\n' + '=' * 100)
    print('ALL EXPERIMENTS FINISHED')
    print('Outputs:', OUTPUT_ROOT)
    print('=' * 100)
if __name__ == '__main__':
    main()
