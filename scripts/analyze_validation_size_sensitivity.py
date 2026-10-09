from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / 'datasets'
BASE_SCRIPT = PROJECT_ROOT / 'construct_cp_bounded_scc.py'
EXPERIMENT_ROOT = PROJECT_ROOT / 'Validation_Size_CP_Experiment'
DATASETS = {'CCTSDB_50_30_20': {'path': 'Image_Data_Distribution_50_30_20/_CCTSDB', 'expected_n': 9018, 'sizes': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 'FULL']}, 'CICIDS18': {'path': 'Tabular_validation_test_files/CICIDS18', 'expected_n': 30000, 'sizes': [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 'FULL']}}
SEED = 42
TARGET_ALRS = [0.01, 0.001, 0.0001]
CONFIDENCE_LEVELS = [(0.9, '90'), (0.95, '95'), (0.99, '99'), (0.999, '99.9')]
SAVE_SEARCH_TRACE = True

def load_base():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f'Original SCC script not found: {BASE_SCRIPT}')
    spec = importlib.util.spec_from_file_location('base_cp_scc', BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not import: {BASE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
base = load_base()
REQUIRED = ['alr_tag', 'read_prediction_csv', 'infer_um_columns', 'list_val_files', 'corresponding_test_file', 'labels_equal', 'looks_like_combined', 'infer_um_direction', 'numeric_array', 'to_risk_scaled_valtest', 'build_threshold_profile', 'search_cp_valid_scc', 'apply_rejection', 'rejection_mask', 'compute_scc_counts', 'counts_to_metrics', 'clopper_pearson_upper']

def verify_base():
    missing = [name for name in REQUIRED if not hasattr(base, name)]
    if missing:
        raise RuntimeError('Original SCC script is missing: ' + ', '.join(missing))
    if hasattr(base, 'TARGET_ALRS'):
        if list(base.TARGET_ALRS) != TARGET_ALRS:
            raise RuntimeError(f'ALR mismatch: original={base.TARGET_ALRS}, new={TARGET_ALRS}')
    if hasattr(base, 'CONFIDENCE_LEVELS'):
        if list(base.CONFIDENCE_LEVELS) != CONFIDENCE_LEVELS:
            raise RuntimeError(f'Confidence mismatch: original={base.CONFIDENCE_LEVELS}, new={CONFIDENCE_LEVELS}')

def size_tag(size):
    if size == 'FULL':
        return 'FULL'
    return str(int(size))

def confidence_folder(label):
    return 'confidence_' + label.replace('.', '_')

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def get_tmp_folder(relative_path):
    path = DATASETS_ROOT / relative_path / 'tmp'
    if not path.is_dir():
        raise FileNotFoundError(f'Dataset tmp folder not found: {path}')
    return path

def ensure_fresh_output():
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    existing = [p for p in EXPERIMENT_ROOT.rglob('*') if p.is_file()]
    if existing:
        preview = '\n'.join((f'  {p}' for p in existing[:10]))
        raise RuntimeError(f'Validation_Size_CP_Experiment already contains files.\n\nTo avoid mixing experiments, clean only that experiment folder before starting again.\n\nExisting files include:\n{preview}')

def nested_subsets(n_full, sizes):
    finite = [int(size) for size in sizes if size != 'FULL']
    if finite != sorted(finite):
        raise ValueError(f'Sizes must be increasing: {finite}')
    if len(set(finite)) != len(finite):
        raise ValueError(f'Duplicate sizes: {finite}')
    if max(finite) >= n_full:
        raise ValueError(f'Largest finite size {max(finite)} must be below FULL={n_full}')
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(n_full)
    subsets = {}
    for size in sizes:
        if size == 'FULL':
            subsets[size] = np.arange(n_full, dtype=int)
        else:
            subsets[size] = np.sort(permutation[:int(size)]).astype(int)
    for smaller, larger in zip(finite[:-1], finite[1:]):
        if not set(subsets[smaller]).issubset(set(subsets[larger])):
            raise RuntimeError(f'Nestedness failed: {smaller} not contained in {larger}')
    return subsets

def combined_risk(validation_values, test_values):
    val_raw = base.numeric_array(validation_values)
    test_raw = base.numeric_array(test_values)
    tolerance = 1e-12
    if np.min(val_raw) < -1.0 - tolerance or np.max(val_raw) > 1.0 + tolerance or np.min(test_raw) < -1.0 - tolerance or (np.max(test_raw) > 1.0 + tolerance):
        raise ValueError('Combined / Multiple-Combined score outside expected [-1,1] range.')
    val_risk = np.clip((1.0 - val_raw) / 2.0, 0.0, 1.0)
    test_risk = np.clip((1.0 - test_raw) / 2.0, 0.0, 1.0)
    return (val_risk, test_risk)

def risk_scores(df_val, df_test, um):
    is_risk = base.infer_um_direction(um)
    if is_risk is None:
        return None
    if base.looks_like_combined(um):
        return combined_risk(df_val[um].to_numpy(), df_test[um].to_numpy())
    return base.to_risk_scaled_valtest(df_val[um].to_numpy(), df_test[um].to_numpy(), um_name=um, is_risk=is_risk)

def preflight():
    verify_base()
    print('=' * 105)
    print('PREFLIGHT CHECK')
    print('=' * 105)
    print('Original SCC script:', BASE_SCRIPT)
    print('Experiment root    :', EXPERIMENT_ROOT)
    print('ALRs               :', TARGET_ALRS)
    print('Confidence         :', [label for _, label in CONFIDENCE_LEVELS])
    for alias, config in DATASETS.items():
        folder = get_tmp_folder(config['path'])
        val_files = base.list_val_files(folder)
        if not val_files:
            raise FileNotFoundError(f'No *_VAL.csv files for {alias}')
        reference_val = None
        reference_test = None
        combined_count = 0
        for val_path in val_files:
            test_path = base.corresponding_test_file(val_path)
            if not test_path.exists():
                raise FileNotFoundError(f'Missing TEST: {test_path}')
            df_val = base.read_prediction_csv(val_path)
            df_test = base.read_prediction_csv(test_path)
            y_val = df_val['true_label'].to_numpy()
            y_test = df_test['true_label'].to_numpy()
            if len(y_val) != config['expected_n']:
                raise RuntimeError(f"{alias}: expected VAL={config['expected_n']}, found {len(y_val)} in {val_path.name}")
            if reference_val is None:
                reference_val = y_val.copy()
                reference_test = y_test.copy()
            else:
                if not base.labels_equal(reference_val, y_val):
                    raise RuntimeError(f'{alias}: VAL alignment differs in {val_path.name}')
                if not base.labels_equal(reference_test, y_test):
                    raise RuntimeError(f'{alias}: TEST alignment differs in {test_path.name}')
            for um in base.infer_um_columns(df_val):
                if um not in df_test.columns:
                    raise RuntimeError(f'{alias}: {um} missing in TEST')
                if base.looks_like_combined(um):
                    combined_count += 1
                    combined_risk(df_val[um].to_numpy(), df_test[um].to_numpy())
        subsets = nested_subsets(config['expected_n'], config['sizes'])
        print()
        print(f'[OK] {alias}')
        print('     path             :', config['path'])
        print('     full VAL         :', config['expected_n'])
        print('     VAL files        :', len(val_files))
        print('     Combined checks  :', combined_count)
        print('     experiment sizes :', [len(subsets[size]) for size in config['sizes']])
    print()
    print('PREFLIGHT PASSED')

def save_indices(alias, y_true_full, sizes, subsets):
    output_dir = EXPERIMENT_ROOT / 'sampled_indices' / alias / f'seed_{SEED}'
    membership = pd.DataFrame({'original_row_position': np.arange(len(y_true_full), dtype=int), 'true_label': y_true_full})
    for size in sizes:
        indices = subsets[size]
        save_csv(pd.DataFrame({'original_row_position': indices, 'true_label': y_true_full[indices]}), output_dir / f'n_{size_tag(size)}.csv')
        mask = np.zeros(len(y_true_full), dtype=bool)
        mask[indices] = True
        membership[f'in_n_{size_tag(size)}'] = mask
    save_csv(membership, output_dir / 'nested_subset_membership.csv')

def process_size(alias, relative_path, folder, indices, size_label):
    val_files = base.list_val_files(folder)
    keys = [(confidence, confidence_label, target_alr) for confidence, confidence_label in CONFIDENCE_LEVELS for target_alr in TARGET_ALRS]
    val_outputs = {}
    test_outputs = {}
    metadata = {key: [] for key in keys}
    traces = {key: [] for key in keys}
    reference_full_val = None
    reference_subset_val = None
    reference_test = None
    rows = []
    print()
    print('=' * 105)
    print(f'{alias} | n={size_tag(size_label)} ({len(indices)} samples)')
    print('=' * 105)
    for val_path in val_files:
        classifier = val_path.name.replace('_VAL.csv', '')
        test_path = base.corresponding_test_file(val_path)
        df_val_full = base.read_prediction_csv(val_path)
        df_test = base.read_prediction_csv(test_path)
        y_full = df_val_full['true_label'].to_numpy()
        if reference_full_val is None:
            reference_full_val = y_full.copy()
        elif not base.labels_equal(reference_full_val, y_full):
            raise RuntimeError(f'{alias}: full VAL alignment differs in {val_path.name}')
        if np.max(indices) >= len(df_val_full):
            raise IndexError(f'{alias}: subset index exceeds {val_path.name}')
        df_val = df_val_full.iloc[indices].reset_index(drop=True)
        y_val = df_val['true_label'].to_numpy()
        pred_val = df_val['predicted_label'].to_numpy()
        y_test = df_test['true_label'].to_numpy()
        pred_test = df_test['predicted_label'].to_numpy()
        if reference_subset_val is None:
            reference_subset_val = y_val.copy()
            reference_test = y_test.copy()
            for key in keys:
                val_outputs[key] = pd.DataFrame({'true_label': reference_subset_val})
                test_outputs[key] = pd.DataFrame({'true_label': reference_test})
        else:
            if not base.labels_equal(reference_subset_val, y_val):
                raise RuntimeError(f'{alias}: subset VAL alignment differs in {val_path.name}')
            if not base.labels_equal(reference_test, y_test):
                raise RuntimeError(f'{alias}: TEST alignment differs in {test_path.name}')
        print()
        print(f'Classifier: {classifier}')
        for um in base.infer_um_columns(df_val):
            if um not in df_test.columns:
                raise RuntimeError(f'{alias}: {um} missing in TEST')
            pair = risk_scores(df_val, df_test, um)
            if pair is None:
                print(f'  [SKIP] Unknown UM direction: {um}')
                continue
            val_risk, test_risk = pair
            profile = base.build_threshold_profile(y_true=y_val, y_pred=pred_val, risk_scores=val_risk)
            scc_column = f'{classifier}_SCC_{um}'
            for confidence, confidence_label, target_alr in keys:
                key = (confidence, confidence_label, target_alr)
                result, trace = base.search_cp_valid_scc(profile=profile, target_alr=target_alr, confidence=confidence)
                if SAVE_SEARCH_TRACE:
                    for trace_row in trace:
                        traces[key].append({'dataset_alias': alias, 'dataset_path': relative_path, 'validation_size_label': size_tag(size_label), 'validation_size': len(indices), 'seed': SEED, 'classifier': classifier, 'uncertainty_measure': um, 'target_alr': target_alr, 'confidence': confidence, 'confidence_label': confidence_label, **trace_row})
                state = result['state']
                common = {'dataset_alias': alias, 'dataset_path': relative_path, 'validation_size_label': size_tag(size_label), 'validation_size': len(indices), 'full_validation_size': len(df_val_full), 'seed': SEED, 'classifier': classifier, 'uncertainty_measure': um, 'scc_column': scc_column, 'target_alr': target_alr, 'confidence': confidence, 'confidence_label': confidence_label, 'status': result['status'], 'selected_x': result['selected_x'], 'allowed_misclass_budget': result['allowed_misclass_budget'], 'threshold': result['threshold'], 'cp_upper': result['cp_upper'], 'cp_margin_to_target': target_alr - result['cp_upper'] if np.isfinite(result['cp_upper']) else np.nan, 'search_attempts': result['attempts']}
                if result['status'] != 'PASS':
                    state_total = int(state.get('total', len(y_val)))
                    state_misclass = state.get('misclass', np.nan)
                    state_epsilon = state_misclass / state_total if state_total > 0 and np.isfinite(state_misclass) else np.nan
                    row = {**common, 'val_total': len(y_val), 'val_correct': state.get('correct', np.nan), 'val_misclass': state_misclass, 'val_rejected': state.get('rejected', np.nan), 'val_aw': np.nan, 'val_epsilon': state_epsilon, 'val_phi': np.nan, 'val_accepted_error': np.nan, 'test_total': len(y_test), 'test_correct': np.nan, 'test_misclass': np.nan, 'test_rejected': np.nan, 'test_aw': np.nan, 'test_epsilon': np.nan, 'test_phi': np.nan, 'test_accepted_error': np.nan}
                    metadata[key].append(row)
                    rows.append(row)
                    print(f"  [{confidence_label}% | ALR={base.alr_tag(target_alr)}] {um} -> {result['status']}")
                    continue
                threshold = float(result['threshold'])
                val_scc = base.apply_rejection(pred_val, val_risk, threshold)
                test_scc = base.apply_rejection(pred_test, test_risk, threshold)
                if np.all(base.rejection_mask(val_scc)):
                    raise RuntimeError('PASS configuration unexpectedly rejects all validation samples.')
                val_counts = base.compute_scc_counts(y_val, val_scc)
                test_counts = base.compute_scc_counts(y_test, test_scc)
                val_metrics = base.counts_to_metrics(val_counts)
                test_metrics = base.counts_to_metrics(test_counts)
                cp_check = base.clopper_pearson_upper(val_counts['misclass'], val_counts['total'], confidence)
                if not cp_check < target_alr:
                    raise RuntimeError(f'CP verification failed: {alias}, n={len(y_val)}, {classifier}, {um}, CL={confidence}, ALR={target_alr}, CP={cp_check}')
                val_outputs[key][scc_column] = val_scc
                test_outputs[key][scc_column] = test_scc
                row = {**common, 'status': 'PASS', 'cp_upper': cp_check, 'cp_margin_to_target': target_alr - cp_check, 'val_total': val_counts['total'], 'val_correct': val_counts['correct'], 'val_misclass': val_counts['misclass'], 'val_rejected': val_counts['rejected'], 'val_aw': val_metrics['aw'], 'val_epsilon': val_metrics['ew'], 'val_phi': val_metrics['phi'], 'val_accepted_error': val_metrics['ew_ans'], 'test_total': test_counts['total'], 'test_correct': test_counts['correct'], 'test_misclass': test_counts['misclass'], 'test_rejected': test_counts['rejected'], 'test_aw': test_metrics['aw'], 'test_epsilon': test_metrics['ew'], 'test_phi': test_metrics['phi'], 'test_accepted_error': test_metrics['ew_ans']}
                metadata[key].append(row)
                rows.append(row)
                print(f"  [PASS {confidence_label}% | ALR={base.alr_tag(target_alr)}] {um} | k={val_counts['misclass']}/{val_counts['total']} | eps={val_metrics['ew']:.10g} | CP={cp_check:.10g} | phi={val_metrics['phi']:.4f}")
    for confidence, confidence_label, target_alr in keys:
        key = (confidence, confidence_label, target_alr)
        target_text = base.alr_tag(target_alr)
        root = EXPERIMENT_ROOT / alias / f'seed_{SEED}' / f'n_{size_tag(size_label)}' / confidence_folder(confidence_label)
        details_dir = root / 'SCC_metadata_traces'
        save_csv(pd.DataFrame(metadata[key]), details_dir / f'SCC_METADATA_ALR_{target_text}.csv')
        if SAVE_SEARCH_TRACE:
            save_csv(pd.DataFrame(traces[key]), details_dir / f'SCC_SEARCH_TRACE_ALR_{target_text}.csv')
        pass_count = sum((row.get('status') == 'PASS' for row in metadata[key]))
        if pass_count > 0:
            output_dir = root / 'SCC_outputs'
            save_csv(val_outputs[key], output_dir / f'SCC_VALIDATION_ALR_{target_text}.csv')
            save_csv(test_outputs[key], output_dir / f'SCC_TEST_ALR_{target_text}.csv')
    return rows

def process_dataset(alias, config):
    folder = get_tmp_folder(config['path'])
    val_files = base.list_val_files(folder)
    if not val_files:
        raise FileNotFoundError(f'No *_VAL.csv files for {alias}')
    first = base.read_prediction_csv(val_files[0])
    y_full = first['true_label'].to_numpy()
    if len(y_full) != config['expected_n']:
        raise RuntimeError(f"{alias}: expected FULL VAL={config['expected_n']}, found {len(y_full)}")
    subsets = nested_subsets(config['expected_n'], config['sizes'])
    save_indices(alias=alias, y_true_full=y_full, sizes=config['sizes'], subsets=subsets)
    rows = []
    for size in config['sizes']:
        rows.extend(process_size(alias=alias, relative_path=config['path'], folder=folder, indices=subsets[size], size_label=size))
    save_csv(pd.DataFrame(rows), EXPERIMENT_ROOT / 'summaries' / f'{alias}_seed_{SEED}_all_results.csv')
    return rows

def write_manifest():
    lines = ['Validation-size Clopper-Pearson SCC experiment', '', f'Original SCC script: {BASE_SCRIPT}', f'Seed: {SEED}', f'Target ALRs: {TARGET_ALRS}', f'Confidence levels: {CONFIDENCE_LEVELS}', '', 'Datasets:']
    for alias, config in DATASETS.items():
        lines.extend([f'  {alias}', f"    path={config['path']}", f"    full_validation_n={config['expected_n']}", f"    sizes={config['sizes']}"])
    lines.extend(['', 'Finite validation subsets are nested.', 'TEST is never subsampled.', 'Validation-selected threshold is applied unchanged to full TEST.', 'PASS requires strict one-sided Clopper-Pearson U_CP < target ALR.', 'Combined-family preprocessing is fixed and not selected by TEST.'])
    path = EXPERIMENT_ROOT / 'experiment_manifest.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def run_experiment():
    verify_base()
    ensure_fresh_output()
    for folder in [EXPERIMENT_ROOT, EXPERIMENT_ROOT / 'summaries', EXPERIMENT_ROOT / 'sampled_indices', EXPERIMENT_ROOT / 'plots', EXPERIMENT_ROOT / 'logs']:
        folder.mkdir(parents=True, exist_ok=True)
    write_manifest()
    all_rows = []
    for alias, config in DATASETS.items():
        all_rows.extend(process_dataset(alias, config))
    all_results = pd.DataFrame(all_rows)
    all_path = EXPERIMENT_ROOT / 'summaries' / f'ALL_RESULTS_seed_{SEED}.csv'
    save_csv(all_results, all_path)
    passing = all_results[all_results['status'] == 'PASS'].copy()
    pass_path = EXPERIMENT_ROOT / 'summaries' / f'PASSING_SCCS_seed_{SEED}.csv'
    save_csv(passing, pass_path)
    condition_summary = all_results.groupby(['dataset_alias', 'validation_size_label', 'validation_size', 'confidence', 'confidence_label', 'target_alr'], dropna=False).agg(attempted_sccs=('status', 'size'), passing_sccs=('status', lambda values: int((values == 'PASS').sum()))).reset_index()
    condition_path = EXPERIMENT_ROOT / 'summaries' / f'CONDITION_SUMMARY_seed_{SEED}.csv'
    save_csv(condition_summary, condition_path)
    print()
    print('=' * 105)
    print('EXPERIMENT FINISHED')
    print('=' * 105)
    print('Output root        :', EXPERIMENT_ROOT)
    print('Total SCC attempts :', len(all_results))
    print('PASS SCCs          :', len(passing))
    print('All results        :', all_path)
    print('Condition summary  :', condition_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true', help='Check inputs and experimental design without running SCC crafting.')
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        run_experiment()
if __name__ == '__main__':
    main()
