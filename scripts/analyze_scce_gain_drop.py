from __future__ import annotations
import itertools
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path.home() / 'SAC_Extension_Clopper_Upper_Bound_2026-09-04'
CP_OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / '95'
ALRS = [0.01, 0.001, 0.0001]
OUTPUT_DIR = PROJECT_ROOT / 'gain_drop_analysis_CP95'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_ALIASES = {'CCTSDB2021': ['cctsdb2021', 'cctsdb'], 'TT100K': ['tt100k'], 'NCT-CRC-HE-100K': ['nctcrche100k', 'medicaldataset', 'medical'], 'BackBlaze2023': ['backblaze2023', 'blackblaze2023', 'backblaze'], 'CICIDS18': ['cicids18'], 'UNSW': ['unsw', 'unswnb'], 'ARANCINO': ['arancino']}

def normalize_name(value) -> str:
    return re.sub('[^a-z0-9]+', '', str(value).lower())

def alr_tag(value: float) -> str:
    known = {0.01: '0.01', 0.001: '0.001', 0.0001: '0.0001'}
    for k, text in known.items():
        if np.isclose(value, k, rtol=0.0, atol=1e-15):
            return text
    return f'{value:.12g}'

def normalize_label(value):
    try:
        if pd.isna(value):
            return ('missing', None)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = value.strip()
        try:
            numeric = Decimal(text)
            if numeric.is_finite():
                return ('numeric', numeric.normalize())
        except (InvalidOperation, ValueError):
            pass
        return ('string', text)
    if isinstance(value, (int, np.integer)):
        return ('numeric', Decimal(int(value)))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return ('numeric', Decimal(str(float(value))).normalize())
    return ('string', str(value).strip())

def label_equality_mask(a, b):
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    if len(a) != len(b):
        raise ValueError('Prediction and true-label arrays have different lengths.')
    return np.fromiter((normalize_label(x) == normalize_label(y) for x, y in zip(a, b)), dtype=bool, count=len(a))

def is_reject_value(value) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return np.isfinite(value) and float(value) == -1.0
        except TypeError:
            return False
    if isinstance(value, str):
        return value.strip() == '-1'
    return False

def rejection_mask(predictions):
    predictions = np.asarray(predictions, dtype=object)
    return np.fromiter((is_reject_value(x) for x in predictions), dtype=bool, count=len(predictions))

def mask_to_bits(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=np.uint8)
    packed = np.packbits(mask, bitorder='little')
    return int.from_bytes(packed.tobytes(), byteorder='little', signed=False)

def scc_metrics(pred, y_true):
    pred = np.asarray(pred, dtype=object)
    y_true = np.asarray(y_true, dtype=object)
    n = len(y_true)
    rejected = rejection_mask(pred)
    accepted = ~rejected
    equal = label_equality_mask(pred, y_true)
    correct = accepted & equal
    misclass = accepted & ~equal
    c = int(np.count_nonzero(correct))
    e = int(np.count_nonzero(misclass))
    r = int(np.count_nonzero(rejected))
    if c + e + r != n:
        raise RuntimeError('SCC count consistency error.')
    return {'correct': c, 'misclass': e, 'rejected': r, 'alpha': c / n, 'epsilon': e / n, 'phi': r / n, 'correct_bits': mask_to_bits(correct), 'misclass_bits': mask_to_bits(misclass), 'reject_bits': mask_to_bits(rejected)}

def short_name(column: str) -> str:
    name = str(column).strip()
    name = name.replace('CUSTOM_', '')
    name = name.replace('_SCC_', ' ')
    if 'MaxProb Calculator' in name:
        name = name.replace('MaxProb Calculator', 'RS1')
    if 'Entropy Calculator' in name:
        name = name.replace('Entropy Calculator', 'RS2')
    if 'Multiple Combined Calculator' in name:
        name = re.sub('Multiple Combined Calculator.*', 'RS4', name)
    elif 'Combined Calculator' in name:
        name = re.sub('Combined Calculator.*', 'RS3', name)
    if 'AutoEncoder Loss' in name:
        name = re.sub('AutoEncoder Loss.*', 'RS5', name)
    name = name.replace('EfficientNet', 'EffNet')
    name = name.replace('ShuffleNetV2', 'ShuffleNet')
    name = re.sub('\\s+', ' ', name).strip()
    return name

def paper_tuple(column: str, scc_alr: float) -> str:
    """
    Convert:

        CUSTOM_CN_SCC_MaxProb Calculator

    approximately into:

        (CN, RS1, 0.001)

    whenever the SCC name has the expected form.
    """
    short = short_name(column)
    match = re.match('^(.*?)\\s+(RS[1-5])$', short)
    if match:
        classifier = match.group(1).strip()
        rs = match.group(2).strip()
        return f'({classifier}, {rs}, {alr_tag(scc_alr)})'
    return f'({short}, {alr_tag(scc_alr)})'

def infer_dataset_name(path: Path) -> str | None:
    relative = path.relative_to(CP_OUTPUT_ROOT).as_posix()
    norm = normalize_name(relative)
    for dataset, aliases in DATASET_ALIASES.items():
        for alias in aliases:
            if normalize_name(alias) in norm:
                return dataset
    return None

def discover_validation_files():
    if not CP_OUTPUT_ROOT.exists():
        raise FileNotFoundError(f'95% CP output root not found:\n{CP_OUTPUT_ROOT}')
    found = {}
    for path in sorted(CP_OUTPUT_ROOT.rglob('SCC_VALIDATION_ALR_*.csv')):
        path_text = path.as_posix()
        if 'Image_Data_Distribution' in path_text:
            if 'Image_Data_Distribution_50_30_20' not in path_text:
                continue
        name = path.name
        match = re.search('SCC_VALIDATION_ALR_([0-9.]+)\\.csv$', name)
        if not match:
            continue
        scc_alr = float(match.group(1))
        if not any((np.isclose(scc_alr, x, rtol=0.0, atol=1e-15) for x in ALRS)):
            continue
        dataset = infer_dataset_name(path)
        if dataset is None:
            print('[WARNING] Could not identify dataset:')
            print('         ', path)
            continue
        key = (dataset, scc_alr)
        if key in found:
            raise RuntimeError(f'\nDuplicate validation files for {key}:\n1) {found[key]}\n2) {path}\n')
        found[key] = path
    return found

def read_scc_output(path: Path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if 'true_label' not in df.columns:
        raise ValueError(f'true_label missing from:\n{path}')
    y_true = df['true_label'].to_numpy(dtype=object)
    for value in y_true:
        if normalize_label(value) == ('numeric', Decimal(-1)):
            raise RuntimeError(f'\nDataset contains -1 as a genuine class label.\nCannot safely identify rejection markers in:\n{path}')
    scc_columns = [c for c in df.columns if c != 'true_label']
    if not scc_columns:
        raise RuntimeError(f'No SCC columns found in:\n{path}')
    metrics = {}
    for col in scc_columns:
        pred = df[col].to_numpy(dtype=object)
        metrics[col] = scc_metrics(pred, y_true)
    return (df, y_true, scc_columns, metrics)

def build_pair_candidates(dataset: str, scc_alr: float, columns: list[str], metrics: dict, n: int) -> pd.DataFrame:
    rows = []
    for ia, ib in itertools.permutations(range(len(columns)), 2):
        a = columns[ia]
        b = columns[ib]
        ma = metrics[a]
        mb = metrics[b]
        gain2 = (ma['reject_bits'] & mb['correct_bits']).bit_count()
        drop2 = (ma['reject_bits'] & mb['misclass_bits']).bit_count()
        final_rejected = (ma['reject_bits'] & mb['reject_bits']).bit_count()
        final_correct = ma['correct'] + gain2
        final_misclass = ma['misclass'] + drop2
        if final_correct + final_misclass + final_rejected != n:
            raise RuntimeError(f'Pair accounting failure:\n{dataset}, {a} -> {b}')
        total_gain = gain2 / n
        total_drop = drop2 / n
        rows.append({'Dataset': dataset, 'SCC_ALR': scc_alr, 'Total_ALR': 2 * scc_alr, 'k': 2, 'SCC1': short_name(a), 'SCC2': short_name(b), 'SCC3': '', 'SCC1_Tuple': paper_tuple(a, scc_alr), 'SCC2_Tuple': paper_tuple(b, scc_alr), 'SCC3_Tuple': 'Not a triple (k=2)', 'SCCE_Order': f'{short_name(a)} -> {short_name(b)}', 'SCC1_correct': ma['correct'], 'SCC1_misclass': ma['misclass'], 'SCC1_rejected': ma['rejected'], 'SCC1_alpha': ma['alpha'], 'SCC1_epsilon': ma['epsilon'], 'SCC1_phi': ma['phi'], 'Step2_gain_count': gain2, 'Step2_drop_count': drop2, 'Step3_gain_count': 0, 'Step3_drop_count': 0, 'Total_gain_count': gain2, 'Total_drop_count': drop2, 'Total_Gain': total_gain, 'Total_Drop': total_drop, 'Total_gain_drop': total_gain - total_drop, 'Final_correct': final_correct, 'Final_misclass': final_misclass, 'Final_rejected': final_rejected, 'Final_alpha': final_correct / n, 'Final_epsilon': final_misclass / n, 'Final_phi': final_rejected / n, 'Total_samples': n})
    return pd.DataFrame(rows)

def build_triple_candidates(dataset: str, scc_alr: float, columns: list[str], metrics: dict, n: int) -> pd.DataFrame:
    rows = []
    for ia, ib, ic in itertools.permutations(range(len(columns)), 3):
        a = columns[ia]
        b = columns[ib]
        c = columns[ic]
        ma = metrics[a]
        mb = metrics[b]
        mc = metrics[c]
        step2_gain = (ma['reject_bits'] & mb['correct_bits']).bit_count()
        step2_drop = (ma['reject_bits'] & mb['misclass_bits']).bit_count()
        prefix_reject = ma['reject_bits'] & mb['reject_bits']
        step3_gain = (prefix_reject & mc['correct_bits']).bit_count()
        step3_drop = (prefix_reject & mc['misclass_bits']).bit_count()
        total_gain_count = step2_gain + step3_gain
        total_drop_count = step2_drop + step3_drop
        final_rejected = (prefix_reject & mc['reject_bits']).bit_count()
        final_correct = ma['correct'] + total_gain_count
        final_misclass = ma['misclass'] + total_drop_count
        if final_correct + final_misclass + final_rejected != n:
            raise RuntimeError(f'Triple accounting failure:\n{dataset}, {a} -> {b} -> {c}')
        total_gain = total_gain_count / n
        total_drop = total_drop_count / n
        rows.append({'Dataset': dataset, 'SCC_ALR': scc_alr, 'Total_ALR': 3 * scc_alr, 'k': 3, 'SCC1': short_name(a), 'SCC2': short_name(b), 'SCC3': short_name(c), 'SCC1_Tuple': paper_tuple(a, scc_alr), 'SCC2_Tuple': paper_tuple(b, scc_alr), 'SCC3_Tuple': paper_tuple(c, scc_alr), 'SCCE_Order': f'{short_name(a)} -> {short_name(b)} -> {short_name(c)}', 'SCC1_correct': ma['correct'], 'SCC1_misclass': ma['misclass'], 'SCC1_rejected': ma['rejected'], 'SCC1_alpha': ma['alpha'], 'SCC1_epsilon': ma['epsilon'], 'SCC1_phi': ma['phi'], 'Step2_gain_count': step2_gain, 'Step2_drop_count': step2_drop, 'Step3_gain_count': step3_gain, 'Step3_drop_count': step3_drop, 'Total_gain_count': total_gain_count, 'Total_drop_count': total_drop_count, 'Total_Gain': total_gain, 'Total_Drop': total_drop, 'Total_gain_drop': total_gain - total_drop, 'Final_correct': final_correct, 'Final_misclass': final_misclass, 'Final_rejected': final_rejected, 'Final_alpha': final_correct / n, 'Final_epsilon': final_misclass / n, 'Final_phi': final_rejected / n, 'Total_samples': n})
    return pd.DataFrame(rows)

def summarize_candidates(cands: pd.DataFrame) -> list[dict]:
    if cands.empty:
        return []
    positive = int((cands['Total_gain_drop'] > 0).sum())
    zero = int((cands['Total_gain_drop'] == 0).sum())
    negative = int((cands['Total_gain_drop'] < 0).sum())
    total = len(cands)
    best = cands.loc[cands['Final_correct'].idxmax()]
    median_alpha = cands['Final_alpha'].median()
    temp = cands.copy()
    temp['_median_distance'] = (temp['Final_alpha'] - median_alpha).abs()
    median = temp.sort_values(by=['_median_distance', 'Final_misclass', 'Total_gain_drop'], ascending=[True, True, False]).iloc[0].drop(labels=['_median_distance'])
    worst = cands.sort_values(by=['Total_gain_drop', 'Final_alpha'], ascending=[True, True]).iloc[0]

    def enrich(row, case_name):
        out = row.to_dict()
        out['Case'] = case_name
        out['Positive_gain_drop_candidates'] = positive
        out['Zero_gain_drop_candidates'] = zero
        out['Negative_gain_drop_candidates'] = negative
        out['Total_candidates'] = total
        out['Positive_gain_drop_pct'] = 100 * positive / total
        out['Zero_gain_drop_pct'] = 100 * zero / total
        out['Negative_gain_drop_pct'] = 100 * negative / total
        return out
    return [enrich(best, 'Best SCCE'), enrich(median, 'Median SCCE'), enrich(worst, 'Worst SCCE')]

def condition_stats(dataset: str, scc_alr: float, k: int, scc_count: int, cands: pd.DataFrame):
    total = len(cands)
    positive = int((cands['Total_gain_drop'] > 0).sum())
    zero = int((cands['Total_gain_drop'] == 0).sum())
    negative = int((cands['Total_gain_drop'] < 0).sum())
    return {'Dataset': dataset, 'Confidence': '95%', 'SCC_ALR': scc_alr, 'Total_ALR': k * scc_alr, 'k': k, 'Available_SCCs': scc_count, 'Total_candidates': total, 'Positive_gain_drop': positive, 'Zero_gain_drop': zero, 'Negative_gain_drop': negative, 'Positive_pct': 100 * positive / total if total else np.nan, 'Zero_pct': 100 * zero / total if total else np.nan, 'Negative_pct': 100 * negative / total if total else np.nan}

def make_paper_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = pd.DataFrame({'Dataset': df['Dataset'], 'k': df['k'], 'Case': df['Case'], 'SCC_ALR': df['SCC_ALR'], 'Total_ALR': df['Total_ALR'], 'SCC1': df['SCC1_Tuple'], 'SCC2': df['SCC2_Tuple'], 'SCC3': df['SCC3_Tuple'], 'SCC1 alpha': df['SCC1_alpha'], 'SCC1 epsilon': df['SCC1_epsilon'], 'Total Gain': df['Total_Gain'], 'Total Drop': df['Total_Drop'], 'Gain-Drop': df['Total_gain_drop'], 'SCCE alpha': df['Final_alpha'], 'SCCE epsilon': df['Final_epsilon'], 'SCCE phi': df['Final_phi'], 'Positive candidates (%)': df['Positive_gain_drop_pct'], 'Negative candidates (%)': df['Negative_gain_drop_pct']})
    metric_columns = ['SCC1 alpha', 'SCC1 epsilon', 'Total Gain', 'Total Drop', 'Gain-Drop', 'SCCE alpha', 'SCCE epsilon', 'SCCE phi']
    for col in metric_columns:
        out[col] = pd.to_numeric(out[col], errors='coerce').round(4)
    for col in ['Positive candidates (%)', 'Negative candidates (%)']:
        out[col] = pd.to_numeric(out[col], errors='coerce').round(2)
    return out

def main():
    print('=' * 90)
    print('95% CP GAIN/DROP ANALYSIS')
    print('ALL ORDERED SCC PAIRS AND TRIPLES')
    print('=' * 90)
    print('Project root :', PROJECT_ROOT)
    print('Input root   :', CP_OUTPUT_ROOT)
    print('Output       :', OUTPUT_DIR)
    files = discover_validation_files()
    if not files:
        raise RuntimeError('No 95% SCC validation files found.')
    print(f'\nFound {len(files)} dataset/ALR SCC files.')
    pair_summary_rows = []
    triple_summary_rows = []
    stats_rows = []
    dataset_order = ['CCTSDB2021', 'TT100K', 'NCT-CRC-HE-100K', 'BackBlaze2023', 'CICIDS18', 'UNSW', 'ARANCINO']
    for dataset in dataset_order:
        for scc_alr in ALRS:
            key = (dataset, scc_alr)
            if key not in files:
                print(f'\n[NQ / MISSING] {dataset} | SCC ALR={scc_alr}')
                continue
            path = files[key]
            print('\n' + '-' * 90)
            print(f'Dataset : {dataset}')
            print(f'SCC ALR : {scc_alr}')
            print(f'File    : {path}')
            df, y_true, columns, metrics = read_scc_output(path)
            n = len(y_true)
            m = len(columns)
            print(f'Samples : {n}')
            print(f'CP-valid SCCs: {m}')
            if m >= 2:
                pair_count = m * (m - 1)
                print(f'Building {pair_count:,} ordered pairs...')
                pair_candidates = build_pair_candidates(dataset=dataset, scc_alr=scc_alr, columns=columns, metrics=metrics, n=n)
                pair_summary_rows.extend(summarize_candidates(pair_candidates))
                stats_rows.append(condition_stats(dataset=dataset, scc_alr=scc_alr, k=2, scc_count=m, cands=pair_candidates))
                del pair_candidates
            else:
                print('Not enough SCCs for ordered pairs.')
            if m >= 3:
                triple_count = m * (m - 1) * (m - 2)
                print(f'Building {triple_count:,} ordered triples...')
                triple_candidates = build_triple_candidates(dataset=dataset, scc_alr=scc_alr, columns=columns, metrics=metrics, n=n)
                triple_summary_rows.extend(summarize_candidates(triple_candidates))
                stats_rows.append(condition_stats(dataset=dataset, scc_alr=scc_alr, k=3, scc_count=m, cands=triple_candidates))
                del triple_candidates
            else:
                print('Not enough SCCs for ordered triples.')
    pair_summary = pd.DataFrame(pair_summary_rows)
    triple_summary = pd.DataFrame(triple_summary_rows)
    stats_df = pd.DataFrame(stats_rows)
    all_summary = pd.concat([pair_summary, triple_summary], ignore_index=True)
    paper_table = make_paper_table(all_summary)
    pair_paper = make_paper_table(pair_summary)
    triple_paper = make_paper_table(triple_summary)
    pair_csv = OUTPUT_DIR / 'pair_best_median_worst_CP95.csv'
    triple_csv = OUTPUT_DIR / 'triple_best_median_worst_CP95.csv'
    combined_csv = OUTPUT_DIR / 'paper_representative_gain_drop_CP95.csv'
    stats_csv = OUTPUT_DIR / 'gain_drop_condition_statistics_CP95.csv'
    pair_summary.to_csv(pair_csv, index=False)
    triple_summary.to_csv(triple_csv, index=False)
    paper_table.to_csv(combined_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    xlsx = OUTPUT_DIR / 'gain_drop_best_median_worst_CP95.xlsx'
    with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
        paper_table.to_excel(writer, sheet_name='Paper_Table', index=False)
        pair_paper.to_excel(writer, sheet_name='Pairs_Paper', index=False)
        triple_paper.to_excel(writer, sheet_name='Triples_Paper', index=False)
        pair_summary.to_excel(writer, sheet_name='Pairs_Audit', index=False)
        triple_summary.to_excel(writer, sheet_name='Triples_Audit', index=False)
        stats_df.to_excel(writer, sheet_name='Condition_Stats', index=False)
    print('\n' + '=' * 90)
    print('GAIN/DROP ANALYSIS FINISHED')
    print('=' * 90)
    print(f'Representative pair rows   : {len(pair_summary)}')
    print(f'Representative triple rows : {len(triple_summary)}')
    print('\nOutputs:')
    print(pair_csv)
    print(triple_csv)
    print(combined_csv)
    print(stats_csv)
    print(xlsx)
    print("\nUse the 'Paper_Table' sheet to identify representative datasets for the revised Table III.")
if __name__ == '__main__':
    main()
