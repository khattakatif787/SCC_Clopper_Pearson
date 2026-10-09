from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
PROJECT_ROOT = Path.home() / 'SAC_Extension_Clopper_Upper_Bound_2026-09-04'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
VALIDATION_EXCEL = PROJECT_ROOT / 'SCC_SCCE_Validation_Results.xlsx'
TEST_EXCEL = PROJECT_ROOT / 'SCC_SCCE_Test_Results.xlsx'
CONFIDENCE_LEVELS = ['90', '95', '99', '99.9']
ALRS = ['0.01', '0.001', '0.0001']
DATASET_NAMES = {'Image_Data_Distribution 50_20_30/cctsdb_50_20_30': 'CCTSDB (50/20/30)', 'Image_Data_Distribution 50_20_30/med_50_20_30': 'Medical (50/20/30)', 'Image_Data_Distribution 50_20_30/tt100k_50_20_30': 'TT100K (50/20/30)', 'Image_Data_Distribution_50_30_20/_CCTSDB': 'CCTSDB (50/30/20)', 'Image_Data_Distribution_50_30_20/_Medical_dataset': 'Medical (50/30/20)', 'Image_Data_Distribution_50_30_20/_tt100k': 'TT100K (50/30/20)', 'Tabular_validation_test_files/BlackBlaze2019': 'BackBlaze2019', 'Tabular_validation_test_files/BlackBlaze2021': 'BackBlaze2021', 'Tabular_validation_test_files/BlackBlaze2023': 'BackBlaze2023', 'Tabular_validation_test_files/CICIDS18': 'CICIDS18', 'Tabular_validation_test_files/UNSW': 'UNSW', 'Tabular_validation_test_files/arancino': 'ARANCINO'}
DATASET_ORDER = ['CCTSDB (50/20/30)', 'Medical (50/20/30)', 'TT100K (50/20/30)', 'CCTSDB (50/30/20)', 'Medical (50/30/20)', 'TT100K (50/30/20)', 'BackBlaze2019', 'BackBlaze2021', 'BackBlaze2023', 'CICIDS18', 'UNSW', 'ARANCINO']
HEADERS = ['Dataset', 'ALR', 'Configuration', 'SCC / SCCE Order', 'Correct', 'Misclassified', 'Rejected', 'Total']


def discover_datasets(confidence):
    confidence_root = OUTPUT_ROOT / confidence
    lookup = {}
    if not confidence_root.exists():
        return lookup
    for scc_output_folder in confidence_root.rglob('SCC_outputs'):
        if not scc_output_folder.is_dir():
            continue
        dataset_dir = scc_output_folder.parent
        relative_name = dataset_dir.relative_to(confidence_root).as_posix()
        display_name = DATASET_NAMES.get(relative_name, relative_name)
        lookup[display_name] = dataset_dir
    return lookup


def get_best_single_scc(dataset_dir, alr):
    metrics_dir = dataset_dir / 'SCC_metrics'
    val_file = metrics_dir / f'SCC_METRICS_VALIDATION_ALR_{alr}.csv'
    test_file = metrics_dir / f'SCC_METRICS_TEST_ALR_{alr}.csv'
    if not val_file.exists() or not test_file.exists():
        return None
    val_df = pd.read_csv(val_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)
    if val_df.empty:
        return None
    val_df['Correct'] = pd.to_numeric(val_df['Correct'], errors='raise')
    best_index = val_df['Correct'].idxmax()
    best_val = val_df.loc[best_index]
    scc_name = str(best_val['SCC'])
    test_match = test_df[test_df['SCC'].astype(str) == scc_name]
    if len(test_match) != 1:
        raise RuntimeError(f'\nCould not uniquely locate validation-selected SCC on TEST.\nSCC: {scc_name}\nFile: {test_file}\n')
    best_test = test_match.iloc[0]
    return {'order': scc_name, 'validation': {'Correct': int(best_val['Correct']), 'Misclassified': int(best_val['Misclassified']), 'Rejected': int(best_val['Rejected']), 'Total': int(best_val['Total'])}, 'test': {'Correct': int(best_test['Correct']), 'Misclassified': int(best_test['Misclassified']), 'Rejected': int(best_test['Rejected']), 'Total': int(best_test['Total'])}}


def get_scce(dataset_dir, alr, k):
    result_dir = dataset_dir / 'SCCE_exact_ordered' / f'k_{k}' / f'individual_ALR_{alr}'
    order_file = result_dir / 'SELECTED_ORDER.csv'
    val_file = result_dir / 'SCCE_METRICS_VALIDATION.csv'
    test_file = result_dir / 'SCCE_METRICS_TEST.csv'
    if not order_file.exists() or not val_file.exists() or (not test_file.exists()):
        return None
    order_df = pd.read_csv(order_file, low_memory=False)
    val_df = pd.read_csv(val_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)
    if len(order_df) != k:
        raise RuntimeError(f'\nExpected exactly {k} SCCs but found {len(order_df)} in:\n{order_file}\n')
    order_df = order_df.sort_values('Position')
    ensemble_order = ' -> '.join(order_df['SCC'].astype(str).tolist())
    val = val_df.iloc[0]
    test = test_df.iloc[0]
    return {'order': ensemble_order, 'validation': {'Correct': int(val['Correct']), 'Misclassified': int(val['Misclassified']), 'Rejected': int(val['Rejected']), 'Total': int(val['Total'])}, 'test': {'Correct': int(test['Correct']), 'Misclassified': int(test['Misclassified']), 'Rejected': int(test['Rejected']), 'Total': int(test['Total'])}}


def make_row(dataset, alr, configuration, result, split):
    if result is None:
        return [dataset, alr, configuration, 'N/A - no qualified SCC', None, None, None, None]
    metrics = result[split]
    return [dataset, alr, configuration, result['order'], metrics['Correct'], metrics['Misclassified'], metrics['Rejected'], metrics['Total']]


def build_rows(confidence, split):
    dataset_lookup = discover_datasets(confidence)
    rows = []
    for dataset_name in DATASET_ORDER:
        if dataset_name not in dataset_lookup:
            print(f'[WARNING] Missing dataset {dataset_name} at {confidence}%')
            continue
        dataset_dir = dataset_lookup[dataset_name]
        for alr in ALRS:
            single = get_best_single_scc(dataset_dir, alr)
            rows.append(make_row(dataset_name, alr, 'Single SCC', single, split))
            scce2 = get_scce(dataset_dir, alr, 2)
            rows.append(make_row(dataset_name, alr, 'SCCE-2', scce2, split))
            scce3 = get_scce(dataset_dir, alr, 3)
            rows.append(make_row(dataset_name, alr, 'SCCE-3', scce3, split))
    return rows


def format_worksheet(ws):
    header_fill = PatternFill(fill_type='solid', fgColor='1F4E78')
    header_font = Font(color='FFFFFF', bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.row_dimensions[1].height = 32
    single_fill = PatternFill(fill_type='solid', fgColor='E2F0D9')
    scce2_fill = PatternFill(fill_type='solid', fgColor='DDEBF7')
    scce3_fill = PatternFill(fill_type='solid', fgColor='FFF2CC')
    na_fill = PatternFill(fill_type='solid', fgColor='E7E6E6')
    thin = Side(style='thin', color='D9D9D9')
    medium = Side(style='medium', color='5B9BD5')
    for row in range(2, ws.max_row + 1):
        config = ws.cell(row=row, column=3).value
        order = str(ws.cell(row=row, column=4).value or '')
        if 'N/A' in order:
            fill = na_fill
        elif config == 'Single SCC':
            fill = single_fill
        elif config == 'SCCE-2':
            fill = scce2_fill
        else:
            fill = scce3_fill
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.fill = fill
            cell.alignment = Alignment(vertical='center', wrap_text=True)
            cell.border = Border(bottom=thin)
        for col in [2, 3, 5, 6, 7, 8]:
            ws.cell(row=row, column=col).alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        if config == 'SCCE-3':
            for col in range(1, ws.max_column + 1):
                ws.cell(row=row, column=col).border = Border(bottom=medium)
    ws.column_dimensions['A'].width = 24
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 16
    ws.column_dimensions['D'].width = 75
    ws.column_dimensions['E'].width = 14
    ws.column_dimensions['F'].width = 18
    ws.column_dimensions['G'].width = 14
    ws.column_dimensions['H'].width = 14
    for row in range(2, ws.max_row + 1):
        ws.row_dimensions[row].height = 42
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions


def create_workbook(output_path, split):
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    for confidence in CONFIDENCE_LEVELS:
        print(f'Creating {split.upper()} sheet {confidence}% ...')
        ws = wb.create_sheet(title=f'{confidence}%')
        ws.append(HEADERS)
        rows = build_rows(confidence, split)
        for row in rows:
            ws.append(row)
        format_worksheet(ws)
        print(f'  Rows: {len(rows)}')
    wb.save(output_path)
    print(f'\nSaved:\n{output_path}')


def main():
    print('=' * 90)
    print('CREATING VALIDATION WORKBOOK')
    print('=' * 90)
    create_workbook(VALIDATION_EXCEL, 'validation')
    print('\n' + '=' * 90)
    print('CREATING TEST WORKBOOK')
    print('=' * 90)
    create_workbook(TEST_EXCEL, 'test')
    print('\n' + '=' * 90)
    print('ALL EXCEL RESULTS CREATED')
    print('\nValidation:')
    print(VALIDATION_EXCEL)
    print('\nTest:')
    print(TEST_EXCEL)
    print('=' * 90)

if __name__ == '__main__':
    main()
