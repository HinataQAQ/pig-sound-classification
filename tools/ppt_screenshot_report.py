from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def read_csv(path):
    p = REPORTS / path
    if not p.exists():
        print(f'[MISSING] {p}')
        return None
    return pd.read_csv(p)


def fmt(df, cols=None, sort_col=None, top=None):
    if df is None or len(df) == 0:
        print('(no data)')
        return
    d = df.copy()
    if cols:
        for c in cols:
            if c not in d.columns:
                d[c] = ''
        d = d[cols]
    if sort_col and sort_col in d.columns:
        d[sort_col] = pd.to_numeric(d[sort_col], errors='coerce')
        d = d.sort_values(sort_col, ascending=False)
    if top:
        d = d.head(top)
    print(d.to_string(index=False))


print('\n' + '='*92)
print('[A] MCTAFD / Gate-MCTAFD balanced CV5 summary')
print('='*92)
df = read_csv('cv5_variant_summary.csv')
if df is not None:
    keep = ['logmel', 'mctafd_no_se_direct', 'mctafd_se', 'mctafd_no_se_gated']
    df = df[df['model'].isin(keep)]
    fmt(df, ['model','n','mean_macro_f1','std_macro_f1','min_macro_f1','max_macro_f1','mean_gate'], 'mean_macro_f1')

print('\n' + '='*92)
print('[B] CV10 / pretrain check: Gate-MCTAFD is not the final clean mainline')
print('='*92)
df10 = read_csv('cv10_variant_summary.csv')
if df10 is not None:
    fmt(df10[df10['model'].isin(['logmel','mctafd_no_se_gated'])], ['model','n','mean_macro_f1','std_macro_f1','min_macro_f1','max_macro_f1','mean_gate'], 'mean_macro_f1')
pre = read_csv('cv10_pretrain_variant_summary.csv')
if pre is not None:
    fmt(pre, ['model','n','mean_macro_f1','std_macro_f1','min_macro_f1','max_macro_f1','mean_gate'], 'mean_macro_f1')

print('\n' + '='*92)
print('[C] cap3x + preprocessing summary')
print('='*92)
exp = read_csv('cv5_expanded_variant_summary.csv')
if exp is not None:
    cap3 = exp[exp['protocol'].astype(str).str.lower().eq('cap3x')]
    fmt(cap3, ['protocol','model','n','mean_macro_f1','std_macro_f1','min_macro_f1','max_macro_f1','mean_gate'], 'mean_macro_f1')

print('\n' + '='*92)
print('[D] cap3x confusion summary')
print('='*92)
cm = read_csv('cv5_cap3x_logmel_confusion_summary.csv')
if cm is not None:
    fmt(cm, ['model','n_runs','macro_f1_from_cm','feeding_f1','stress_f1','feeding_to_stress','stress_to_feeding'])

print('\n' + '='*92)
print('[E] Recommended screenshot commands')
print('='*92)
print('PyCharm Run Config: Script path = tools/ppt_screenshot_report.py')
print('Working directory = C:\\py\\pigsound\\pig-sound-classification')
print('Environment = PYTHONNOUSERSITE=1')
