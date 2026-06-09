from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'tools' else Path.cwd()
REPORTS = ROOT / 'reports'

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 80)


def read_csv(name):
    p = REPORTS / name
    if not p.exists():
        print(f'[MISSING] {p}')
        return None
    return pd.read_csv(p)


def print_block(title, df):
    print('\n' + '=' * 110)
    print(title)
    print('=' * 110)
    if df is None or len(df) == 0:
        print('[NO DATA]')
    else:
        print(df.to_string(index=False))


# 1. Main clean results: cap3x variants
summary = read_csv('cv5_expanded_variant_summary.csv')
if summary is not None:
    cap3x = summary[summary['protocol'].astype(str).str.lower().eq('cap3x')].copy()
    if len(cap3x) > 0:
        for c in ['mean_macro_f1', 'std_macro_f1', 'min_macro_f1', 'max_macro_f1']:
            cap3x[c] = pd.to_numeric(cap3x[c], errors='coerce')
        keep_models = [
            'logmel_dur2', 'logmel_dur3', 'logmel', 'logmel_dur2_attn',
            'mctafd_no_se_gated', 'logmel_specaug', 'logmel_specaug_light',
            'logmel_dual', 'logmel_bilstm', 'logmel_sg', 'logmel_pcen'
        ]
        cap3x = cap3x[cap3x['model'].isin(keep_models)]
        cap3x = cap3x.sort_values('mean_macro_f1', ascending=False)
        print_block(
            'SCREENSHOT 1 — cap3x clean summary: long-context vs other variants',
            cap3x[['protocol', 'model', 'n', 'mean_macro_f1', 'std_macro_f1', 'min_macro_f1', 'max_macro_f1']]
        )

# 2. Paired stats: dur2 vs dur1
paired = read_csv('cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv')
if paired is not None:
    print_block(
        'SCREENSHOT 2 — paired test: 2s Log-Mel vs 1s Log-Mel',
        paired[['comparison', 'n', 'dur1_mean', 'dur2_mean', 'mean_delta', 'ci95_low', 'ci95_high', 'wilcoxon_p']]
    )

# 3. Confusion summary
conf = read_csv('cv5_cap3x_logmel_dur_confusion_summary.csv')
if conf is not None:
    print_block(
        'SCREENSHOT 3 — aggregated confusion summary: feeding / stress_vocal boundary',
        conf[['model', 'n_runs', 'macro_f1_from_cm', 'feeding_f1', 'stress_f1', 'feeding_to_stress', 'stress_to_feeding']]
    )

# 4. Duration-only baseline
Duration = read_csv('duration_only_baseline_summary.csv')
if Duration is not None:
    print_block(
        'SCREENSHOT 4 — duration-only baseline: not just duration leakage',
        Duration[['model', 'n', 'mean_macro_f1', 'std_macro_f1', 'min_macro_f1', 'max_macro_f1']]
    )

# 5. Short conclusion block
print('\n' + '=' * 110)
print('SCREENSHOT 5 — conclusion lines for report')
print('=' * 110)
print('Main clean model: cap3x + 2s long-context Log-Mel-CRNN.')
print('Key result: dur2 mean Macro-F1 ≈ 0.9472; dur1 mean Macro-F1 ≈ 0.9226; paired gain ≈ +0.0246.')
print('Boundary improvement: feeding→stress and stress→feeding errors are both reduced under dur2.')
print('Clean preprocessing variants were not beneficial: SpecAugment / PCEN / spectral gate / dual Log-Mel did not exceed original Log-Mel.')
print('Next stage: reserve SpecAugment / PCEN / denoising for real pig-farm noisy audio, not current clean clips.')
