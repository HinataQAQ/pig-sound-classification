from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


FEATURE_MODEL_NAMES = {
    "logmel",
    "logmel_hpf",
    "logmel_sg",
    "logmel_pcen",
    "logmel_sg_pcen",
    "logmel_dual",
    "logmel_dual_pcen",
    "mel_mfcc",
    "mel_mfcc_dyn",
    "mel_mfcc_dyn_flux",
    "mctafd",
}


def parse_fold(exp: str):
    m = re.search(r"fold(\d+)", exp)
    return int(m.group(1)) if m else None


def parse_protocol(exp: str):
    """
    Examples:
      cv5_expanded_cap2x_fold0_logmel_seed3407
      cv5_expanded_cap2x_fold0_mctafd_no_se_gated_seed3407
      cv5_expanded_cap3x_fold0_logmel_seed3407
      cv5_expanded_cap3x_fold0_logmel_sg_seed3407
      cv5_expanded_cap3x_fold0_logmel_pcen_seed3407
      cv5_expanded_cap3x_fold0_logmel_dual_seed3407
      cv5_expanded_all_fold0_logmel_seed3407
      cv5_expanded_alltrain_fold0_logmel_seed3407
    """
    s = exp.lower()

    if "cap2x" in s:
        return "cap2x"

    if "cap3x" in s:
        return "cap3x"

    tokens = s.split("_")
    if "all" in tokens or "alltrain" in s or "expanded_all" in s:
        return "all"

    return "unknown"


def parse_model(exp: str, meta: dict | None = None):
    """
    Prefer explicit experiment-name parsing for special variants.
    Then fall back to summary.json['feature_mode'].

    This prevents new names such as logmel_sg/logmel_dual from being
    collapsed into plain logmel.
    """
    s = exp.lower()
    meta = meta or {}
    feature_mode = str(meta.get("feature_mode", "")).strip().lower()
    use_se = meta.get("use_se")

    # Order matters. More specific experiment names first.
    if "_mctafd_no_se_gated_" in s:
        return "mctafd_no_se_gated"

    if "_mctafd_no_se_" in s:
        return "mctafd_no_se"

    if "_mctafd_se_" in s:
        return "mctafd_se"

    if "_logmel_specaug_light_" in s:
        return "logmel_specaug_light"

    if "_logmel_specaug_" in s:
        return "logmel_specaug"

    if "_logmel_dur2_attn_" in s:
        return "logmel_dur2_attn"

    if "_logmel_dur2_" in s:
        return "logmel_dur2"

    if "_logmel_dur3_" in s:
        return "logmel_dur3"

    if "_logmel_bilstm_" in s:
        return "logmel_bilstm"

    if "_logmel_bigru_" in s:
        return "logmel_bigru"

    # New preprocessing feature modes. Match before plain logmel.
    for name in [
        "logmel_dual_pcen",
        "logmel_dual",
        "logmel_sg_pcen",
        "logmel_pcen",
        "logmel_sg",
        "logmel_hpf",
    ]:
        if f"_{name}_" in s:
            return name

    # Use summary.json feature_mode when available. This is the safest path
    # for newly added features.
    if feature_mode in FEATURE_MODEL_NAMES:
        if feature_mode == "mctafd":
            if use_se is False:
                return "mctafd_no_se"
            if use_se is True:
                return "mctafd_se"
            return "mctafd"
        return feature_mode

    # Last fallback by experiment name.
    if "_logmel_" in s:
        return "logmel"

    if "_mel_mfcc_dyn_flux_" in s:
        return "mel_mfcc_dyn_flux"

    if "_mel_mfcc_dyn_" in s:
        return "mel_mfcc_dyn"

    if "_mel_mfcc_" in s:
        return "mel_mfcc"

    if "_mctafd_" in s:
        return "mctafd"

    return None


rows = []

for summary_path in REPORTS.glob("cv5_expanded*/summary.json"):
    exp = summary_path.parent.name

    with open(summary_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    model = parse_model(exp, m)

    if model is None:
        continue

    gate = m.get("aux_gate")
    gate_mean = None
    gate_max = None

    if isinstance(gate, list) and len(gate) > 0:
        gate_values = [float(x) for x in gate]
        gate_mean = sum(gate_values) / len(gate_values)
        gate_max = max(gate_values)

    rows.append({
        "experiment": exp,
        "protocol": parse_protocol(exp),
        "model": model,
        "fold": parse_fold(exp),
        "seed": m.get("seed"),

        "feature_mode": m.get("feature_mode"),
        "use_se": m.get("use_se"),

        "n_mels": m.get("n_mels"),
        "n_mfcc": m.get("n_mfcc"),
        "fmin": m.get("fmin"),
        "fmax": m.get("fmax"),
        "n_fft": m.get("n_fft"),
        "hop_length": m.get("hop_length"),
        "win_length": m.get("win_length"),
        "sr": m.get("sr"),
        "dur_s": m.get("dur_s"),

        "fusion_type": m.get("fusion_type"),
        "aux_gate_init": m.get("aux_gate_init"),
        "gate_mean": gate_mean,
        "gate_max": gate_max,

        "boundary_loss_weight": m.get("boundary_loss_weight"),
        "boundary_warmup_epochs": m.get("boundary_warmup_epochs"),

        "best_epoch": m.get("best_epoch"),
        "best_val_macro_f1": m.get("best_val_macro_f1"),
        "test_acc": m.get("test_acc"),
        "test_macro_f1": m.get("test_macro_f1"),
    })


df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No expanded CV5 summary.json found under reports/cv5_expanded*/summary.json.")


numeric_cols = [
    "fold",
    "seed",
    "n_mels",
    "n_mfcc",
    "fmin",
    "fmax",
    "n_fft",
    "hop_length",
    "win_length",
    "sr",
    "dur_s",
    "aux_gate_init",
    "gate_mean",
    "gate_max",
    "boundary_loss_weight",
    "boundary_warmup_epochs",
    "best_epoch",
    "best_val_macro_f1",
    "test_acc",
    "test_macro_f1",
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# Remove broken rows without test score.
df = df[df["test_macro_f1"].notna()].copy()


protocol_order = {
    "cap2x": 0,
    "cap3x": 1,
    "all": 2,
    "unknown": 99,
}

model_order = {
    "logmel": 0,
    "logmel_bigru": 0,
    "logmel_bilstm": 1,
    "logmel_hpf": 1,
    "logmel_sg": 2,
    "logmel_pcen": 3,
    "logmel_sg_pcen": 4,
    "logmel_dual": 5,
    "logmel_dual_pcen": 6,
    "mel_mfcc": 20,
    "mel_mfcc_dyn": 21,
    "mel_mfcc_dyn_flux": 22,
    "mctafd": 30,
    "mctafd_se": 31,
    "mctafd_no_se": 32,
    "mctafd_no_se_gated": 33,
    "logmel_specaug": 1,
    "logmel_dur2": 1,
    "logmel_dur3": 2,
    "logmel_dur2_attn": 0,
}


df["protocol_order"] = df["protocol"].map(protocol_order).fillna(99)
df["model_order"] = df["model"].map(model_order).fillna(999)

df = df.sort_values([
    "protocol_order",
    "protocol",
    "model_order",
    "model",
    "seed",
    "fold",
    "experiment",
]).drop(columns=["protocol_order", "model_order"])


runs_out = REPORTS / "cv5_expanded_variant_runs.csv"
df.to_csv(runs_out, index=False, encoding="utf-8-sig")


summary = (
    df.groupby(["protocol", "model"])
      .agg(
          n=("test_macro_f1", "count"),
          mean_macro_f1=("test_macro_f1", "mean"),
          std_macro_f1=("test_macro_f1", "std"),
          min_macro_f1=("test_macro_f1", "min"),
          max_macro_f1=("test_macro_f1", "max"),
          mean_best_val=("best_val_macro_f1", "mean"),
          mean_test_acc=("test_acc", "mean"),
          mean_gate=("gate_mean", "mean"),
          max_gate=("gate_max", "max"),
      )
      .reset_index()
)

summary["protocol_order"] = summary["protocol"].map(protocol_order).fillna(99)
summary["model_order"] = summary["model"].map(model_order).fillna(999)

summary = summary.sort_values([
    "protocol_order",
    "mean_macro_f1",
], ascending=[True, False]).drop(columns=["protocol_order", "model_order"])


summary_out = REPORTS / "cv5_expanded_variant_summary.csv"
summary.to_csv(summary_out, index=False, encoding="utf-8-sig")


print("\n[EXPANDED CV5 RUNS]")
print(df.to_string(index=False))

print("\n[EXPANDED CV5 SUMMARY]")
print(summary.to_string(index=False))

print(f"\n[OK] wrote -> {runs_out}")
print(f"[OK] wrote -> {summary_out}")


print("\n[CHECK COUNTS]")
for _, r in summary.iterrows():
    protocol = r["protocol"]
    model = r["model"]
    n = int(r["n"])

    if protocol in {"cap2x", "cap3x"} and n not in {15, 25}:
        print(f"[WARN] {protocol}/{model}: n={n}. Expected 15 for 3 seeds x 5 folds or 25 for 5 seeds x 5 folds.")
    else:
        print(f"[OK] {protocol}/{model}: n={n}")