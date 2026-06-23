from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prototype_model_adapter import cosine_similarities, load_prototype_bundle, read_json, require_file, write_json


def safe_name(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(label).strip().lower())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot prototype similarity matrices and train Log-Mel mean/std atlas maps."
    )
    ap.add_argument("--prototype_bundle", required=True)
    ap.add_argument("--out_dir", required=True, help="Phase root directory; atlas/ is created inside it.")
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def prepare_atlas_dir(root: Path, allow_overwrite: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / "atlas"
    if out_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing atlas directory: {out_dir}")
    (out_dir / "main").mkdir(parents=True, exist_ok=True)
    (out_dir / "auxiliary").mkdir(parents=True, exist_ok=True)
    return out_dir


def assert_new_file(path: Path, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def plot_matrix(matrix: np.ndarray, labels: list[str], title: str, out_png: Path, allow_overwrite: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert_new_file(out_png, allow_overwrite)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.8), max(4, len(labels) * 0.7)))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_map(values: np.ndarray, title: str, out_png: Path, allow_overwrite: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert_new_file(out_png, allow_overwrite)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(values, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("time frame")
    ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def sample_roles(samples: pd.DataFrame, label_col: str, dist_col: str, margin_col: str, labels: list[str], top_n: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    keep_cols = [
        c
        for c in [
            "path",
            "label",
            "subtype",
            "source_id",
            "md5",
            label_col,
            dist_col,
            margin_col,
        ]
        if c in samples.columns
    ]
    for label in labels:
        part = samples[samples[label_col].astype(str) == label].copy()
        if part.empty:
            continue
        for role, ordered in [
            ("closest_to_prototype", part.sort_values(dist_col, ascending=True).head(top_n)),
            ("farthest_from_prototype", part.sort_values(dist_col, ascending=False).head(top_n)),
            ("most_ambiguous", part.sort_values(margin_col, ascending=True).head(top_n)),
        ]:
            out = ordered[keep_cols].copy()
            out.insert(0, "sample_role", role)
            out.insert(1, "rank", range(1, len(out) + 1))
            rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["sample_role", "rank", *keep_cols])
    return pd.concat(rows, ignore_index=True)


def flatten_distance_stats(section: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, payload in section["classes"].items():
        row: dict[str, Any] = {"label": label, "count": payload["count"]}
        for distance_name in ["cosine_similarity", "cosine_distance", "euclidean_distance"]:
            for stat, value in payload[distance_name].items():
                row[f"{distance_name}_{stat}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    bundle_path = require_file(args.prototype_bundle, "prototype bundle")
    out_dir = prepare_atlas_dir(Path(args.out_dir), args.allow_overwrite)
    main_dir = out_dir / "main"
    aux_dir = out_dir / "auxiliary"
    bundle = load_prototype_bundle(bundle_path)
    metadata = bundle["metadata"]

    main_labels = bundle["main_labels"]
    aux_labels = bundle["aux_labels"]
    main_prototypes = np.asarray(bundle["main_prototypes"], dtype=np.float32)
    aux_prototypes = np.asarray(bundle["aux_prototypes"], dtype=np.float32)

    main_sim = cosine_similarities(main_prototypes, main_prototypes)
    aux_sim = cosine_similarities(aux_prototypes, aux_prototypes)
    pd.DataFrame(main_sim, index=main_labels, columns=main_labels).to_csv(
        out_dir / "main_prototype_cosine_similarity.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(aux_sim, index=aux_labels, columns=aux_labels).to_csv(
        out_dir / "auxiliary_prototype_cosine_similarity.csv",
        encoding="utf-8-sig",
    )
    plot_matrix(main_sim, main_labels, "Main prototype cosine similarity", out_dir / "prototype_similarity_matrix.png", args.allow_overwrite)
    plot_matrix(main_sim, main_labels, "Main prototype cosine similarity", out_dir / "main_prototype_similarity_matrix.png", args.allow_overwrite)
    plot_matrix(aux_sim, aux_labels, "Auxiliary prototype cosine similarity", out_dir / "auxiliary_prototype_similarity_matrix.png", args.allow_overwrite)

    pd.DataFrame(
        {
            "label": main_labels,
            "prototype_norm": np.linalg.norm(main_prototypes, axis=1),
            "count": np.asarray(bundle["main_counts"], dtype=np.int64),
        }
    ).to_csv(main_dir / "prototype_norms.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "label": aux_labels,
            "prototype_norm": np.linalg.norm(aux_prototypes, axis=1),
            "count": np.asarray(bundle["aux_counts"], dtype=np.int64),
        }
    ).to_csv(aux_dir / "prototype_norms.csv", index=False, encoding="utf-8-sig")

    if "feature_main_mean" not in bundle or "feature_main_std" not in bundle:
        raise RuntimeError("Prototype bundle does not contain feature_main_mean/std arrays.")
    if "feature_aux_mean" not in bundle or "feature_aux_std" not in bundle:
        raise RuntimeError("Prototype bundle does not contain feature_aux_mean/std arrays.")

    main_mean = np.asarray(bundle["feature_main_mean"], dtype=np.float32)
    main_std = np.asarray(bundle["feature_main_std"], dtype=np.float32)
    aux_mean = np.asarray(bundle["feature_aux_mean"], dtype=np.float32)
    aux_std = np.asarray(bundle["feature_aux_std"], dtype=np.float32)

    for idx, label in enumerate(main_labels):
        mean_map = main_mean[idx, 0] if main_mean.ndim == 4 else main_mean[idx]
        std_map = main_std[idx, 0] if main_std.ndim == 4 else main_std[idx]
        plot_map(mean_map, f"{label} train Log-Mel mean", main_dir / f"logmel_mean_{safe_name(label)}.png", args.allow_overwrite)
        plot_map(std_map, f"{label} train Log-Mel std", main_dir / f"logmel_std_{safe_name(label)}.png", args.allow_overwrite)

    for idx, label in enumerate(aux_labels):
        mean_map = aux_mean[idx, 0] if aux_mean.ndim == 4 else aux_mean[idx]
        std_map = aux_std[idx, 0] if aux_std.ndim == 4 else aux_std[idx]
        plot_map(mean_map, f"{label} train Log-Mel mean", aux_dir / f"logmel_mean_{safe_name(label)}.png", args.allow_overwrite)
        plot_map(std_map, f"{label} train Log-Mel std", aux_dir / f"logmel_std_{safe_name(label)}.png", args.allow_overwrite)

    distance_stats: dict[str, Any] | None = None
    if metadata.get("distance_statistics_path"):
        distance_stats = read_json(metadata["distance_statistics_path"])
        flatten_distance_stats(distance_stats["main"]).to_csv(main_dir / "distance_statistics.csv", index=False, encoding="utf-8-sig")
        flatten_distance_stats(distance_stats["auxiliary"]).to_csv(aux_dir / "distance_statistics.csv", index=False, encoding="utf-8-sig")

    artifacts_dir = Path(bundle_path).resolve().parent
    main_samples_path = artifacts_dir / "train_main_prototype_samples.csv"
    aux_samples_path = artifacts_dir / "train_aux_prototype_samples.csv"
    if main_samples_path.exists():
        main_samples = pd.read_csv(main_samples_path)
        sample_roles(
            main_samples,
            "main_label",
            "main_own_cosine_distance",
            "main_confusion_margin",
            main_labels,
            args.top_n,
        ).to_csv(main_dir / "prototype_sample_atlas.csv", index=False, encoding="utf-8-sig")
    if aux_samples_path.exists():
        aux_samples = pd.read_csv(aux_samples_path)
        sample_roles(
            aux_samples,
            "aux_label",
            "aux_own_cosine_distance",
            "aux_confusion_margin",
            aux_labels,
            args.top_n,
        ).to_csv(aux_dir / "prototype_sample_atlas.csv", index=False, encoding="utf-8-sig")

    feeding_stress: dict[str, Any] = {
        "labels": ["feeding", "stress_vocal"],
        "main_similarity": None,
        "main_cosine_distance": None,
        "note": "Distances are computed on L2-normalized prototype vectors; Euclidean distance is determined by cosine similarity.",
    }
    if "feeding" in main_labels and "stress_vocal" in main_labels:
        i = main_labels.index("feeding")
        j = main_labels.index("stress_vocal")
        sim = float(main_sim[i, j])
        feeding_stress["main_similarity"] = sim
        feeding_stress["main_cosine_distance"] = float(1.0 - sim)
        feeding_stress["main_euclidean_distance"] = float(np.sqrt(max(0.0, 2.0 - 2.0 * sim)))
        if distance_stats:
            feeding_stress["feeding_distance_stats"] = distance_stats["main"]["classes"]["feeding"]
            feeding_stress["stress_vocal_distance_stats"] = distance_stats["main"]["classes"]["stress_vocal"]
    write_json(out_dir / "feeding_stress_similarity_analysis.json", feeding_stress)

    atlas_manifest = {
        "artifact_type": "hier_acoustic_prototype_atlas",
        "prototype_bundle": str(bundle_path.resolve()),
        "fold": metadata.get("fold"),
        "seed": metadata.get("seed"),
        "main_labels": main_labels,
        "aux_labels": aux_labels,
        "audio_copied": False,
        "outputs": {
            "main_dir": str(main_dir.resolve()),
            "auxiliary_dir": str(aux_dir.resolve()),
            "main_similarity_matrix": str((out_dir / "main_prototype_similarity_matrix.png").resolve()),
            "auxiliary_similarity_matrix": str((out_dir / "auxiliary_prototype_similarity_matrix.png").resolve()),
            "feeding_stress_analysis": str((out_dir / "feeding_stress_similarity_analysis.json").resolve()),
        },
    }
    write_json(out_dir / "atlas_manifest.json", atlas_manifest)

    print(f"[OK] wrote atlas -> {out_dir}")


if __name__ == "__main__":
    main()
