"""
Batch evaluation: runs triggered evaluation on all files in data_organized/,
trains per-subject models, and plots comparative statistics.
"""

import json
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.predictor.core.mi_pipeline import load_epochs, build_pipeline, STUDY_EVENTS
from src.predictor.core.errp_pipeline import load_errp_epochs, build_pipeline as build_errp_pipeline
from run_offline_e2e import run_triggered_evaluation

import joblib

DATA_DIR = Path("D:/STUDIA/ZPB-antigravity/data_organized")
MODELS_DIR = Path("models/batch")
RESULTS_DIR = Path("results/batch")


def parse_filename(path: Path) -> dict:
    match = re.match(r"(subject\d+)_(ses\d+)_(run\d+)_", path.name)
    if match:
        return {"subject": match.group(1), "session": match.group(2), "run": match.group(3)}
    return {}


def get_all_files() -> dict[str, list[Path]]:
    """Group files by subject."""
    files = sorted(DATA_DIR.glob("*.fif"))
    grouped = defaultdict(list)
    for f in files:
        info = parse_filename(f)
        if info:
            grouped[info["subject"]].append(f)
    return dict(grouped)


def train_models(subject: str, train_files: list[Path]) -> tuple[Path, Path]:
    """Train MI and ErrP models for a subject, return model paths."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mi_path = MODELS_DIR / f"{subject}_mi.joblib"
    errp_path = MODELS_DIR / f"{subject}_errp.joblib"

    if not mi_path.exists():
        print(f"  Training MI model for {subject} ({len(train_files)} files)...")
        epochs = load_epochs([str(f) for f in train_files])
        X = epochs.get_data()
        y = epochs.events[:, -1]
        pipeline = build_pipeline()
        pipeline.fit(X, y)
        joblib.dump(pipeline, mi_path)
        print(f"  MI model saved: {mi_path} ({len(y)} epochs)")
    else:
        print(f"  MI model exists: {mi_path}")

    if not errp_path.exists():
        print(f"  Training ErrP model for {subject}...")
        try:
            epochs = load_errp_epochs([str(f) for f in train_files])
            X = epochs.get_data()
            y = epochs.events[:, -1]
            if len(np.unique(y)) >= 2:
                pipeline = build_errp_pipeline()
                pipeline.fit(X, y)
                joblib.dump(pipeline, errp_path)
                print(f"  ErrP model saved: {errp_path} ({len(y)} epochs)")
            else:
                print(f"  ErrP: only one class found, skipping")
                return mi_path, None
        except Exception as e:
            print(f"  ErrP training failed: {e}")
            return mi_path, None
    else:
        print(f"  ErrP model exists: {errp_path}")

    return mi_path, errp_path


def evaluate_file(data_file: Path, mi_model: Path, errp_model: Path,
                  strategy: str = "baseline", speed: float = 10.0) -> dict | None:
    """Run triggered evaluation on a single file."""
    if errp_model is None:
        print(f"  Skipping {data_file.name} (no ErrP model)")
        return None

    try:
        result = run_triggered_evaluation(
            str(data_file), str(mi_model), str(errp_model),
            strategy_name=strategy, speed=speed,
        )
        return result
    except Exception as e:
        print(f"  ERROR evaluating {data_file.name}: {e}")
        return None


def plot_results(all_results: dict[str, list[dict]], output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Per-file accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = []
    accuracies = []
    colors = []
    subject_colors = {}
    color_cycle = plt.cm.Set2.colors

    for i, (subject, results) in enumerate(sorted(all_results.items())):
        color = color_cycle[i % len(color_cycle)]
        subject_colors[subject] = color
        for r in results:
            labels.append(r["file_label"])
            accuracies.append(r["mi_accuracy"] * 100)
            colors.append(color)

    bars = ax.bar(range(len(labels)), accuracies, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MI Accuracy (%)")
    ax.set_title("Triggered Evaluation — MI Accuracy per Recording")
    ax.axhline(25, color="red", linestyle="--", linewidth=1, label="Chance (25%)")
    ax.set_ylim(0, 100)
    ax.legend(
        handles=[Patch(facecolor=c, label=s) for s, c in subject_colors.items()] +
                [plt.Line2D([0], [0], color="red", linestyle="--", label="Chance")],
        loc="upper right",
    )
    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_per_file.png", dpi=150)
    plt.close(fig)

    # --- 2. Per-class accuracy heatmap ---
    all_classes = ["REST", "LEFT", "RIGHT", "FORWARD"]
    file_labels = []
    class_acc_matrix = []

    for subject, results in sorted(all_results.items()):
        for r in results:
            file_labels.append(r["file_label"])
            row = [r["per_class_accuracy"].get(c, 0.0) for c in all_classes]
            class_acc_matrix.append(row)

    matrix = np.array(class_acc_matrix)
    fig, ax = plt.subplots(figsize=(8, max(6, len(file_labels) * 0.4)))
    im = ax.imshow(matrix * 100, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(all_classes)))
    ax.set_xticklabels(all_classes)
    ax.set_yticks(range(len(file_labels)))
    ax.set_yticklabels(file_labels, fontsize=8)
    ax.set_title("Per-Class Accuracy (%)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j] * 100
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if val < 40 else "black")

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    fig.savefig(output_dir / "per_class_heatmap.png", dpi=150)
    plt.close(fig)

    # --- 3. Subject summary ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy per subject (mean ± std)
    subjects = sorted(all_results.keys())
    means = []
    stds = []
    for s in subjects:
        accs = [r["mi_accuracy"] for r in all_results[s]]
        means.append(np.mean(accs) * 100)
        stds.append(np.std(accs) * 100)

    ax = axes[0]
    x = range(len(subjects))
    ax.bar(x, means, yerr=stds, color=[subject_colors[s] for s in subjects],
           edgecolor="black", linewidth=0.5, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel("MI Accuracy (%)")
    ax.set_title("Mean Accuracy per Subject")
    ax.axhline(25, color="red", linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)

    # Confidence per subject
    ax = axes[1]
    conf_means = []
    conf_stds = []
    for s in subjects:
        confs = [r["avg_confidence"] for r in all_results[s]]
        conf_means.append(np.mean(confs))
        conf_stds.append(np.std(confs))

    ax.bar(x, conf_means, yerr=conf_stds, color=[subject_colors[s] for s in subjects],
           edgecolor="black", linewidth=0.5, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel("Avg Confidence")
    ax.set_title("Mean Classifier Confidence per Subject")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / "subject_summary.png", dpi=150)
    plt.close(fig)

    # --- 4. Confusion matrix aggregate ---
    confusion_total = defaultdict(int)
    for subject, results in all_results.items():
        for r in results:
            for k, v in r.get("confusion", {}).items():
                confusion_total[k] += v

    conf_matrix = np.zeros((4, 4))
    for key, count in confusion_total.items():
        parts = key.split("→")
        if len(parts) == 2:
            true_idx = all_classes.index(parts[0]) if parts[0] in all_classes else -1
            pred_idx = all_classes.index(parts[1]) if parts[1] in all_classes else -1
            if true_idx >= 0 and pred_idx >= 0:
                conf_matrix[true_idx, pred_idx] = count

    # Normalize rows
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_norm = conf_matrix / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(conf_norm * 100, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(4))
    ax.set_xticklabels(all_classes)
    ax.set_yticks(range(4))
    ax.set_yticklabels(all_classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Aggregate Confusion Matrix (normalized %)")

    for i in range(4):
        for j in range(4):
            val = conf_norm[i, j] * 100
            count = int(conf_matrix[i, j])
            ax.text(j, i, f"{val:.0f}%\n({count})", ha="center", va="center",
                    fontsize=9, color="white" if val > 50 else "black")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {output_dir}/")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    grouped_files = get_all_files()
    print(f"Found {sum(len(v) for v in grouped_files.values())} files across {len(grouped_files)} subjects\n")

    all_results: dict[str, list[dict]] = {}

    for subject, files in sorted(grouped_files.items()):
        print(f"\n{'='*50}")
        print(f"Subject: {subject} ({len(files)} recordings)")
        print(f"{'='*50}")

        mi_model, errp_model = train_models(subject, files)

        subject_results = []
        for data_file in files:
            info = parse_filename(data_file)
            file_label = f"{info['subject']}_{info['session']}_{info['run']}"
            print(f"\n  Evaluating: {file_label}")

            result = evaluate_file(data_file, mi_model, errp_model, speed=10.0)
            if result and "error" not in result:
                result["file_label"] = file_label
                result["file_path"] = str(data_file)
                subject_results.append(result)

                # Save individual result
                result_path = RESULTS_DIR / f"{file_label}.json"
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)

        if subject_results:
            all_results[subject] = subject_results

    # Save aggregate
    summary_path = RESULTS_DIR / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {summary_path}")

    # Plot
    if all_results:
        plot_results(all_results, RESULTS_DIR)


if __name__ == "__main__":
    main()
