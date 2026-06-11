"""
results_tracker.py
==================
Pneumonia Detection — Experiment Results Tracker & Plot Generator

Usage:
    python results_tracker.py                        # interactive menu
    python results_tracker.py --plot-only            # regenerate all plots from saved runs
    python results_tracker.py --list                 # print all runs as a table

Stores runs in:  results_log/runs.json
Saves plots to:  results_log/plots/
"""

import os
import sys
import json
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)

# ─────────────────────────────────────────────
#  Directories
# ─────────────────────────────────────────────
LOG_DIR   = Path("results_log")
PLOTS_DIR = LOG_DIR / "plots"
RUNS_FILE = LOG_DIR / "runs.json"

LOG_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
#  Colour palette  (consistent across all plots)
# ─────────────────────────────────────────────
PALETTE = {
    "Custom CNN":   "#378ADD",
    "ResNet152":    "#1D9E75",
    "EfficientNet": "#EF9F27",
    "DenseNet121":  "#D85A30",
    "Other":        "#7F77DD",
}
NORMAL_COLOR    = "#378ADD"
PNEUMONIA_COLOR = "#D85A30"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F8F8",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 — Run storage
# ═══════════════════════════════════════════════════════════════════

def load_runs() -> list[dict]:
    if RUNS_FILE.exists():
        with open(RUNS_FILE) as f:
            return json.load(f)
    return []


def save_runs(runs: list[dict]):
    with open(RUNS_FILE, "w") as f:
        json.dump(runs, f, indent=2)


def add_run(runs: list[dict], run: dict) -> list[dict]:
    run["id"]   = len(runs) + 1
    run["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    runs.append(run)
    save_runs(runs)
    print(f"\n✅  Run #{run['id']} — '{run['name']}' saved.")
    return runs


def delete_run(runs: list[dict], run_id: int) -> list[dict]:
    before = len(runs)
    runs = [r for r in runs if r["id"] != run_id]
    if len(runs) < before:
        save_runs(runs)
        print(f"🗑️   Run #{run_id} deleted.")
    else:
        print(f"⚠️   No run with id={run_id} found.")
    return runs


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 — Console display helpers
# ═══════════════════════════════════════════════════════════════════

def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "—"


def print_runs_table(runs: list[dict]):
    if not runs:
        print("  (no runs logged yet)")
        return

    header = f"{'ID':>3}  {'Date':<16}  {'Name':<28}  {'Model':<14}  "
    header += f"{'Acc%':>7}  {'AUC':>7}  {'F1':>7}  {'Loss':>7}  {'Epochs':>6}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    best_acc = max(r["accuracy"] for r in runs)
    for r in runs:
        marker = " ★" if r["accuracy"] == best_acc else ""
        row = (
            f"{r['id']:>3}  {r['date'][:16]:<16}  {r['name'][:28]:<28}  "
            f"{r['model'][:14]:<14}  "
            f"{_fmt(r['accuracy'], 2):>7}  "
            f"{_fmt(r.get('auc'), 4):>7}  "
            f"{_fmt(r.get('f1'), 4):>7}  "
            f"{_fmt(r.get('val_loss'), 4):>7}  "
            f"{str(r.get('epochs', '—')):>6}"
            f"{marker}"
        )
        print(row)
        if r.get("notes"):
            print(f"     ↳ {textwrap.shorten(r['notes'], 90)}")

    print("─" * len(header))
    print(f"  Total runs: {len(runs)}  |  Best accuracy: {best_acc:.2f}%\n")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 — Diagnostic plots
# ═══════════════════════════════════════════════════════════════════

# ── 3a. Confusion matrix ─────────────────────────────────────────

def plot_confusion_matrix(
    y_true, y_pred,
    run_name="",
    class_names=("NORMAL", "PNEUMONIA"),
    normalize=False,
    save=True,
):
    """
    Heatmap confusion matrix.

    Args:
        y_true       : list/array of ground-truth labels (0 or 1)
        y_pred       : list/array of predicted labels    (0 or 1)
        run_name     : label for the title and filename
        class_names  : tuple of class names
        normalize    : if True, show row-normalised percentages
        save         : save PNG to results_log/plots/
    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, title_suffix = ".2%", " (normalised)"
    else:
        cm_display = cm
        fmt, title_suffix = "d", ""

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="white", ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"Confusion matrix{title_suffix}\n{run_name}", fontsize=13, pad=12)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    ax.set_xlabel(
        f"Predicted label\n\nSensitivity (recall): {sensitivity:.3f}  "
        f"|  Specificity: {specificity:.3f}",
        fontsize=11,
    )

    plt.tight_layout()
    if save:
        slug = run_name.replace(" ", "_").replace("/", "-")
        path = PLOTS_DIR / f"confusion_matrix_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   saved → {path}")
    return fig


# ── 3b. ROC curve ────────────────────────────────────────────────

def plot_roc_curve(
    y_true, y_prob,
    run_name="",
    save=True,
):
    """
    ROC curve with AUC annotation and random-baseline reference.

    Args:
        y_true  : ground-truth labels
        y_prob  : predicted probability for the POSITIVE class (pneumonia)
        run_name: label for title/filename
        save    : save PNG
    Returns:
        matplotlib Figure, float auc_score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)

    # Youden's J — best threshold
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#378ADD", lw=2.5,
            label=f"ROC  (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random baseline")
    ax.scatter(
        fpr[best_idx], tpr[best_idx],
        color="#D85A30", s=80, zorder=5,
        label=f"Best threshold = {best_thr:.2f}\n"
              f"(TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})",
    )
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"ROC curve\n{run_name}", pad=12)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    if save:
        slug = run_name.replace(" ", "_").replace("/", "-")
        path = PLOTS_DIR / f"roc_curve_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   saved → {path}")
    return fig, auc_score


# ── 3c. Precision–Recall curve ───────────────────────────────────

def plot_precision_recall_curve(
    y_true, y_prob,
    run_name="",
    save=True,
):
    """
    Precision–Recall curve with AP annotation.
    Especially informative for imbalanced datasets.

    Args:
        y_true  : ground-truth labels
        y_prob  : predicted probability for POSITIVE class
        run_name: label for title/filename
        save    : save PNG
    Returns:
        matplotlib Figure, float ap_score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = np.sum(y_true) / len(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#1D9E75", lw=2.5,
            label=f"Model  (AP = {ap:.4f})")
    ax.axhline(baseline, color="gray", lw=1, linestyle="--",
               label=f"Random baseline ({baseline:.2f})")

    # Mark F1 = 0.8 contour
    f1_contour_r = np.linspace(0.01, 1.0, 200)
    for f1_target in [0.7, 0.8, 0.9]:
        p_contour = f1_target * f1_contour_r / (2 * f1_contour_r - f1_target)
        mask = (p_contour >= 0) & (p_contour <= 1)
        ax.plot(f1_contour_r[mask], p_contour[mask],
                color="#AAAAAA", lw=0.7, linestyle=":")
        if mask.any():
            ax.annotate(
                f"F1={f1_target}", xy=(f1_contour_r[mask][-1], p_contour[mask][-1]),
                fontsize=8, color="#888888",
            )

    ax.set_xlim([0, 1.01])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"Precision–Recall curve\n{run_name}", pad=12)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if save:
        slug = run_name.replace(" ", "_").replace("/", "-")
        path = PLOTS_DIR / f"pr_curve_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   saved → {path}")
    return fig, ap


# ── 3d. Learning curves ──────────────────────────────────────────

def plot_learning_curves(
    train_losses, val_losses,
    train_aucs=None, val_aucs=None,
    run_name="",
    save=True,
):
    """
    Training & validation loss (and optionally AUC) per epoch.

    Args:
        train_losses : list of training loss values, one per epoch
        val_losses   : list of validation loss values, one per epoch
        train_aucs   : (optional) list of training AUC per epoch
        val_aucs     : (optional) list of validation AUC per epoch
        run_name     : label for title/filename
        save         : save PNG
    Returns:
        matplotlib Figure
    """
    has_auc = (train_aucs is not None and val_aucs is not None
               and len(train_aucs) > 0)
    n_plots = 2 if has_auc else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)
    best_val_epoch = int(np.argmin(val_losses)) + 1

    # Loss subplot
    ax = axes[0]
    ax.plot(epochs, train_losses, color="#378ADD", lw=2, label="Train loss")
    ax.plot(epochs, val_losses,   color="#D85A30", lw=2, label="Val loss",
            linestyle="--")
    ax.axvline(best_val_epoch, color="#1D9E75", lw=1, linestyle=":",
               label=f"Best val epoch ({best_val_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Learning curves — loss\n{run_name}", pad=10)
    ax.legend(fontsize=9)

    # AUC subplot
    if has_auc:
        ax2 = axes[1]
        best_auc_epoch = int(np.argmax(val_aucs)) + 1
        ax2.plot(epochs, train_aucs, color="#378ADD", lw=2, label="Train AUC")
        ax2.plot(epochs, val_aucs,   color="#D85A30", lw=2, label="Val AUC",
                 linestyle="--")
        ax2.axvline(best_auc_epoch, color="#1D9E75", lw=1, linestyle=":",
                    label=f"Best val epoch ({best_auc_epoch})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC")
        ax2.set_title(f"Learning curves — AUC\n{run_name}", pad=10)
        ax2.legend(fontsize=9)

    plt.tight_layout()
    if save:
        slug = run_name.replace(" ", "_").replace("/", "-")
        path = PLOTS_DIR / f"learning_curves_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   saved → {path}")
    return fig


# ── 3e. Cross-run comparison bar chart ───────────────────────────

def plot_runs_comparison(runs: list[dict], save=True):
    """
    Side-by-side bar chart comparing Accuracy, AUC and F1 across all runs.
    Bars are colour-coded by model type.

    Args:
        runs : list of run dicts from load_runs()
        save : save PNG
    Returns:
        matplotlib Figure
    """
    if len(runs) < 2:
        print("⚠️  Need at least 2 runs for comparison plot.")
        return None

    names  = [f"#{r['id']} {r['name'][:18]}" for r in runs]
    acc    = [r["accuracy"]     for r in runs]
    aucs   = [r.get("auc")      for r in runs]
    f1s    = [r.get("f1")       for r in runs]
    colors = [PALETTE.get(r["model"], "#7F77DD") for r in runs]

    has_auc = any(v is not None for v in aucs)
    has_f1  = any(v is not None for v in f1s)
    n_cols  = 1 + int(has_auc) + int(has_f1)

    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    def _bar(ax, values, title, ylabel, ylim=None, fmt="{:.2f}"):
        safe = [v if v is not None else 0 for v in values]
        bars = ax.bar(range(len(names)), safe, color=colors,
                      edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, pad=10)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        best_idx = int(np.argmax(safe))
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val is None:
                continue
            label = fmt.format(val)
            if i == best_idx:
                label += " ★"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] - ylim[0]) * 0.01 if ylim else bar.get_height() * 1.01,
                label, ha="center", va="bottom", fontsize=8,
                fontweight="bold" if i == best_idx else "normal",
            )

    col = 0
    min_acc = max(0, min(acc) - 5)
    _bar(axes[col], acc, "Accuracy (%)", "Accuracy (%)",
         ylim=(min_acc, 101), fmt="{:.2f}")
    col += 1

    if has_auc:
        min_auc = max(0, min(v for v in aucs if v) - 0.05)
        _bar(axes[col], aucs, "AUC (ROC)", "AUC",
             ylim=(min_auc, 1.02), fmt="{:.4f}")
        col += 1

    if has_f1:
        min_f1 = max(0, min(v for v in f1s if v) - 0.05)
        _bar(axes[col], f1s, "F1 score", "F1",
             ylim=(min_f1, 1.02), fmt="{:.4f}")

    # Legend for model colours
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=m)
        for m, c in PALETTE.items()
        if any(r["model"] == m for r in runs)
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.05), frameon=False)

    fig.suptitle("Cross-run comparison", fontsize=14, y=1.01)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "comparison_all_runs.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   saved → {path}")
    return fig


# ── 3f. Prediction probability distribution ──────────────────────

def plot_probability_distribution(
    y_true, y_prob,
    threshold=0.5,
    run_name="",
    save=True,
):
    """
    Histogram of predicted probabilities split by true class.
    Shows how well the model separates the two classes.

    Args:
        y_true    : ground-truth labels
        y_prob    : predicted probability for POSITIVE class
        threshold : decision threshold to draw a vertical line
        run_name  : label for title/filename
        save      : save PNG
    Returns:
        matplotlib Figure
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 40)

    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.6,
            color=NORMAL_COLOR,    label="True NORMAL",    density=True)
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.6,
            color=PNEUMONIA_COLOR, label="True PNEUMONIA", density=True)
    ax.axvline(threshold, color="#333333", lw=1.5, linestyle="--",
               label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel("Predicted probability (pneumonia)")
    ax.set_ylabel("Density")
    ax.set_title(f"Predicted probability distribution\n{run_name}", pad=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        slug = run_name.replace(" ", "_").replace("/", "-")
        path = PLOTS_DIR / f"prob_dist_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   saved → {path}")
    return fig


# ── 3g. Full diagnostic summary (all-in-one) ─────────────────────

def plot_diagnostic_summary(
    y_true, y_pred, y_prob,
    train_losses=None, val_losses=None,
    train_aucs=None,  val_aucs=None,
    threshold=0.5,
    run_name="",
    class_names=("NORMAL", "PNEUMONIA"),
    save=True,
):
    """
    Single figure with 6 subplots:
      top row   : confusion matrix | ROC curve | PR curve
      bottom row: probability dist | learning loss | learning AUC (or blank)

    Args:
        y_true       : ground-truth labels
        y_pred       : predicted labels
        y_prob       : predicted probability for positive class
        train_losses : (optional) list of training losses per epoch
        val_losses   : (optional) list of val losses per epoch
        train_aucs   : (optional) list of training AUC per epoch
        val_aucs     : (optional) list of val AUC per epoch
        threshold    : decision threshold
        run_name     : title prefix
        class_names  : class names
        save         : save PNG
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Diagnostic summary — {run_name}", fontsize=15, y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax_cm, ax_roc, ax_pr, ax_pd, ax_lc_loss, ax_lc_auc = axes

    # 1 — Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="white", ax=ax_cm,
        annot_kws={"size": 13, "weight": "bold"},
    )
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    ax_cm.set_title("Confusion matrix")
    ax_cm.set_xlabel(f"Predicted\nSensitivity={sens:.3f}  Specificity={spec:.3f}",
                     fontsize=9)
    ax_cm.set_ylabel("True label")

    # 2 — ROC curve
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    best_idx = int(np.argmax(tpr - fpr))
    ax_roc.plot(fpr, tpr, color="#378ADD", lw=2,
                label=f"AUC = {auc_score:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax_roc.scatter(fpr[best_idx], tpr[best_idx],
                   color="#D85A30", s=60, zorder=5,
                   label=f"thr={thr_roc[best_idx]:.2f}")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(fontsize=8, loc="lower right")

    # 3 — PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline_pr = np.sum(y_true) / len(y_true)
    ax_pr.plot(rec, prec, color="#1D9E75", lw=2, label=f"AP = {ap:.4f}")
    ax_pr.axhline(baseline_pr, color="gray", lw=1, linestyle="--",
                  label=f"Baseline={baseline_pr:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall curve")
    ax_pr.legend(fontsize=8)

    # 4 — Probability distribution
    bins = np.linspace(0, 1, 35)
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    ax_pd.hist(y_prob_arr[y_true_arr == 0], bins=bins, alpha=0.6,
               color=NORMAL_COLOR,    label="NORMAL",    density=True)
    ax_pd.hist(y_prob_arr[y_true_arr == 1], bins=bins, alpha=0.6,
               color=PNEUMONIA_COLOR, label="PNEUMONIA", density=True)
    ax_pd.axvline(threshold, color="#333", lw=1.5, linestyle="--",
                  label=f"thr={threshold:.2f}")
    ax_pd.set_xlabel("P(pneumonia)")
    ax_pd.set_ylabel("Density")
    ax_pd.set_title("Probability distribution")
    ax_pd.legend(fontsize=8)

    # 5 — Learning curve (loss)
    if train_losses and val_losses:
        ep = range(1, len(train_losses) + 1)
        ax_lc_loss.plot(ep, train_losses, color="#378ADD", lw=2, label="Train")
        ax_lc_loss.plot(ep, val_losses,   color="#D85A30", lw=2,
                        linestyle="--", label="Val")
        best_ep = int(np.argmin(val_losses)) + 1
        ax_lc_loss.axvline(best_ep, color="#1D9E75", lw=1, linestyle=":",
                           label=f"Best ep {best_ep}")
        ax_lc_loss.set_xlabel("Epoch")
        ax_lc_loss.set_ylabel("Loss")
        ax_lc_loss.set_title("Learning curves (loss)")
        ax_lc_loss.legend(fontsize=8)
    else:
        ax_lc_loss.text(0.5, 0.5, "No epoch data\nlogged for this run",
                        ha="center", va="center", transform=ax_lc_loss.transAxes,
                        fontsize=10, color="#888888")
        ax_lc_loss.set_title("Learning curves (loss)")

    # 6 — Learning curve (AUC)
    if train_aucs and val_aucs:
        ep = range(1, len(train_aucs) + 1)
        ax_lc_auc.plot(ep, train_aucs, color="#378ADD", lw=2, label="Train")
        ax_lc_auc.plot(ep, val_aucs,   color="#D85A30", lw=2,
                       linestyle="--", label="Val")
        best_ep_a = int(np.argmax(val_aucs)) + 1
        ax_lc_auc.axvline(best_ep_a, color="#1D9E75", lw=1, linestyle=":",
                          label=f"Best ep {best_ep_a}")
        ax_lc_auc.set_xlabel("Epoch")
        ax_lc_auc.set_ylabel("AUC")
        ax_lc_auc.set_title("Learning curves (AUC)")
        ax_lc_auc.legend(fontsize=8)
    else:
        ax_lc_auc.text(0.5, 0.5, "No AUC epoch data\nlogged for this run",
                       ha="center", va="center", transform=ax_lc_auc.transAxes,
                       fontsize=10, color="#888888")
        ax_lc_auc.set_title("Learning curves (AUC)")

    if save:
        slug = run_name.replace(" ", "_").replace("/", "-")
        path = PLOTS_DIR / f"diagnostic_summary_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   saved → {path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 — Interactive CLI
# ═══════════════════════════════════════════════════════════════════

def _ask(prompt, cast=str, allow_empty=False, default=None):
    """Prompt user for input with optional type casting."""
    while True:
        raw = input(prompt).strip()
        if raw == "" and allow_empty:
            return default
        if raw == "" and default is not None:
            return default
        if raw == "":
            continue
        try:
            return cast(raw)
        except ValueError:
            print(f"   ⚠️  Expected {cast.__name__}, got '{raw}'. Try again.")


def _ask_float(prompt):
    return _ask(prompt, float, allow_empty=True, default=None)


def menu_add_run(runs):
    print("\n── New experiment run ───────────────────────────────")
    name  = _ask("Run name: ")
    model = _ask(
        "Model type [1=Custom CNN | 2=ResNet152 | 3=EfficientNet | 4=DenseNet121 | 5=Other]: ",
        int, allow_empty=True, default=1,
    )
    model_map = {1:"Custom CNN", 2:"ResNet152", 3:"EfficientNet",
                 4:"DenseNet121", 5:"Other"}
    model_name = model_map.get(model, "Other")
    accuracy   = _ask("Accuracy % (e.g. 87.5): ", float)
    auc_score  = _ask_float("AUC (e.g. 0.921, leave blank to skip): ")
    f1_score   = _ask_float("F1 score (e.g. 0.880, leave blank to skip): ")
    val_loss   = _ask_float("Val loss (e.g. 0.34, leave blank to skip): ")
    epochs     = _ask("Epochs run (leave blank to skip): ",
                      int, allow_empty=True, default=None)
    threshold  = _ask_float("Threshold used (e.g. 0.55, leave blank to skip): ")
    hp         = _ask("Config notes (hidden_units, lr, etc. — leave blank to skip): ",
                      allow_empty=True, default="")
    notes      = _ask("Free-text notes (leave blank to skip): ",
                      allow_empty=True, default="")

    run = dict(
        name=name, model=model_name,
        accuracy=accuracy, auc=auc_score, f1=f1_score,
        val_loss=val_loss, epochs=epochs, threshold=threshold,
        hp=hp, notes=notes,
    )
    runs = add_run(runs, run)
    return runs


def menu_generate_plots(runs):
    print("\n── Generate plots ───────────────────────────────────")
    print("These plots require actual model output (y_true, y_pred, y_prob).")
    print("Call the plot functions directly from your training script, e.g.:\n")
    print("  from results_tracker import plot_diagnostic_summary")
    print("  plot_diagnostic_summary(")
    print("      y_true=all_labels, y_pred=all_preds, y_prob=all_preds_probs,")
    print("      train_losses=train_loss_history, val_losses=val_loss_history,")
    print("      run_name='ResNet152 v2', threshold=0.55)")
    print()
    if len(runs) >= 2:
        print("Generating cross-run comparison chart from saved runs…")
        plot_runs_comparison(runs)
        plt.show()
    else:
        print("(Log at least 2 runs to generate the comparison chart.)")


def interactive_menu():
    runs = load_runs()
    while True:
        print("\n╔══════════════════════════════════════╗")
        print("║  Pneumonia Detection — Results Log   ║")
        print("╚══════════════════════════════════════╝")
        print(f"  Runs logged: {len(runs)}")
        print()
        print("  1  List all runs")
        print("  2  Add new run")
        print("  3  Delete a run")
        print("  4  Generate cross-run comparison chart")
        print("  5  Show plot function usage")
        print("  0  Exit")
        choice = _ask("\nChoice: ", int, allow_empty=True, default=0)

        if choice == 1:
            print_runs_table(runs)
        elif choice == 2:
            runs = menu_add_run(runs)
        elif choice == 3:
            print_runs_table(runs)
            rid = _ask("Delete run ID: ", int)
            runs = delete_run(runs, rid)
        elif choice == 4:
            runs = load_runs()
            if runs:
                plot_runs_comparison(runs)
                plt.show()
            else:
                print("No runs yet.")
        elif choice == 5:
            menu_generate_plots(runs)
        elif choice == 0:
            print("Bye!")
            break


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5 — Integration helpers (call from main.py / resnet152.py)
# ═══════════════════════════════════════════════════════════════════

def log_and_plot(
    run_name: str,
    model_type: str,
    accuracy: float,
    y_true,
    y_pred,
    y_prob,
    auc_score: float = None,
    f1_score: float  = None,
    val_loss: float  = None,
    epochs: int      = None,
    threshold: float = 0.5,
    train_losses     = None,
    val_losses       = None,
    train_aucs       = None,
    val_aucs         = None,
    hp_notes: str    = "",
    notes: str       = "",
):
    """
    One-call convenience: log a run AND generate the full diagnostic summary.

    Example — paste at the bottom of your training loop in main.py or resnet152.py:

        from results_tracker import log_and_plot

        log_and_plot(
            run_name    = "ResNet152 with class weights",
            model_type  = "ResNet152",
            accuracy    = accuracy * 100,
            y_true      = all_labels,
            y_pred      = all_preds,
            y_prob      = all_preds_probs,
            auc_score   = test_auc,
            f1_score    = f1_score(all_labels, all_preds, average='weighted'),
            val_loss    = best_val_loss,
            epochs      = epochs_run,
            threshold   = 0.55,
            train_losses= train_loss_history,
            val_losses  = val_loss_history,
            train_aucs  = train_auc_history,
            val_aucs    = val_auc_history,
            hp_notes    = "hu=32, lr=1e-3, class_weights=True",
            notes       = "First run after adding class weights",
        )
    """
    runs = load_runs()
    run  = dict(
        name=run_name, model=model_type,
        accuracy=accuracy, auc=auc_score, f1=f1_score,
        val_loss=val_loss, epochs=epochs, threshold=threshold,
        hp=hp_notes, notes=notes,
    )
    runs = add_run(runs, run)

    print(f"\n📊  Generating diagnostic plots for '{run_name}' …")
    plot_diagnostic_summary(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        train_losses=train_losses, val_losses=val_losses,
        train_aucs=train_aucs,     val_aucs=val_aucs,
        threshold=threshold,
        run_name=run_name,
        save=True,
    )

    if len(runs) >= 2:
        print("📊  Updating cross-run comparison chart …")
        plot_runs_comparison(runs, save=True)

    plt.show()


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pneumonia results tracker")
    parser.add_argument("--list",      action="store_true", help="Print all runs and exit")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate comparison chart and exit")
    args = parser.parse_args()

    if args.list:
        print_runs_table(load_runs())
    elif args.plot_only:
        runs = load_runs()
        plot_runs_comparison(runs)
        plt.show()
    else:
        interactive_menu()