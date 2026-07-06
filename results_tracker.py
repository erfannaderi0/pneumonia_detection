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
Saves logs to:   results_log/terminal_logs/

Fields tracked in runs.json (v4):
    Identity
        id, date, run_name, model_type, notes

    Core metrics
        accuracy, auc, f1 (weighted), val_loss, test_loss
        sensitivity (recall for PNEUMONIA), specificity

    Per-class metrics (NORMAL / PNEUMONIA)
        precision, recall, f1 per class
        prec_at_recall_90, prec_at_recall_95  (clinical operating points)
        ece  (Expected Calibration Error)
        mcc  (Matthews Correlation Coefficient — robust on imbalanced data)

    Epoch & timing
        epochs_trained  (actual, early-stopping aware)
        max_epochs      (configured ceiling)
        early_stop_patience
        best_epoch      (epoch with best val metric)
        train_time_secs
        epoch_history   (NEW: stores all epoch-by-epoch metrics for learning curves)

    Hyperparameters (structured)
        lr, lr_scheduler, batch_size, optimizer, weight_decay
        img_size, pretrained, freeze_backbone_epochs
        augmentations, class_weighting, dropout_rate

    Dataset
        train_total, train_normal, train_pneumonia  (+ imbalance_ratio)
        val_total,   val_normal,   val_pneumonia
        test_total,  test_normal,  test_pneumonia
        preprocessing (normalisation strategy string)

    Environment (auto-captured)
        python_version, framework, framework_version
        gpu_name, hostname

    Terminal log
        log_file  (path to the saved terminal log for this run)
"""

import os
import sys
import json
import time
import socket
import platform
import argparse
import textwrap
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
    precision_score, recall_score, f1_score as sklearn_f1,
    matthews_corrcoef,
)

# ─────────────────────────────────────────────
#  Directories
# ─────────────────────────────────────────────
LOG_DIR   = Path("results_log")
PLOTS_DIR = LOG_DIR / "plots"
RUNS_FILE = LOG_DIR / "runs.json"
TERM_DIR  = LOG_DIR / "terminal_logs"   # one .log file per run

LOG_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
TERM_DIR.mkdir(exist_ok=True)


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
#  SECTION 0 — Terminal logger  (tee stdout+stderr to a .log file)
# ═══════════════════════════════════════════════════════════════════

class TerminalLogger:
    """
    Redirect both stdout and stderr so every line printed to the console
    is *also* written to a timestamped .log file.

    Usage — call start_terminal_log() at the very top of your training
    script, before any other output:

        from results_tracker import start_terminal_log, stop_terminal_log

        log_path = start_terminal_log("ResNet152_run3")
        # ... all your training code here ...
        stop_terminal_log()

    The log_path string is then passed into log_and_plot() so the run
    record in runs.json links directly to the file.

    What gets captured:
        • Every print() call (train loss, val AUC, early-stop messages…)
        • Keras / PyTorch epoch progress bars (if they use stdout)
        • Any warnings or tracebacks sent to stderr
        • A header with timestamp, hostname, Python version, and GPU info

    What does NOT get captured:
        • C-level output from CUDA / cuDNN (it bypasses Python streams)
        • Output from subprocesses unless you redirect them explicitly
    """

    def __init__(self, log_path: Path):
        self.log_path  = log_path
        self.log_file  = open(log_path, "w", encoding="utf-8", buffering=1)
        self._stdout   = sys.stdout
        self._stderr   = sys.stderr

    # ── stream interface ──────────────────────────────────────────
    def write(self, msg):
        self._stdout.write(msg)      # still print to terminal
        self.log_file.write(msg)     # and write to file

    def flush(self):
        self._stdout.flush()
        self.log_file.flush()

    def isatty(self):
        return False                 # prevents some libraries skipping progress bars

    # ── context manager ───────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.log_file.flush()
        self.log_file.close()


# Module-level handle so stop_terminal_log() can reach it
_active_logger: TerminalLogger | None = None


def start_terminal_log(run_name: str) -> str:
    """
    Start capturing all terminal output for *run_name*.

    Writes a header section with environment info, then tees every
    subsequent print/stderr line to the log file.

    Args:
        run_name : human-readable run label (used in the filename)

    Returns:
        str — absolute path to the log file (store this and pass it
              to log_and_plot() as `log_file=...`)

    Example::

        from results_tracker import start_terminal_log, stop_terminal_log

        log_path = start_terminal_log("EfficientNet_aug_v2")
        try:
            # ... training loop ...
        finally:
            stop_terminal_log()   # always stop, even on crash
    """
    global _active_logger

    slug     = run_name.replace(" ", "_").replace("/", "-")
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = TERM_DIR / f"{ts}_{slug}.log"

    _active_logger = TerminalLogger(log_path)
    sys.stdout = _active_logger
    sys.stderr = _active_logger

    # Write a header so the log is self-contained
    env = _capture_environment()
    header_lines = [
        "=" * 70,
        f"  RUN : {run_name}",
        f"  DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  HOST: {env.get('hostname', '?')}",
        f"  GPU : {env.get('gpu_name', 'not detected')}",
        f"  PY  : {env.get('python_version', '?')}",
        f"  FW  : {env.get('framework', '?')}  {env.get('framework_version', '')}",
        "=" * 70,
        "",
    ]
    print("\n".join(header_lines))
    return str(log_path.resolve())


def stop_terminal_log():
    """
    Stop capturing terminal output and close the log file.

    Always call this in a ``finally`` block so the file is closed
    cleanly even if training crashes mid-way:

        log_path = start_terminal_log("run_name")
        try:
            ...
        finally:
            stop_terminal_log()
    """
    global _active_logger
    if _active_logger is not None:
        footer = [
            "",
            "=" * 70,
            f"  LOG CLOSED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
        ]
        print("\n".join(footer))
        _active_logger.close()
        _active_logger = None


# ─────────────────────────────────────────────────────────────────
#  Environment helper  (auto-filled into every run record)
# ─────────────────────────────────────────────────────────────────

def _capture_environment() -> dict:
    """
    Silently detect Python version, ML framework, and GPU name.
    Never raises — missing info becomes None so the run still saves.
    """
    env = {
        "python_version": platform.python_version(),
        "hostname":       socket.gethostname(),
        "gpu_name":       None,
        "framework":      None,
        "framework_version": None,
    }

    # Detect framework + GPU
    try:
        import tensorflow as tf
        env["framework"]         = "TensorFlow"
        env["framework_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # TF doesn't expose GPU name easily; use subprocess as fallback
            env["gpu_name"] = gpus[0].name
    except ImportError:
        pass

    if env["framework"] is None:
        try:
            import torch
            env["framework"]         = "PyTorch"
            env["framework_version"] = torch.__version__
            if torch.cuda.is_available():
                env["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

    # Fallback GPU detection via nvidia-smi
    if env["gpu_name"] is None:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=3,
            )
            env["gpu_name"] = out.decode().strip().split("\n")[0]
        except Exception:
            pass

    return env
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

    # Auto-capture environment info unless caller already supplied it
    if "python_version" not in run:
        env = _capture_environment()
        run.setdefault("python_version",    env.get("python_version"))
        run.setdefault("framework",         env.get("framework"))
        run.setdefault("framework_version", env.get("framework_version"))
        run.setdefault("gpu_name",          env.get("gpu_name"))
        run.setdefault("hostname",          env.get("hostname"))

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
    header += f"{'Acc%':>7}  {'AUC':>7}  {'F1':>7}  {'MCC':>7}  {'Loss':>7}  {'Epochs':>12}  {'Time':>8}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    best_acc = max(r["accuracy"] for r in runs)
    for r in runs:
        marker = " ★" if r["accuracy"] == best_acc else ""

        # Show "trained/max" if early stopping info available
        ep_max    = r.get("max_epochs")
        ep_actual = r.get("epochs_trained", r.get("epochs"))
        best_ep   = r.get("best_epoch")
        if ep_max and ep_actual and ep_actual != ep_max:
            epochs_str = f"{ep_actual}/{ep_max} (ES)"
        elif ep_actual:
            epochs_str = str(ep_actual)
        else:
            epochs_str = "—"
        if best_ep:
            epochs_str += f" [best={best_ep}]"

        train_secs = r.get("train_time_secs")
        if train_secs:
            mins, secs = divmod(int(train_secs), 60)
            time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        else:
            time_str = "—"

        row = (
            f"{r['id']:>3}  {r['date'][:16]:<16}  {r['name'][:28]:<28}  "
            f"{r['model'][:14]:<14}  "
            f"{_fmt(r['accuracy'], 2):>7}  "
            f"{_fmt(r.get('auc'), 4):>7}  "
            f"{_fmt(r.get('f1'), 4):>7}  "
            f"{_fmt(r.get('mcc'), 4):>7}  "
            f"{_fmt(r.get('val_loss'), 4):>7}  "
            f"{epochs_str:>12}  "
            f"{time_str:>8}"
            f"{marker}"
        )
        print(row)

        # Check if epoch history exists
        has_epoch_history = r.get("epoch_history") and len(r.get("epoch_history", {}).get("train_loss", [])) > 0
        if has_epoch_history:
            print(f"     ↳ epoch history: {len(r['epoch_history']['train_loss'])} epochs recorded")

        # Per-class metrics line
        pc = r.get("per_class_metrics", {})
        if pc:
            pn  = pc.get("NORMAL",    {})
            pp  = pc.get("PNEUMONIA", {})
            print(
                f"     ↳ NORMAL    prec={_fmt(pn.get('precision'),3)}  "
                f"rec={_fmt(pn.get('recall'),3)}  f1={_fmt(pn.get('f1'),3)}"
            )
            print(
                f"     ↳ PNEUMONIA prec={_fmt(pp.get('precision'),3)}  "
                f"rec={_fmt(pp.get('recall'),3)}  f1={_fmt(pp.get('f1'),3)}  "
                f"P@R90={_fmt(r.get('prec_at_recall_90'),3)}  "
                f"P@R95={_fmt(r.get('prec_at_recall_95'),3)}  "
                f"ECE={_fmt(r.get('ece'),4)}"
            )

        # Dataset sizes line
        ds = r.get("dataset", {})
        if ds:
            imb = ds.get("imbalance_ratio")
            imb_str = f"  imbalance={imb:.2f}" if imb else ""
            print(
                f"     ↳ dataset   train={ds.get('train_total','?')} "
                f"(N:{ds.get('train_normal','?')} / P:{ds.get('train_pneumonia','?')})  "
                f"val={ds.get('val_total','?')}  test={ds.get('test_total','?')}"
                f"{imb_str}"
            )
            if ds.get("preprocessing"):
                print(f"     ↳ preproc   {ds['preprocessing']}")

        # Hyperparams line
        hp = r.get("hp") or r.get("hyperparams", {})
        if isinstance(hp, dict) and hp:
            hp_str = "  ".join(f"{k}={v}" for k, v in hp.items() if v is not None)
            print(f"     ↳ hp        {textwrap.shorten(hp_str, 100)}")
        elif isinstance(hp, str) and hp:
            print(f"     ↳ hp        {textwrap.shorten(hp, 100)}")

        # Environment line
        fw  = r.get("framework")
        fwv = r.get("framework_version")
        gpu = r.get("gpu_name")
        host = r.get("hostname")
        env_parts = []
        if fw:  env_parts.append(f"{fw} {fwv or ''}")
        if gpu: env_parts.append(f"GPU={gpu}")
        if host: env_parts.append(f"host={host}")
        if env_parts:
            print(f"     ↳ env       {' | '.join(env_parts)}")

        # Terminal log link
        lf = r.get("log_file")
        if lf:
            print(f"     ↳ log       {lf}")

        if r.get("notes"):
            print(f"     ↳ notes     {textwrap.shorten(r['notes'], 100)}")

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


# ── 3g. Calibration curve (reliability diagram) ──────────────────

def plot_calibration_curve(
    y_true, y_prob,
    run_name="",
    n_bins=10,
    save=True,
):
    """
    Reliability diagram showing whether predicted probabilities match
    observed frequencies.  A perfectly calibrated model follows the
    diagonal.  Also plots a histogram of confidence scores.

    Args:
        y_true   : ground-truth labels (0 or 1)
        y_prob   : predicted probability for POSITIVE class (pneumonia)
        run_name : label for title/filename
        n_bins   : number of calibration bins
        save     : save PNG
    Returns:
        matplotlib Figure
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Expected Calibration Error (ECE)
    bins       = np.linspace(0, 1, n_bins + 1)
    bin_ids    = np.digitize(y_prob, bins[1:-1])
    ece        = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            acc_b  = y_true[mask].mean()
            conf_b = y_prob[mask].mean()
            ece   += mask.mean() * abs(acc_b - conf_b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: reliability diagram ──────────────────────────────────
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax1.plot(prob_pred, prob_true, color="#378ADD", lw=2.5, marker="o",
             markersize=6, label=f"Model  (ECE = {ece:.4f})")
    ax1.fill_between(prob_pred, prob_pred, prob_true,
                     alpha=0.12, color="#D85A30", label="Calibration gap")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(f"Calibration curve (reliability diagram)\n{run_name}", pad=10)
    ax1.legend(fontsize=9, loc="upper left")

    # ── Right: confidence histogram ────────────────────────────────
    ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.6,
             color=NORMAL_COLOR,    label="True NORMAL",    density=True)
    ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.6,
             color=PNEUMONIA_COLOR, label="True PNEUMONIA", density=True)
    ax2.set_xlabel("Predicted probability (pneumonia)")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence score distribution")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save:
        slug = run_name.replace(" ", "_").replace("/", "-")
        path = PLOTS_DIR / f"calibration_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   saved → {path}")
    return fig


# ── 3h. Full diagnostic summary (all-in-one) ─────────────────────

def plot_diagnostic_summary(
    y_true, y_pred, y_prob,
    train_losses=None, val_losses=None,
    train_aucs=None,  val_aucs=None,
    epoch_history=None,  # NEW: can pass full epoch history dict
    threshold=0.5,
    run_name="",
    class_names=("NORMAL", "PNEUMONIA"),
    save=True,
):
    """
    Single figure with 7 subplots (2 rows × 4 cols, last cell = calibration):
      top row   : confusion matrix | ROC curve | PR curve | Calibration
      bottom row: probability dist | learning loss | learning AUC | per-class bar

    Args:
        y_true       : ground-truth labels
        y_pred       : predicted labels
        y_prob       : predicted probability for positive class
        train_losses : (optional) list of training losses per epoch
        val_losses   : (optional) list of val losses per epoch
        train_aucs   : (optional) list of training AUC per epoch
        val_aucs     : (optional) list of val AUC per epoch
        epoch_history: (NEW) dict with 'train_loss', 'val_loss', 'train_auc', 'val_auc'
        threshold    : decision threshold
        run_name     : title prefix
        class_names  : class names
        save         : save PNG
    Returns:
        matplotlib Figure
    """
    # Extract data from epoch_history if provided and individual lists are None
    if epoch_history and train_losses is None:
        train_losses = epoch_history.get("train_loss", [])
        val_losses = epoch_history.get("val_loss", [])
        train_aucs = epoch_history.get("train_auc", [])
        val_aucs = epoch_history.get("val_auc", [])
    
    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f"Diagnostic summary — {run_name}", fontsize=15, y=1.01)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.38)

    ax_cm       = fig.add_subplot(gs[0, 0])
    ax_roc      = fig.add_subplot(gs[0, 1])
    ax_pr       = fig.add_subplot(gs[0, 2])
    ax_cal      = fig.add_subplot(gs[0, 3])
    ax_pd       = fig.add_subplot(gs[1, 0])
    ax_lc_loss  = fig.add_subplot(gs[1, 1])
    ax_lc_auc   = fig.add_subplot(gs[1, 2])
    ax_pc       = fig.add_subplot(gs[1, 3])

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)

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
    # Mark P@R90 and P@R95 on the curve
    for target_recall, color, label in [(0.90, "#EF9F27", "R90"), (0.95, "#D85A30", "R95")]:
        idx = np.searchsorted(-rec[::-1], -target_recall)
        idx = len(rec) - 1 - idx
        if 0 <= idx < len(rec):
            ax_pr.scatter(rec[idx], prec[idx], color=color, s=50, zorder=5,
                          label=f"P@{label}={prec[idx]:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall curve")
    ax_pr.legend(fontsize=7)

    # 4 — Calibration curve
    prob_true_cal, prob_pred_cal = calibration_curve(y_true_arr, y_prob_arr, n_bins=10)
    bins_cal   = np.linspace(0, 1, 11)
    bin_ids    = np.digitize(y_prob_arr, bins_cal[1:-1])
    ece        = sum(
        (bin_ids == b).mean() * abs(y_true_arr[bin_ids == b].mean() - y_prob_arr[bin_ids == b].mean())
        for b in range(10) if (bin_ids == b).any()
    )
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    ax_cal.plot(prob_pred_cal, prob_true_cal, color="#378ADD", lw=2,
                marker="o", markersize=5, label=f"Model (ECE={ece:.4f})")
    ax_cal.fill_between(prob_pred_cal, prob_pred_cal, prob_true_cal,
                        alpha=0.12, color="#D85A30")
    ax_cal.set_xlim([0, 1]); ax_cal.set_ylim([0, 1])
    ax_cal.set_xlabel("Mean predicted prob.")
    ax_cal.set_ylabel("Fraction of positives")
    ax_cal.set_title("Calibration curve")
    ax_cal.legend(fontsize=8, loc="upper left")

    # 5 — Probability distribution
    bins = np.linspace(0, 1, 35)
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

    # 6 — Learning curve (loss) - IMPROVED
    if train_losses and val_losses and len(train_losses) > 0:
        ep = range(1, len(train_losses) + 1)
        ax_lc_loss.plot(ep, train_losses, color="#378ADD", lw=2, label="Train")
        ax_lc_loss.plot(ep, val_losses, color="#D85A30", lw=2,
                        linestyle="--", label="Val")
        best_ep = int(np.argmin(val_losses)) + 1
        ax_lc_loss.axvline(best_ep, color="#1D9E75", lw=1, linestyle=":",
                           label=f"Best ep {best_ep}")
        ax_lc_loss.set_xlabel("Epoch")
        ax_lc_loss.set_ylabel("Loss")
        ax_lc_loss.set_title(f"Learning curves (loss)\n{len(train_losses)} epochs")
        ax_lc_loss.legend(fontsize=8)
        ax_lc_loss.grid(True, alpha=0.3)
    else:
        ax_lc_loss.text(0.5, 0.5, "No epoch data\nlogged for this run",
                        ha="center", va="center", transform=ax_lc_loss.transAxes,
                        fontsize=10, color="#888888")
        ax_lc_loss.set_title("Learning curves (loss)")

    # 7 — Learning curve (AUC) - IMPROVED
    if train_aucs and val_aucs and len(train_aucs) > 0:
        ep = range(1, len(train_aucs) + 1)
        ax_lc_auc.plot(ep, train_aucs, color="#378ADD", lw=2, label="Train")
        ax_lc_auc.plot(ep, val_aucs, color="#D85A30", lw=2,
                       linestyle="--", label="Val")
        best_ep_a = int(np.argmax(val_aucs)) + 1
        ax_lc_auc.axvline(best_ep_a, color="#1D9E75", lw=1, linestyle=":",
                          label=f"Best ep {best_ep_a}")
        ax_lc_auc.set_xlabel("Epoch")
        ax_lc_auc.set_ylabel("AUC")
        ax_lc_auc.set_title(f"Learning curves (AUC)\n{len(train_aucs)} epochs")
        ax_lc_auc.legend(fontsize=8)
        ax_lc_auc.grid(True, alpha=0.3)
    else:
        ax_lc_auc.text(0.5, 0.5, "No AUC epoch data\nlogged for this run",
                       ha="center", va="center", transform=ax_lc_auc.transAxes,
                       fontsize=10, color="#888888")
        ax_lc_auc.set_title("Learning curves (AUC)")

    # 8 — Per-class metrics bar chart
    pc_prec  = [precision_score(y_true_arr, y_pred, pos_label=i, zero_division=0)
                for i in range(2)]
    pc_rec   = [recall_score(y_true_arr, y_pred, pos_label=i, zero_division=0)
                for i in range(2)]
    pc_f1    = [sklearn_f1(y_true_arr, y_pred, pos_label=i, zero_division=0)
                for i in range(2)]
    x        = np.arange(2)
    w        = 0.25
    ax_pc.bar(x - w, pc_prec, w, label="Precision", color="#378ADD", zorder=2)
    ax_pc.bar(x,     pc_rec,  w, label="Recall",    color="#1D9E75", zorder=2)
    ax_pc.bar(x + w, pc_f1,   w, label="F1",        color="#EF9F27", zorder=2)
    ax_pc.set_xticks(x)
    ax_pc.set_xticklabels(class_names, fontsize=10)
    ax_pc.set_ylim([0, 1.12])
    ax_pc.set_ylabel("Score")
    ax_pc.set_title("Per-class metrics")
    ax_pc.legend(fontsize=8)
    for xi, (p, r, f) in enumerate(zip(pc_prec, pc_rec, pc_f1)):
        for offset, val in [(-w, p), (0, r), (w, f)]:
            ax_pc.text(xi + offset, val + 0.02, f"{val:.2f}",
                       ha="center", va="bottom", fontsize=7)

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
    f1_score   = _ask_float("F1 score — weighted (e.g. 0.880, leave blank to skip): ")
    val_loss   = _ask_float("Val loss (e.g. 0.34, leave blank to skip): ")

    print("\n── Epoch info (early stopping) ──────────────────────")
    epochs_trained = _ask("Epochs actually trained (early-stop epoch, e.g. 14): ",
                          int, allow_empty=True, default=None)
    max_epochs     = _ask("Max epochs configured (e.g. 50, leave blank to skip): ",
                          int, allow_empty=True, default=None)

    print("\n── Training time ────────────────────────────────────")
    train_mins = _ask_float("Training time in minutes (leave blank to skip): ")
    train_time_secs = int(train_mins * 60) if train_mins else None

    print("\n── Hyperparameters ──────────────────────────────────")
    lr          = _ask_float("Learning rate (e.g. 0.001): ")
    batch_size  = _ask("Batch size (e.g. 32, leave blank to skip): ",
                       int, allow_empty=True, default=None)
    optimizer   = _ask("Optimizer (e.g. Adam, SGD — leave blank to skip): ",
                       allow_empty=True, default="")
    img_size    = _ask("Image size (e.g. 224, leave blank to skip): ",
                       int, allow_empty=True, default=None)
    augment     = _ask("Augmentations used (e.g. flip,rotate,zoom — leave blank to skip): ",
                       allow_empty=True, default="")
    class_w     = _ask("Class weighting used? [y/n, leave blank to skip]: ",
                       allow_empty=True, default="")

    hyperparams = dict(
        lr=lr, batch_size=batch_size, optimizer=optimizer or None,
        img_size=img_size, augmentations=augment or None,
        class_weighting=class_w.lower() in ("y", "yes") if class_w else None,
    )

    print("\n── Dataset split sizes ──────────────────────────────")
    train_total     = _ask("Train set total (leave blank to skip): ",
                           int, allow_empty=True, default=None)
    train_normal    = _ask("  Train NORMAL count (leave blank to skip): ",
                           int, allow_empty=True, default=None)
    train_pneumonia = _ask("  Train PNEUMONIA count (leave blank to skip): ",
                           int, allow_empty=True, default=None)
    val_total       = _ask("Val set total (leave blank to skip): ",
                           int, allow_empty=True, default=None)
    test_total      = _ask("Test set total (leave blank to skip): ",
                           int, allow_empty=True, default=None)

    dataset = dict(
        train_total=train_total, train_normal=train_normal,
        train_pneumonia=train_pneumonia,
        val_total=val_total, test_total=test_total,
    )

    threshold  = _ask_float("Threshold used (e.g. 0.55, leave blank to skip): ")
    notes      = _ask("Free-text notes (leave blank to skip): ",
                      allow_empty=True, default="")

    run = dict(
        name=name, model=model_name,
        accuracy=accuracy, auc=auc_score, f1=f1_score,
        val_loss=val_loss,
        epochs_trained=epochs_trained, max_epochs=max_epochs,
        train_time_secs=train_time_secs,
        threshold=threshold,
        hyperparams=hyperparams,
        dataset=dataset,
        notes=notes,
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
    # ── Core metrics (auto-computed from y_* if not supplied) ──────
    auc_score: float        = None,
    f1_score: float         = None,
    val_loss: float         = None,
    test_loss: float        = None,
    # ── Epoch info ─────────────────────────────────────────────────
    epochs_trained: int     = None,
    max_epochs: int         = None,
    early_stop_patience: int = None,
    best_epoch: int         = None,
    # ── Training time ──────────────────────────────────────────────
    train_time_secs: float  = None,
    # ── Threshold ──────────────────────────────────────────────────
    threshold: float        = 0.5,
    # ── Epoch history (for learning curves) ────────────────────────
    train_losses            = None,
    val_losses              = None,
    train_aucs              = None,
    val_aucs                = None,
    epoch_history           = None,  # NEW: dict containing all epoch data
    # ── Structured hyperparameters ─────────────────────────────────
    lr: float               = None,
    lr_scheduler: str       = None,
    batch_size: int         = None,
    optimizer: str          = None,
    weight_decay: float     = None,
    img_size: int           = None,
    pretrained: bool        = None,
    freeze_backbone_epochs: int = None,
    augmentations: str      = None,
    class_weighting: bool   = None,
    dropout_rate: float     = None,
    # ── Dataset ────────────────────────────────────────────────────
    train_total: int        = None,
    train_normal: int       = None,
    train_pneumonia: int    = None,
    val_total: int          = None,
    val_normal: int         = None,
    val_pneumonia: int      = None,
    test_total: int         = None,
    test_normal: int        = None,
    test_pneumonia: int     = None,
    preprocessing: str      = None,
    # ── Terminal log ───────────────────────────────────────────────
    log_file: str           = None,
    # ── Free-text notes ────────────────────────────────────────────
    notes: str              = "",
):
    """
    One-call convenience: log a run AND generate the full diagnostic summary.

    Metrics that can be derived from y_true / y_pred / y_prob (MCC, ECE,
    per-class metrics, P@R90/95) are computed automatically — you don't
    need to pass them in.

    NEW: epoch_history parameter allows you to pass all epoch data in one dict.
    Example:
        epoch_history = {
            "train_loss": [...],
            "val_loss": [...],
            "train_auc": [...],
            "val_auc": [...]
        }

    Minimal example — at the training loop::

        from results_tracker import log_and_plot

        log_and_plot(
            run_name        = "ResNet152 with class weights",
            model_type      = "ResNet152",
            accuracy        = test_acc * 100,
            y_true          = all_labels,
            y_pred          = all_preds,
            y_prob          = all_probs,
            val_loss        = best_val_loss,
            epochs_trained  = early_stop_epoch,
            max_epochs      = 50,
            best_epoch      = best_epoch,
            train_losses    = train_loss_history,
            val_losses      = val_loss_history,
            train_aucs      = train_auc_history,
            val_aucs        = val_auc_history,
            lr              = 1e-3,
            batch_size      = 32,
            optimizer       = "Adam",
            img_size        = 224,
            augmentations   = "flip,rotate,zoom",
            class_weighting = True,
            train_total     = 4000,
            train_normal    = 1341,
            train_pneumonia = 2659,
            val_total       = 624,
            test_total      = 624,
            log_file        = log_path,   # from start_terminal_log()
            notes           = "First run with class weighting",
        )

    Full early-stopping + terminal-log pattern::

        from results_tracker import start_terminal_log, stop_terminal_log, log_and_plot

        log_path = start_terminal_log("ResNet152_run3")
        try:
            # ... your training loop ...
        finally:
            stop_terminal_log()

        log_and_plot(..., log_file=log_path)
    """
    import numpy as np

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_prob_arr = np.array(y_prob)

    # ── Extract data from epoch_history if provided ────────────────
    if epoch_history is not None:
        if train_losses is None:
            train_losses = epoch_history.get("train_loss", [])
        if val_losses is None:
            val_losses = epoch_history.get("val_loss", [])
        if train_aucs is None:
            train_aucs = epoch_history.get("train_auc", [])
        if val_aucs is None:
            val_aucs = epoch_history.get("val_auc", [])
        # Also get epochs list if available
        epochs_list = epoch_history.get("epochs", [])

    # ── Auto-compute metrics from predictions ──────────────────────
    if auc_score is None:
        from sklearn.metrics import roc_auc_score
        try:
            auc_score = float(roc_auc_score(y_true_arr, y_prob_arr))
        except Exception:
            pass

    if f1_score is None:
        f1_score = float(sklearn_f1(y_true_arr, y_pred_arr, average="weighted",
                                    zero_division=0))

    mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))

    # ECE
    try:
        prob_true_c, prob_pred_c = calibration_curve(y_true_arr, y_prob_arr, n_bins=10)
        bins_c  = np.linspace(0, 1, 11)
        bin_ids = np.digitize(y_prob_arr, bins_c[1:-1])
        ece     = float(sum(
            (bin_ids == b).mean() * abs(
                y_true_arr[bin_ids == b].mean() - y_prob_arr[bin_ids == b].mean()
            )
            for b in range(10) if (bin_ids == b).any()
        ))
    except Exception:
        ece = None

    # P@R90 and P@R95
    prec_curve, rec_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    def _prec_at_recall(target):
        try:
            idx = np.searchsorted(-rec_curve[::-1], -target)
            idx = len(rec_curve) - 1 - idx
            return float(prec_curve[idx]) if 0 <= idx < len(prec_curve) else None
        except Exception:
            return None

    prec_at_r90 = _prec_at_recall(0.90)
    prec_at_r95 = _prec_at_recall(0.95)

    # Per-class metrics
    per_class = {}
    for label, cls_name in [(0, "NORMAL"), (1, "PNEUMONIA")]:
        per_class[cls_name] = dict(
            precision = float(precision_score(y_true_arr, y_pred_arr,
                                              pos_label=label, zero_division=0)),
            recall    = float(recall_score(y_true_arr, y_pred_arr,
                                           pos_label=label, zero_division=0)),
            f1        = float(sklearn_f1(y_true_arr, y_pred_arr,
                                         pos_label=label, zero_division=0)),
        )

    # Sensitivity / specificity from confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else None
    specificity = float(tn / (tn + fp)) if (tn + fp) else None

    # Imbalance ratio
    imbalance_ratio = None
    if train_pneumonia and train_normal and train_normal > 0:
        imbalance_ratio = round(train_pneumonia / train_normal, 3)

    # ── Build the run record ───────────────────────────────────────
    runs = load_runs()
    
    # Build epoch_history dict if individual lists are provided
    epoch_history_dict = epoch_history
    if epoch_history_dict is None and (train_losses or val_losses or train_aucs or val_aucs):
        epoch_history_dict = {}
        if train_losses:
            epoch_history_dict["train_loss"] = list(train_losses)
        if val_losses:
            epoch_history_dict["val_loss"] = list(val_losses)
        if train_aucs:
            epoch_history_dict["train_auc"] = list(train_aucs)
        if val_aucs:
            epoch_history_dict["val_auc"] = list(val_aucs)
        # Add epochs
        if train_losses and len(train_losses) > 0:
            epoch_history_dict["epochs"] = list(range(1, len(train_losses) + 1))
    
    run  = dict(
        name    = run_name,
        model   = model_type,
        # core metrics
        accuracy    = accuracy,
        auc         = auc_score,
        f1          = f1_score,
        mcc         = mcc,
        val_loss    = val_loss,
        test_loss   = test_loss,
        sensitivity = sensitivity,
        specificity = specificity,
        ece         = ece,
        prec_at_recall_90 = prec_at_r90,
        prec_at_recall_95 = prec_at_r95,
        per_class_metrics = per_class,
        # epoch info
        epochs_trained       = epochs_trained,
        max_epochs           = max_epochs,
        early_stop_patience  = early_stop_patience,
        best_epoch           = best_epoch,
        # NEW: store full epoch history
        epoch_history        = epoch_history_dict,
        # timing
        train_time_secs = train_time_secs,
        threshold       = threshold,
        # hyperparams (structured)
        hyperparams = dict(
            lr                    = lr,
            lr_scheduler          = lr_scheduler,
            batch_size            = batch_size,
            optimizer             = optimizer,
            weight_decay          = weight_decay,
            img_size              = img_size,
            pretrained            = pretrained,
            freeze_backbone_epochs = freeze_backbone_epochs,
            augmentations         = augmentations,
            class_weighting       = class_weighting,
            dropout_rate          = dropout_rate,
        ),
        # dataset
        dataset = dict(
            train_total     = train_total,
            train_normal    = train_normal,
            train_pneumonia = train_pneumonia,
            val_total       = val_total,
            val_normal      = val_normal,
            val_pneumonia   = val_pneumonia,
            test_total      = test_total,
            test_normal     = test_normal,
            test_pneumonia  = test_pneumonia,
            preprocessing   = preprocessing,
            imbalance_ratio = imbalance_ratio,
        ),
        log_file = log_file,
        notes    = notes,
    )
    runs = add_run(runs, run)

    # ── Generate plots ─────────────────────────────────────────────
    print(f"\n📊  Generating diagnostic plots for '{run_name}' …")
    
    # If we have epoch_history, use it; otherwise use individual lists
    if epoch_history_dict and (train_losses is None and val_losses is None):
        # Use epoch_history directly
        plot_diagnostic_summary(
            y_true=y_true, y_pred=y_pred, y_prob=y_prob,
            epoch_history=epoch_history_dict,
            threshold=threshold,
            run_name=run_name,
            save=True,
        )
    else:
        # Use individual lists (backward compatible)
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