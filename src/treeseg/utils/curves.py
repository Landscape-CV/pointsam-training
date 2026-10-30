"""
curves.py - turn a run's history.csv into PNG curves and a per-epoch CSV.

Each figure has one y-axis: loss and IoU are on different scales, so they get separate charts.
Epochs are on the top axis of the step charts, since one x position has both meanings.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _epoch_axis(ax, history: pd.DataFrame) -> None:
    """Add an epoch scale on top of a chart whose x axis is steps."""
    steps_per_epoch = history["step"].max() / max(history["epoch"].max(), 1e-9)
    top = ax.secondary_xaxis("top", functions=(lambda s: s / steps_per_epoch, lambda e: e * steps_per_epoch))
    top.set_xlabel("epoch", color=MUTED)
    top.tick_params(colors=MUTED)
    top.spines["top"].set_color(GRID)


def per_epoch(history: pd.DataFrame) -> pd.DataFrame:
    """Mean training loss per whole epoch, next to the validation at or before that epoch's end."""
    train = history[history["split"] == "train"].copy()
    val = history[history["split"] == "val"].sort_values("epoch")
    train["whole_epoch"] = train["epoch"].apply(lambda e: int(e) + 1 if e % 1 else int(e))
    table = train.groupby("whole_epoch", as_index=False).agg(train_loss=("loss", "mean"), train_iou_first=("iou_first", "mean"))
    if len(val):
        columns = ["epoch", "val_loss", "val_iou", "val_iou_first", "val_iou_pick"]
        table = pd.merge_asof(table.astype({"whole_epoch": float}), val[columns], left_on="whole_epoch",
                              right_on="epoch", direction="backward").drop(columns="epoch")
    return table


def plot_training_loss(history: pd.DataFrame, path: str, window: int = 8) -> None:
    train = history[history["split"] == "train"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    ax.plot(train["step"], train["loss"], color=BLUE, alpha=0.25, linewidth=1, label="each logged step")
    smooth = train["loss"].rolling(window, min_periods=1).mean()
    ax.plot(train["step"], smooth, color=BLUE, linewidth=2, label=f"average of last {window}")
    _style(ax, "Training loss", "step", "loss")
    _epoch_axis(ax, history)
    ax.legend(frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_validation_iou(history: pd.DataFrame, path: str, metric: str = "val_iou_pick") -> None:
    val = history[history["split"] == "val"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    series = [("val_iou_pick", "Model's own pick, 1 click", BLUE, "o"), ("val_iou_first", "Best of 3 masks, 1 click", ORANGE, "s"),
              ("val_iou", "After correction clicks", AQUA, "^")]
    for column, label, color, marker in series:
        ax.plot(val["step"], val[column], color=color, linewidth=2, marker=marker, markersize=6, label=label)
    if len(val):
        best = val.loc[val[metric].idxmax()]
        ax.annotate(f"best {best[metric]:.3f}", (best["step"], best[metric]), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    _style(ax, "Validation IoU (val strip, clicks placed using true labels)", "step", "IoU")
    _epoch_axis(ax, history)
    ax.legend(frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_epoch_loss(table: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    ax.plot(table["whole_epoch"], table["train_loss"], color=BLUE, linewidth=2, marker="o", markersize=6, label="Training (epoch mean)")
    if "val_loss" in table:
        ax.plot(table["whole_epoch"], table["val_loss"], color=ORANGE, linewidth=2, marker="s", markersize=6, label="Validation (latest)")
    _style(ax, "Loss by epoch", "epoch", "loss")
    ax.legend(frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_all(history_path: str, out_dir: str) -> dict:
    """Write every figure and the per-epoch table. Safe on a partial history, so it works mid-run."""
    os.makedirs(out_dir, exist_ok=True)
    history = pd.read_csv(history_path)
    table = per_epoch(history)
    paths = {"training_loss": os.path.join(out_dir, "training_loss.png"),
             "validation_iou": os.path.join(out_dir, "validation_iou.png"),
             "epoch_loss": os.path.join(out_dir, "epoch_loss.png"),
             "per_epoch": os.path.join(out_dir, "per_epoch.csv")}
    plot_training_loss(history, paths["training_loss"])
    plot_validation_iou(history, paths["validation_iou"])
    plot_epoch_loss(table, paths["epoch_loss"])
    table.to_csv(paths["per_epoch"], index=False)
    return paths


def plot_f1_bars(summary: pd.DataFrame, path: str, rule: str = "ignore_non_tree") -> None:
    """Mean F1 per click source, stock next to fine-tuned. Each bar is labelled, so colour isn't needed to read it."""
    table = summary[summary["rule"] == rule]
    sources = [s for s in ("grid", "stems", "perfect") if s in set(table["clicks_from"])]
    checkpoints = list(dict.fromkeys(table["checkpoint"]))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    width = 0.8 / max(len(checkpoints), 1)
    for k, (checkpoint, color) in enumerate(zip(checkpoints, (MUTED, BLUE, ORANGE))):
        values = [table[(table["checkpoint"] == checkpoint) & (table["clicks_from"] == s)]["f1"].mean() for s in sources]
        positions = [i + (k - (len(checkpoints) - 1) / 2) * width for i in range(len(sources))]
        ax.bar(positions, values, width * 0.92, color=color, label=checkpoint)
        for x, value in zip(positions, values):
            ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(sources)), [f"{s} clicks" for s in sources])
    _style(ax, f"Instance F1 on the test strip, IoU 0.5 ({rule.replace('_', ' ')})", "", "F1")
    ax.legend(frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
