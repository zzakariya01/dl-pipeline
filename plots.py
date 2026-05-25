"""utils/plots.py — Reusable plotting helpers for the Streamlit app."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay


COLORS = plt.cm.tab10.colors


def plot_training_curves(histories: dict, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (name, hist) in enumerate(histories.items()):
        c = COLORS[i % 10]
        axes[0].plot(hist.get("auc", []),     color=c, lw=2,       label=f"{name} train")
        axes[0].plot(hist.get("val_auc", []), color=c, lw=2, ls="--", label=f"{name} val")
        axes[1].plot(hist.get("loss", []),     color=c, lw=2,       label=f"{name} train")
        axes[1].plot(hist.get("val_loss", []), color=c, lw=2, ls="--", label=f"{name} val")

    axes[0].set_title("AUC over Epochs"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_title("Loss over Epochs"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_roc_curves(results: dict, y_val, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_val, res["preds"])
        ax.plot(fpr, tpr, color=COLORS[i % 10], lw=2,
                label=f"{name}  AUC={res['auc']:.3f}")

    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {title}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_auc_bar(results: dict, title: str = "Model AUC Comparison"):
    labels = list(results.keys())
    aucs   = [v["auc"] for v in results.values()]
    best   = max(aucs)
    colors = ["#4C72B0" if a == best else "#AAAAAA" for a in aucs]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
    bars = ax.bar(labels, aucs, color=colors, edgecolor="white")
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{auc:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(max(0, min(aucs) - 0.05), 1.0)
    ax.set_ylabel("AUC"); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3); plt.xticks(rotation=20)
    plt.tight_layout()
    return fig


def plot_confusion(y_true, y_pred_proba, threshold: float = 0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax,
                                             cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_eda(df, target_col: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"EDA — {title}", fontsize=14, fontweight="bold")

    # 1. Target distribution
    df[target_col].value_counts().plot(kind="bar", ax=axes[0, 0],
                                        color=["#4C72B0", "#DD8452"])
    axes[0, 0].set_title("Target Distribution"); axes[0, 0].set_xlabel("")

    # 2. Missing values
    miss = df.isnull().sum()
    if miss.sum() > 0:
        miss[miss > 0].plot(kind="bar", ax=axes[0, 1], color="#C44E52")
        axes[0, 1].set_title("Missing Values")
    else:
        axes[0, 1].text(0.5, 0.5, "No Missing Values", ha="center", va="center", fontsize=12)
        axes[0, 1].set_title("Missing Values")

    # 3. Correlation heatmap
    num_df = df.select_dtypes(include="number")
    top_cols = num_df.corr()[target_col].abs().nlargest(min(8, len(num_df.columns))).index
    sns.heatmap(num_df[top_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[0, 2], cbar=False, annot_kws={"size": 7})
    axes[0, 2].set_title("Correlation (Top Features)")

    # 4-6. Feature distributions
    feat_cols = [c for c in num_df.columns if c != target_col][:3]
    for i, col in enumerate(feat_cols):
        ax = axes[1, i]
        for val in sorted(df[target_col].unique()):
            subset = df[df[target_col] == val][col].dropna()
            ax.hist(subset, alpha=0.6, bins=25, label=f"{target_col}={val}")
        ax.set_title(f"Distribution: {col}"); ax.legend(fontsize=7)

    plt.tight_layout()
    return fig
