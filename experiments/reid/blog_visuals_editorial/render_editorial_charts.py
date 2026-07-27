#!/usr/bin/env python3
"""Render publication-ready figures with exact experiment metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent

BG = "#0b0d0e"
PANEL = "#111416"
GRID = "#2a2e30"
TEXT = "#f2f4f3"
MUTED = "#9aa1a3"
GREEN = "#76b900"
GREEN_DARK = "#345300"
WHITE_BAR = "#dfe5e2"
RED = "#e56868"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.28,
    )
    plt.close(fig)


def figure_ltcc_impact() -> None:
    experiments = [
        "Real only",
        "+ unfiltered\nsynthetic",
        "+ filtered\nsynthetic",
    ]
    map_scores = np.array([23.8, 43.1, 43.8])
    rank1_scores = np.array([50.3, 74.4, 76.1])
    x = np.arange(len(experiments))
    width = 0.28

    fig, ax = plt.subplots(figsize=(14.4, 8.1))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.77, bottom=0.17)

    highlight = FancyBboxPatch(
        (1.55, -3.5),
        0.9,
        88.5,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        linewidth=1.1,
        edgecolor=GREEN_DARK,
        facecolor="#10170d",
        alpha=0.95,
        zorder=0,
    )
    ax.add_patch(highlight)

    bars_map = ax.bar(
        x - width / 2,
        map_scores,
        width,
        color=WHITE_BAR,
        edgecolor="none",
        label="mAP",
        zorder=3,
    )
    bars_rank1 = ax.bar(
        x + width / 2,
        rank1_scores,
        width,
        color=GREEN,
        edgecolor="none",
        label="Rank-1",
        zorder=3,
    )

    for bars in (bars_map, bars_rank1):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
                color=TEXT,
            )

    ax.set_ylim(0, 86)
    ax.set_yticks(np.arange(0, 81, 20))
    ax.set_ylabel("Score (%)", labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=14)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.08),
        ncols=2,
        frameon=False,
        fontsize=14,
        handlelength=1.5,
    )

    fig.text(
        0.08,
        0.925,
        "LTCC: Synthetic Data Impact",
        fontsize=28,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.08,
        0.875,
        "Swin ReID | Evaluation uses the real LTCC query and gallery split",
        fontsize=13,
        color=MUTED,
        ha="left",
    )

    fig.text(
        0.705,
        0.925,
        "+20.0",
        fontsize=31,
        fontweight="bold",
        color=GREEN,
        ha="left",
    )
    fig.text(
        0.82,
        0.925,
        "mAP points",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        ha="left",
    )
    fig.text(
        0.705,
        0.884,
        "+25.8 Rank-1 points vs. real only",
        fontsize=13,
        color=TEXT,
        ha="left",
    )
    fig.text(
        0.705,
        0.846,
        "6,152 crops | 2.63% of the original synthetic pool",
        fontsize=12,
        color=MUTED,
        ha="left",
    )

    save(fig, "figure_5_editorial_ltcc_impact.png")


def draw_dumbbell_panel(
    ax: plt.Axes,
    title: str,
    real: np.ndarray,
    filtered: np.ndarray,
) -> None:
    datasets = ["LTCC", "PRCC", "Duke"]
    y = np.arange(len(datasets))[::-1]

    for yi, baseline, augmented in zip(y, real, filtered):
        delta = augmented - baseline
        ax.plot(
            [baseline, augmented],
            [yi, yi],
            color=GREEN if delta >= 0 else RED,
            linewidth=4,
            solid_capstyle="round",
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            baseline,
            yi,
            s=155,
            facecolor=BG,
            edgecolor=WHITE_BAR,
            linewidth=2.5,
            zorder=3,
        )
        ax.scatter(
            augmented,
            yi,
            s=175,
            facecolor=GREEN,
            edgecolor=GREEN,
            linewidth=1,
            zorder=4,
        )
        ax.text(
            baseline,
            yi + 0.18,
            f"{baseline:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color=WHITE_BAR,
        )
        ax.text(
            augmented,
            yi - 0.2,
            f"{augmented:.1f}",
            ha="center",
            va="top",
            fontsize=12,
            color=GREEN,
            fontweight="bold",
        )

        change_color = GREEN if delta > 0 else RED if delta < 0 else MUTED
        label_x = max(baseline, augmented) + 4.0
        ax.text(
            label_x,
            yi,
            f"{delta:+.1f}",
            ha="left",
            va="center",
            fontsize=13,
            color=change_color,
            fontweight="bold",
        )

    ax.set_title(title, loc="left", fontsize=19, fontweight="bold", pad=18)
    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.65, 2.65)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Score (%)", labelpad=10)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0)


def figure_cross_domain_impact() -> None:
    map_real = np.array([23.8, 72.1, 89.0])
    map_filtered = np.array([43.8, 72.4, 86.4])
    rank1_real = np.array([50.3, 98.6, 90.6])
    rank1_filtered = np.array([76.1, 98.6, 89.7])

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 8.1))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.67, bottom=0.16, wspace=0.25)

    draw_dumbbell_panel(axes[0], "mAP", map_real, map_filtered)
    draw_dumbbell_panel(axes[1], "Rank-1", rank1_real, rank1_filtered)

    fig.text(
        0.08,
        0.925,
        "Synthetic Augmentation Is Domain-Dependent",
        fontsize=28,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.08,
        0.875,
        "Real-only Swin checkpoint vs. the corresponding model trained with the same filtered synthetic subset",
        fontsize=13,
        color=MUTED,
        ha="left",
    )
    fig.text(
        0.08,
        0.83,
        "Each result uses that benchmark's real query and gallery split",
        fontsize=12,
        color=MUTED,
        ha="left",
    )

    legend_y = 0.765
    fig.add_artist(
        Rectangle(
            (0.08, legend_y),
            0.012,
            0.012,
            transform=fig.transFigure,
            facecolor=BG,
            edgecolor=WHITE_BAR,
            linewidth=1.5,
        )
    )
    fig.text(0.098, legend_y - 0.001, "Real only", fontsize=12, color=TEXT)
    fig.add_artist(
        Rectangle(
            (0.175, legend_y),
            0.012,
            0.012,
            transform=fig.transFigure,
            facecolor=GREEN,
            edgecolor=GREEN,
        )
    )
    fig.text(
        0.193,
        legend_y - 0.001,
        "Real + filtered synthetic",
        fontsize=12,
        color=TEXT,
    )

    fig.text(
        0.08,
        0.055,
        "Largest benefit: LTCC clothing-change retrieval. PRCC is nearly unchanged; Duke favors its real-only model.",
        fontsize=13,
        color=TEXT,
        ha="left",
    )

    save(fig, "figure_6_editorial_cross_domain_impact.png")


def main() -> None:
    configure()
    figure_ltcc_impact()
    figure_cross_domain_impact()


if __name__ == "__main__":
    main()
