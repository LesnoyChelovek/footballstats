"""
Generates static PNG plots using matplotlib and prints football match statistics.
Uses stats_core for shared models, parsing, and aggregations.
"""

from __future__ import annotations

import logging
import os
import matplotlib
import matplotlib.pyplot as plt

import stats_core
from stats_core import (
    DEFAULT_JSON_FOLDER,
    DEFAULT_PLOTS_FOLDER,
    TOP_SCORES_COUNT,
    AggregatedStats,
    aggregate_dataset,
    format_console_summary,
)

# --- Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
PLOT_STYLE = "ggplot"
FONT_SIZE = 12
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
XTICK_FONT_SIZE = 10
YTICK_FONT_SIZE = 10
COLOR_PALETTE = [
    "#5cb85c",
    "#5bc0de",
    "#d9534f",
    "#f0ad4e",
    "#428bca",
    "#9467bd",
    "#e377c2",
]

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# --- Matplotlib Style Setup ---
plt.style.use(PLOT_STYLE)
matplotlib.rcParams["font.size"] = FONT_SIZE
matplotlib.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
matplotlib.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
matplotlib.rcParams["xtick.labelsize"] = XTICK_FONT_SIZE
matplotlib.rcParams["ytick.labelsize"] = YTICK_FONT_SIZE


def _annotate_processed_matches(processed_matches_count: int) -> None:
    """Adds annotation about the number of processed matches to the plot."""
    plt.text(
        0.95,
        0.95,
        f"Обработано матчей: {processed_matches_count:,}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )


def plot_bar_chart(
    data: dict[str, float],
    title: str,
    xlabel: str,
    ylabel: str,
    plot_filename: str,
    processed_matches_count: int,
    y_range: tuple[int, int] = None,
) -> None:
    """Generates and saves a bar chart plot."""
    plt.figure(figsize=(8, 6))
    labels = list(data.keys())
    values = list(data.values())
    bars = plt.bar(labels, values, color=COLOR_PALETTE[1:4], alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_range:
        plt.ylim(y_range)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    _annotate_processed_matches(processed_matches_count)
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()


def plot_top_scores_chart(
    top_scores: list[tuple[str, int]],
    plot_folder: str,
    processed_matches_count: int,
) -> None:
    """Generates and saves a horizontal bar chart for top match scores."""
    scores = [score for score, _ in top_scores]
    counts = [count for _, count in top_scores]

    plt.figure(figsize=(10, 7))
    plt.barh(scores, counts, color=COLOR_PALETTE[4], alpha=0.7)
    plt.title(f"Топ-{len(top_scores)} самых часто встречающихся счетов матчей")
    plt.xlabel("Количество матчей")
    plt.ylabel("Счет матча")
    for index, value in enumerate(counts):
        plt.text(value + 10, index, f"{value:,}", va="center", fontsize=10)

    _annotate_processed_matches(processed_matches_count)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_filename = os.path.join(plot_folder, "top_scores.png")
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()


def plot_minute_distribution_chart(
    regular_dict: dict[int, int],
    stoppage_dict: dict[int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    plot_filename: str,
    processed_matches_count: int,
    base_minute: int = 45,
) -> None:
    """Plots goal minute distribution with stoppage time displayed separately."""
    plt.figure(figsize=(12, 6))

    if base_minute == 45:
        reg_mins = list(range(1, 46))
        reg_counts = [regular_dict.get(m, 0) for m in reg_mins]
        stop_keys = sorted(stoppage_dict.keys())
        stop_labels = [f"45+{k}" for k in stop_keys]
        stop_counts = [stoppage_dict[k] for k in stop_keys]
        labels = [str(m) for m in reg_mins] + stop_labels
        counts = reg_counts + stop_counts
        colors = [COLOR_PALETTE[0]] * len(reg_mins) + [COLOR_PALETTE[3]] * len(
            stop_keys
        )
    else:
        reg_mins = list(range(46, 91))
        reg_counts = [regular_dict.get(m, 0) for m in reg_mins]
        stop_keys = sorted(stoppage_dict.keys())
        stop_labels = [f"90+{k}" for k in stop_keys]
        stop_counts = [stoppage_dict[k] for k in stop_keys]
        labels = [str(m) for m in reg_mins] + stop_labels
        counts = reg_counts + stop_counts
        colors = [COLOR_PALETTE[4]] * len(reg_mins) + [COLOR_PALETTE[3]] * len(
            stop_keys
        )

    x_positions = range(len(labels))
    plt.bar(x_positions, counts, color=colors, alpha=0.8, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show ticks every 5 steps
    step = 5
    tick_indices = [
        i for i in range(0, len(labels), step) if i < len(labels)
    ] + [len(labels) - 1]
    plt.xticks(
        tick_indices,
        [labels[i] for i in tick_indices],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    plt.grid(axis="y", linestyle="--")
    _annotate_processed_matches(processed_matches_count)
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()


def plot_diff_histogram(
    diffs: list[int],
    step: int,
    max_val: int,
    title: str,
    xlabel: str,
    ylabel: str,
    plot_filename: str,
    processed_matches_count: int,
) -> None:
    """Plots goal time difference histogram using pre-binned categories."""
    plt.figure(figsize=(10, 6))
    bins = [f"{i}–{i+step-1}" for i in range(0, max_val, step)]
    counts = [0] * len(bins)
    for d in diffs:
        if d <= max_val:
            idx = min(d // step, len(bins) - 1)
            counts[idx] += 1

    x_pos = range(len(bins))
    plt.bar(
        x_pos, counts, color=COLOR_PALETTE[1], edgecolor="black", alpha=0.7
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x_pos[::2], bins[::2], rotation=45, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--")
    _annotate_processed_matches(processed_matches_count)
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()


def analyze_football_stats(
    json_folder: str = DEFAULT_JSON_FOLDER,
    plots_folder: str = DEFAULT_PLOTS_FOLDER,
) -> AggregatedStats:
    """Analyzes football match data, prints statistics, and generates static matplotlib PNG plots."""
    os.makedirs(plots_folder, exist_ok=True)
    stats = aggregate_dataset(json_folder)
    print(format_console_summary(stats, top_n=TOP_SCORES_COUNT))

    plot_minute_distribution_chart(
        stats.first_half_regular,
        stats.first_half_stoppage,
        title="Распределение минут голов в первом тайме",
        xlabel="Минута",
        ylabel="Количество голов",
        plot_filename=os.path.join(plots_folder, "first_half_goals_minutes.png"),
        processed_matches_count=stats.minute_matches_count,
        base_minute=45,
    )

    plot_minute_distribution_chart(
        stats.second_half_regular,
        stats.second_half_stoppage,
        title="Распределение минут голов во втором тайме",
        xlabel="Минута",
        ylabel="Количество голов",
        plot_filename=os.path.join(
            plots_folder, "second_half_goals_minutes.png"
        ),
        processed_matches_count=stats.minute_matches_count,
        base_minute=90,
    )

    plot_diff_histogram(
        stats.goal_differences,
        step=5,
        max_val=90,
        title="Распределение разницы в минутах между первым и вторым голом",
        xlabel="Разница в минутах",
        ylabel="Количество матчей",
        plot_filename=os.path.join(plots_folder, "goal_difference.png"),
        processed_matches_count=len(stats.goal_differences),
    )

    plot_diff_histogram(
        stats.first_half_goal_differences,
        step=2,
        max_val=45,
        title="Распределение разницы в минутах между первым и вторым голом (Первый тайм)",
        xlabel="Разница в минутах",
        ylabel="Количество матчей",
        plot_filename=os.path.join(
            plots_folder, "first_half_goal_difference.png"
        ),
        processed_matches_count=len(stats.first_half_goal_differences),
    )

    probs = stats.goal_probabilities
    if probs:
        plot_bar_chart(
            probs,
            title="Вероятность забития последующих голов после первого",
            xlabel="Следующий гол",
            ylabel="Вероятность (%)",
            plot_filename=os.path.join(plots_folder, "goal_probabilities.png"),
            processed_matches_count=stats.matches_with_first_goal,
            y_range=(0, 100),
        )

    top_scores = stats.top_scores(top_n=TOP_SCORES_COUNT)
    plot_top_scores_chart(top_scores, plots_folder, stats.total_matches)

    return stats


if __name__ == "__main__":
    analyze_football_stats()