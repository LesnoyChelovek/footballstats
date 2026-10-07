"""
Interactive HTML dashboard and Plotly chart generator for football match statistics.
Builds an all-in-one responsive web dashboard (plots/report.html) and optional standalone charts.
Uses stats_core for all data structures, parsing, and aggregations.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from html import escape
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import stats_core
from stats_core import (
    DEFAULT_JSON_FOLDER,
    DEFAULT_PLOTS_FOLDER,
    TOP_SCORES_COUNT,
    AggregatedStats,
    aggregate_dataset,
    format_console_summary,
)

logger = logging.getLogger("footballstats.html")

COLOR_REGULAR = "#5cb85c"  # Green
COLOR_STOPPAGE = "#f0ad4e"  # Amber/Orange
COLOR_BAR_ALT = "#5bc0de"  # Sky blue
COLOR_ACCENT = "#d9534f"  # Red/Coral
COLOR_NAVY = "#428bca"  # Primary blue
COLOR_PURPLE = "#9467bd"  # Purple
BG_COLOR = "#fdfdfd"
GRID_COLOR = "#e9ecef"


def _create_base_layout(
    title: str,
    subtitle: str,
    xaxis_title: str,
    yaxis_title: str,
    barmode: str = "group",
) -> dict:
    """Returns standard Plotly layout configuration with clean subtitle and margins.

    The title is pinned to the top of the figure. The legend sits just above the
    plot, in the gap under the subtitle, so the two no longer share one row.
    """
    return dict(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size: 13px; color: #6c757d;'>{subtitle}</span>",
            x=0.5,
            xanchor="center",
            y=1,
            yanchor="top",
            yref="container",
            pad=dict(t=4, b=14),
            font=dict(size=18, family="sans-serif", color="#212529"),
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, standoff=12),
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(size=11),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, standoff=8),
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(size=11),
            automargin=True,
        ),
        barmode=barmode,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor="#ffffff",
        height=520,
        margin=dict(l=64, r=28, b=64, t=132),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
    )


def build_first_half_chart(stats: AggregatedStats) -> go.Figure:
    """
    Builds the First Half goal minute distribution bar chart.
    Separates regular minutes (1..45) and stoppage time (45+1..45+N).
    Includes an in-chart toggle between absolute goal counts and percentages.
    """
    total_goals = stats.total_first_half_goals
    regular_minutes = list(range(1, 46))
    stoppage_minutes = sorted(stats.first_half_stoppage.keys())

    # X-axis categories
    x_categories = [str(m) for m in regular_minutes] + [
        f"45+{m}" for m in stoppage_minutes
    ]
    num_reg = len(regular_minutes)
    num_stop = len(stoppage_minutes)

    # Absolute values
    y_reg_abs: list[Optional[int]] = [
        stats.first_half_regular.get(m, 0) for m in regular_minutes
    ] + [None] * num_stop
    y_stop_abs: list[Optional[int]] = [None] * num_reg + [
        stats.first_half_stoppage.get(m, 0) for m in stoppage_minutes
    ]

    # Percentages
    y_reg_pct: list[Optional[float]] = [
        round((stats.first_half_regular.get(m, 0) / total_goals * 100), 2)
        if total_goals > 0
        else 0.0
        for m in regular_minutes
    ] + [None] * num_stop
    y_stop_pct: list[Optional[float]] = [None] * num_reg + [
        round((stats.first_half_stoppage.get(m, 0) / total_goals * 100), 2)
        if total_goals > 0
        else 0.0
        for m in stoppage_minutes
    ]

    # Tooltip custom data: [pct, note]
    custom_reg = []
    for m in regular_minutes:
        c = stats.first_half_regular.get(m, 0)
        p = round((c / total_goals * 100), 2) if total_goals else 0.0
        note = (
            "<br><span style='color:#e67e22;'>⚠️ Включает неразмещенное добавленное время</span>"
            if m == 45
            else ""
        )
        custom_reg.append([p, note])
    custom_reg += [[0.0, ""]] * num_stop

    custom_stop = [[0.0, ""]] * num_reg
    for m in stoppage_minutes:
        c = stats.first_half_stoppage.get(m, 0)
        p = round((c / total_goals * 100), 2) if total_goals else 0.0
        note = "<br><span style='color:#f39c12;'>⏱ Добавленное время 1-го тайма</span>"
        custom_stop.append([p, note])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Основное время (1–45')",
            x=x_categories,
            y=y_reg_abs,
            marker_color=COLOR_REGULAR,
            opacity=0.85,
            customdata=custom_reg,
            hovertemplate="<b>%{x}-я минута</b><br>Голов: <b>%{y:,}</b> (%{customdata[0]:.2f}%)%{customdata[1]}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Компенсированное время (45+')",
            x=x_categories,
            y=y_stop_abs,
            marker_color=COLOR_STOPPAGE,
            opacity=0.85,
            customdata=custom_stop,
            hovertemplate="<b>%{x} (компенсация)</b><br>Голов: <b>%{y:,}</b> (%{customdata[0]:.2f}%)%{customdata[1]}<extra></extra>",
        )
    )

    subtitle = f"Обработано матчей: {stats.minute_matches_count:,} | Всего голов в 1-м тайме: {total_goals:,}"
    layout_cfg = _create_base_layout(
        "Распределение минут голов в первом тайме",
        subtitle,
        "Минута матча",
        "Количество голов",
        barmode="stack",
    )

    # Client-side toggle between Absolute and Percentage
    layout_cfg["updatemenus"] = [
        dict(
            type="buttons",
            direction="left",
            x=0.0,
            xanchor="left",
            y=1.0,
            yanchor="bottom",
            pad=dict(b=6),
            showactive=True,
            buttons=[
                dict(
                    label="Голы (абс.)",
                    method="update",
                    args=[
                        {"y": [y_reg_abs, y_stop_abs]},
                        {
                            "yaxis": {
                                "title": "Количество голов",
                                "gridcolor": GRID_COLOR,
                                "automargin": True,
                            }
                        },
                    ],
                ),
                dict(
                    label="Доля (%)",
                    method="update",
                    args=[
                        {"y": [y_reg_pct, y_stop_pct]},
                        {
                            "yaxis": {
                                "title": "Доля от голов в тайме (%)",
                                "ticksuffix": "%",
                                "gridcolor": GRID_COLOR,
                                "automargin": True,
                            }
                        },
                    ],
                ),
            ],
        )
    ]
    # Dozens of 45+N categories overlap when the labels stay horizontal.
    layout_cfg["xaxis"]["type"] = "category"
    layout_cfg["xaxis"]["tickangle"] = -90
    layout_cfg["xaxis"]["tickfont"] = dict(size=10)
    layout_cfg["margin"]["b"] = 108
    layout_cfg["height"] = 640
    fig.update_layout(layout_cfg)
    return fig


def build_second_half_chart(stats: AggregatedStats) -> go.Figure:
    """
    Builds the Second Half goal minute distribution bar chart.
    Separates regular minutes (46..90) and stoppage time (90+1..90+N).
    Includes an in-chart toggle between absolute goal counts and percentages.
    """
    total_goals = stats.total_second_half_goals
    regular_minutes = list(range(46, 91))
    stoppage_minutes = sorted(stats.second_half_stoppage.keys())

    x_categories = [str(m) for m in regular_minutes] + [
        f"90+{m}" for m in stoppage_minutes
    ]
    num_reg = len(regular_minutes)
    num_stop = len(stoppage_minutes)

    y_reg_abs: list[Optional[int]] = [
        stats.second_half_regular.get(m, 0) for m in regular_minutes
    ] + [None] * num_stop
    y_stop_abs: list[Optional[int]] = [None] * num_reg + [
        stats.second_half_stoppage.get(m, 0) for m in stoppage_minutes
    ]

    y_reg_pct: list[Optional[float]] = [
        round((stats.second_half_regular.get(m, 0) / total_goals * 100), 2)
        if total_goals > 0
        else 0.0
        for m in regular_minutes
    ] + [None] * num_stop
    y_stop_pct: list[Optional[float]] = [None] * num_reg + [
        round((stats.second_half_stoppage.get(m, 0) / total_goals * 100), 2)
        if total_goals > 0
        else 0.0
        for m in stoppage_minutes
    ]

    custom_reg = []
    for m in regular_minutes:
        c = stats.second_half_regular.get(m, 0)
        p = round((c / total_goals * 100), 2) if total_goals else 0.0
        note = (
            "<br><span style='color:#e67e22;'>⚠️ Включает неразмещенное добавленное время</span>"
            if m == 90
            else ""
        )
        custom_reg.append([p, note])
    custom_reg += [[0.0, ""]] * num_stop

    custom_stop = [[0.0, ""]] * num_reg
    for m in stoppage_minutes:
        c = stats.second_half_stoppage.get(m, 0)
        p = round((c / total_goals * 100), 2) if total_goals else 0.0
        note = "<br><span style='color:#f39c12;'>⏱ Добавленное время 2-го тайма</span>"
        custom_stop.append([p, note])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Основное время (46–90')",
            x=x_categories,
            y=y_reg_abs,
            marker_color=COLOR_NAVY,
            opacity=0.85,
            customdata=custom_reg,
            hovertemplate="<b>%{x}-я минута</b><br>Голов: <b>%{y:,}</b> (%{customdata[0]:.2f}%)%{customdata[1]}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Компенсированное время (90+')",
            x=x_categories,
            y=y_stop_abs,
            marker_color=COLOR_STOPPAGE,
            opacity=0.85,
            customdata=custom_stop,
            hovertemplate="<b>%{x} (компенсация)</b><br>Голов: <b>%{y:,}</b> (%{customdata[0]:.2f}%)%{customdata[1]}<extra></extra>",
        )
    )

    subtitle = f"Обработано матчей: {stats.minute_matches_count:,} | Всего голов во 2-м тайме: {total_goals:,}"
    layout_cfg = _create_base_layout(
        "Распределение минут голов во втором тайме",
        subtitle,
        "Минута матча",
        "Количество голов",
        barmode="stack",
    )

    layout_cfg["updatemenus"] = [
        dict(
            type="buttons",
            direction="left",
            x=0.0,
            xanchor="left",
            y=1.0,
            yanchor="bottom",
            pad=dict(b=6),
            showactive=True,
            buttons=[
                dict(
                    label="Голы (абс.)",
                    method="update",
                    args=[
                        {"y": [y_reg_abs, y_stop_abs]},
                        {
                            "yaxis": {
                                "title": "Количество голов",
                                "gridcolor": GRID_COLOR,
                                "automargin": True,
                            }
                        },
                    ],
                ),
                dict(
                    label="Доля (%)",
                    method="update",
                    args=[
                        {"y": [y_reg_pct, y_stop_pct]},
                        {
                            "yaxis": {
                                "title": "Доля от голов в тайме (%)",
                                "ticksuffix": "%",
                                "gridcolor": GRID_COLOR,
                                "automargin": True,
                            }
                        },
                    ],
                ),
            ],
        )
    ]
    # Same crowding as the first-half chart: 90+N labels collide in one row.
    layout_cfg["xaxis"]["type"] = "category"
    layout_cfg["xaxis"]["tickangle"] = -90
    layout_cfg["xaxis"]["tickfont"] = dict(size=10)
    layout_cfg["margin"]["b"] = 108
    layout_cfg["height"] = 640
    fig.update_layout(layout_cfg)
    return fig


def build_goal_diff_chart(stats: AggregatedStats) -> go.Figure:
    """Pre-aggregated histogram of time difference between 1st and 2nd goal."""
    step = 5
    max_diff = 90
    bins = [f"{i}–{i+step-1}" for i in range(0, max_diff, step)] + [
        f"{max_diff}+"
    ]
    counts = [0] * len(bins)

    for d in stats.goal_differences:
        idx = min(d // step, len(bins) - 1)
        counts[idx] += 1

    total_diffs = sum(counts)
    pcts = [round(c / total_diffs * 100, 2) if total_diffs else 0 for c in counts]

    fig = go.Figure(
        data=[
            go.Bar(
                x=bins,
                y=counts,
                marker_color=COLOR_BAR_ALT,
                opacity=0.85,
                customdata=pcts,
                hovertemplate="Разница: <b>%{x} мин</b><br>Матчей: <b>%{y:,}</b> (%{customdata:.2f}%)<extra></extra>",
            )
        ]
    )
    subtitle = f"Обработано матчей с ≥2 голами: {total_diffs:,}"
    fig.update_layout(
        _create_base_layout(
            "Разница в минутах между первым и вторым голом",
            subtitle,
            "Разница во времени (минуты)",
            "Количество матчей",
        )
    )
    # Horizontal labels collide in the half-width card.
    fig.update_xaxes(tickangle=-75, tickfont=dict(size=10), automargin=True)
    return fig


def build_first_half_goal_diff_chart(stats: AggregatedStats) -> go.Figure:
    """Pre-aggregated histogram of time difference between goals in 1st half."""
    step = 2
    max_diff = 45
    bins = [f"{i}–{i+step-1}" for i in range(0, max_diff, step)]
    counts = [0] * len(bins)

    for d in stats.first_half_goal_differences:
        if d <= max_diff:
            idx = min(d // step, len(bins) - 1)
            counts[idx] += 1

    total_diffs = sum(counts)
    pcts = [round(c / total_diffs * 100, 2) if total_diffs else 0 for c in counts]

    fig = go.Figure(
        data=[
            go.Bar(
                x=bins,
                y=counts,
                marker_color=COLOR_PURPLE,
                opacity=0.85,
                customdata=pcts,
                hovertemplate="Разница: <b>%{x} мин</b><br>Матчей: <b>%{y:,}</b> (%{customdata:.2f}%)<extra></extra>",
            )
        ]
    )
    subtitle = f"Обработано матчей с ≥2 голами в 1-м тайме: {total_diffs:,}"
    fig.update_layout(
        _create_base_layout(
            "Разница в минутах между первым и вторым голом<br>(первый тайм)",
            subtitle,
            "Разница во времени (минуты)",
            "Количество матчей",
        )
    )
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=10), automargin=True)
    return fig


def build_probabilities_chart(stats: AggregatedStats) -> go.Figure:
    """Bar chart for subsequent goal scoring probabilities."""
    probs = stats.goal_probabilities
    labels = list(probs.keys())
    values = list(probs.values())

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=[COLOR_BAR_ALT, COLOR_STOPPAGE, COLOR_ACCENT],
                opacity=0.85,
                text=[f"{v:.2f}%" for v in values],
                textposition="outside",
                hovertemplate="Цель: <b>%{x}</b><br>Вероятность: <b>%{y:.2f}%</b><extra></extra>",
            )
        ]
    )
    subtitle = (
        f"База расчета: {stats.matches_with_first_goal:,} матчей с первым голом"
    )
    layout_cfg = _create_base_layout(
        "Вероятность забития последующих голов после первого",
        subtitle,
        "Событие",
        "Вероятность (%)",
    )
    layout_cfg["yaxis"]["range"] = [0, 100]
    fig.update_layout(layout_cfg)
    return fig


def build_top_scores_chart(
    stats: AggregatedStats, top_n: int = TOP_SCORES_COUNT
) -> go.Figure:
    """
    Horizontal bar chart for top N most frequent match scores.
    P1 FIX: Top score is placed at the top of the chart, descending downwards.
    Title uses top_n dynamically.
    """
    top_scores = stats.top_scores(top_n=top_n)
    scores = [s for s, _ in top_scores]
    counts = [c for _, c in top_scores]
    total_m = stats.total_matches
    pcts = [round(c / total_m * 100, 2) if total_m else 0 for c in counts]

    fig = go.Figure(
        data=[
            go.Bar(
                y=scores,
                x=counts,
                orientation="h",
                marker_color=COLOR_NAVY,
                opacity=0.85,
                text=[f"{c:,} ({p}%)" for c, p in zip(counts, pcts)],
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(size=12, color="#ffffff"),
                cliponaxis=False,
                customdata=pcts,
                hovertemplate="Счет: <b>%{y}</b><br>Матчей: <b>%{x:,}</b><br>Доля от всех матчей: <b>%{customdata:.2f}%</b><extra></extra>",
            )
        ]
    )

    title = f"Топ-{len(top_scores)} самых часто встречающихся счетов матчей"
    subtitle = f"Всего обработано матчей: {total_m:,} (включая матчи без поминутной статистики)"
    layout_cfg = _create_base_layout(
        title, subtitle, "Количество матчей", "Счет матча"
    )
    # yaxis reversed autorange ensures the highest count (first element in array) is at the top.
    # Labels sit inside the bars: outside text was clipped by the plot area in a half-width card.
    layout_cfg["yaxis"]["autorange"] = "reversed"
    layout_cfg["margin"]["r"] = 24
    fig.update_layout(layout_cfg)
    return fig


def build_yearly_chart(stats: AggregatedStats) -> go.Figure:
    """
    Historical breakdown across seasons:
    Matches per season, % with goal minute availability, and average goals per match.
    """
    sorted_years = sorted(
        stats.yearly_stats.keys(), key=lambda y: int(y) if y.isdigit() else 9999
    )
    years = [str(y) for y in sorted_years]
    tot_matches = [stats.yearly_stats[y].total_matches for y in sorted_years]
    min_matches = [
        stats.yearly_stats[y].minute_matches_count for y in sorted_years
    ]
    pct_min = [
        round(stats.yearly_stats[y].percentage_with_minutes, 1)
        for y in sorted_years
    ]
    avg_goals = [
        round(stats.yearly_stats[y].average_goals_per_minute_match, 2)
        for y in sorted_years
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            name="Всего матчей",
            x=years,
            y=tot_matches,
            marker_color=COLOR_NAVY,
            opacity=0.6,
            hovertemplate="Год: <b>%{x}</b><br>Всего матчей: <b>%{y:,}</b><extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            name="Матчей с минутами голов",
            x=years,
            y=min_matches,
            marker_color=COLOR_REGULAR,
            opacity=0.85,
            hovertemplate="Год: <b>%{x}</b><br>С минутами: <b>%{y:,}</b><extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="Доля с минутами (%)",
            x=years,
            y=pct_min,
            mode="lines+markers",
            line=dict(color=COLOR_ACCENT, width=3),
            hovertemplate="Год: <b>%{x}</b><br>Доля с минутами: <b>%{y:.1f}%</b><extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            name="Ср. голов за матч",
            x=years,
            y=avg_goals,
            mode="lines+markers",
            line=dict(color=COLOR_STOPPAGE, width=2, dash="dash"),
            hovertemplate="Год: <b>%{x}</b><br>Ср. голов: <b>%{y:.2f}</b><extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text="<b>Динамика по годам (2000–2026): объем данных и средняя результативность</b>",
            x=0.5,
            xanchor="center",
            y=1,
            yanchor="top",
            yref="container",
            pad=dict(t=6, b=8),
            font=dict(size=18, family="sans-serif", color="#212529"),
        ),
        barmode="group",
        plot_bgcolor=BG_COLOR,
        paper_bgcolor="#ffffff",
        height=540,
        margin=dict(l=64, r=64, b=72, t=118),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        xaxis=dict(
            title=dict(text="Сезон / Год", standoff=10),
            gridcolor=GRID_COLOR,
            type="category",
            tickangle=-90,
            automargin=True,
        ),
    )
    fig.update_yaxes(
        title_text="Количество матчей",
        secondary_y=False,
        gridcolor=GRID_COLOR,
        automargin=True,
    )
    fig.update_yaxes(
        title_text="Процент (%) / Ср. голов",
        secondary_y=True,
        automargin=True,
        range=[0, 105],
        gridcolor=GRID_COLOR,
    )
    return fig


DASHBOARD_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "displayModeBar": "hover",
}


def _trace_color(trace) -> str:
    """Pick a single swatch color from a bar or line trace."""
    mode = getattr(trace, "mode", None) or ""
    if getattr(trace, "type", None) == "scatter" or "lines" in mode:
        line = getattr(trace, "line", None)
        color = getattr(line, "color", None) if line is not None else None
        if isinstance(color, str) and color:
            return color
    marker = getattr(trace, "marker", None)
    color = getattr(marker, "color", None) if marker is not None else None
    if isinstance(color, (list, tuple)):
        color = next((item for item in color if isinstance(item, str)), None)
    if isinstance(color, str) and color:
        return color
    return "#64748b"


def _html_legend(fig: go.Figure) -> str:
    """HTML legend that wraps inside a narrow card. SVG legends do not wrap."""
    items: list[str] = []
    for trace in fig.data:
        name = getattr(trace, "name", None)
        if not name:
            continue
        kind = "line" if "lines" in (getattr(trace, "mode", None) or "") else "bar"
        items.append(
            f'<span class="legend-item {kind}">'
            f'<i style="background:{escape(_trace_color(trace))}"></i>'
            f"{escape(str(name))}</span>"
        )
    if not items:
        return ""
    return f'<div class="chart-legend">{"".join(items)}</div>'


def _dashboard_embed(fig: go.Figure) -> tuple[str, str]:
    """Lift the SVG title into HTML so it can wrap, and return the plot div.

    Plotly draws the title as one SVG line. In the two-column grid that line is
    wider than the card, and it also sits on the same pixels as the legend.
    """
    embedded = go.Figure(fig.to_dict())
    heading = embedded.layout.title.text or ""
    legend = _html_legend(embedded)
    # Room for the in-plot toggle, or just the modebar when there is none.
    top = 46 if embedded.layout.updatemenus else 28
    embedded.update_layout(showlegend=False, title_text="", margin_t=top)
    div = embedded.to_html(
        include_plotlyjs=False,
        full_html=False,
        config=DASHBOARD_CONFIG,
    )
    return f'<div class="chart-heading">{heading}</div>{legend}', div


def render_dashboard_html(
    stats: AggregatedStats,
    figures: dict[str, go.Figure],
    offline: bool = False,
    top_n: int = TOP_SCORES_COUNT,
) -> str:
    """Renders the comprehensive single-page HTML report dashboard."""
    gen_time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    pct_with_min = (
        round(stats.minute_matches_count / stats.total_matches * 100, 1)
        if stats.total_matches
        else 0
    )
    pct_score_only = round(100.0 - pct_with_min, 1)
    probs = stats.goal_probabilities
    p_late = (
        f"{stats.late_goal_probability:.2f}%"
        if stats.late_goal_probability is not None
        else "N/A"
    )

    # Plotly script inclusion
    if offline:
        import plotly.offline

        plotly_script_tag = f"<script>{plotly.offline.get_plotlyjs()}</script>"
    else:
        plotly_script_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'

    # Titles and legends are HTML so they wrap inside the cards. The plot
    # itself is a div without a second copy of plotly.js.
    head_first_half, div_first_half = _dashboard_embed(figures["first_half"])
    head_second_half, div_second_half = _dashboard_embed(figures["second_half"])
    head_diff, div_diff = _dashboard_embed(figures["diff"])
    head_fh_diff, div_fh_diff = _dashboard_embed(figures["fh_diff"])
    head_probs, div_probs = _dashboard_embed(figures["probs"])
    head_scores, div_scores = _dashboard_embed(figures["scores"])
    head_yearly, div_yearly = _dashboard_embed(figures["yearly"])

    html_template = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Футбольная статистика — интерактивный отчёт</title>
  {plotly_script_tag}
  <style>
    :root {{
      --primary: #1e3a8a;
      --primary-light: #3b82f6;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --dark: #0f172a;
      --gray-50: #f8fafc;
      --gray-100: #f1f5f9;
      --gray-200: #e2e8f0;
      --gray-600: #475569;
      --gray-900: #0f172a;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--gray-50);
      color: var(--gray-900);
      line-height: 1.5;
    }}
    .header {{
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      color: #ffffff;
      padding: 2.5rem 1.5rem;
      text-align: center;
      border-bottom: 4px solid var(--primary-light);
    }}
    .header h1 {{ font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }}
    .header p {{ color: #cbd5e1; font-size: 1.1rem; max-width: 800px; margin: 0 auto; }}
    .meta-badge {{
      display: inline-block;
      margin-top: 1rem;
      padding: 0.25rem 0.75rem;
      background-color: rgba(255, 255, 255, 0.1);
      border-radius: 9999px;
      font-size: 0.875rem;
      color: #94a3b8;
    }}
    .nav-bar {{
      position: sticky;
      top: 0;
      background-color: #ffffff;
      z-index: 100;
      box-shadow: 0 2px 4px rgba(0,0,0,0.06);
      padding: 0.75rem 1rem;
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .nav-bar a {{
      color: var(--gray-600);
      text-decoration: none;
      font-weight: 600;
      font-size: 0.9rem;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      transition: all 0.2s;
    }}
    .nav-bar a:hover {{ background-color: var(--gray-100); color: var(--primary); }}
    .container {{
      max-width: 1300px;
      margin: 2rem auto;
      padding: 0 1rem;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .kpi-card {{
      background: #ffffff;
      padding: 1.25rem;
      border-radius: 10px;
      border: 1px solid var(--gray-200);
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .kpi-title {{ font-size: 0.85rem; text-transform: uppercase; font-weight: 700; color: var(--gray-600); letter-spacing: 0.5px; }}
    .kpi-value {{ font-size: 1.8rem; font-weight: 800; color: var(--dark); margin: 0.4rem 0; }}
    .kpi-sub {{ font-size: 0.85rem; color: #64748b; }}
    .chart-card {{
      background: #ffffff;
      padding: 1.5rem;
      border-radius: 12px;
      border: 1px solid var(--gray-200);
      box-shadow: 0 2px 6px rgba(0,0,0,0.04);
      margin-bottom: 2rem;
    }}
    .callout {{
      background-color: #fefce8;
      border-left: 4px solid #eab308;
      padding: 1rem;
      border-radius: 4px;
      margin-top: 1rem;
      font-size: 0.92rem;
      color: #713f12;
      line-height: 1.5;
    }}
    .callout b {{ color: #854d0e; }}
    .chart-heading {{
      text-align: center;
      margin: 0 0 0.35rem;
      line-height: 1.35;
      overflow-wrap: break-word;
    }}
    .chart-heading b {{
      font-size: 1.05rem;
      font-weight: 800;
      color: #212529;
    }}
    .chart-legend {{
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 0.35rem 1rem;
      margin: 0.15rem 0 0.55rem;
      font-size: 0.85rem;
      color: #334155;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }}
    .legend-item i {{
      width: 0.85rem;
      height: 0.85rem;
      border-radius: 2px;
      display: inline-block;
      flex: none;
    }}
    .legend-item.line i {{
      width: 1.15rem;
      height: 3px;
      border-radius: 1px;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.5rem;
    }}
    .grid-2 > * {{ min-width: 0; }}
    .chart-card {{ min-width: 0; }}
    /* Plotly's modebar uses a high z-index and otherwise paints over the sticky nav. */
    .modebar-container {{ z-index: 1 !important; }}
    @media (min-width: 900px) {{
      .grid-2 {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 640px) {{
      .header {{ padding: 1.6rem 1rem; }}
      .header h1 {{ font-size: 1.45rem; }}
      .header p {{ font-size: 0.95rem; }}
      .kpi-value {{ font-size: 1.45rem; }}
    }}
    footer {{
      text-align: center;
      padding: 2rem;
      color: var(--gray-600);
      font-size: 0.875rem;
      border-top: 1px solid var(--gray-200);
      margin-top: 3rem;
    }}
  </style>
</head>
<body>

  <header class="header">
    <h1>Футбольная статистика & Аналитика</h1>
    <p>Комплексный анализ распределения голов, временных интервалов и паттернов результатов матчей</p>
    <div class="meta-badge">Сгенерировано: {gen_time} | Сезоны: 2000–2026</div>
  </header>

  <nav class="nav-bar">
    <a href="#summary">Сводка</a>
    <a href="#halves">Минуты голов</a>
    <a href="#differences">Интервалы голов</a>
    <a href="#probabilities">Вероятности</a>
    <a href="#scores">Популярные счета</a>
    <a href="#yearly">Динамика по годам</a>
  </nav>

  <main class="container">

    <!-- KPI Cards -->
    <section id="summary" class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">Всего матчей</div>
        <div class="kpi-value">{stats.total_matches:,}</div>
        <div class="kpi-sub">С минутами: {stats.minute_matches_count:,} ({pct_with_min}%)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Всего голов (с минутами)</div>
        <div class="kpi-value">{stats.total_goals_in_minute_matches:,}</div>
        <div class="kpi-sub">1-й: {stats.total_first_half_goals:,} | 2-й: {stats.total_second_half_goals:,} | Овертайм: {stats.extra_time_goals_count:,}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Вероятность 2-го гола</div>
        <div class="kpi-value" style="color: #2563eb;">{probs.get('Второй гол', 0.0):.2f}%</div>
        <div class="kpi-sub">3-й гол: {probs.get('Третий гол', 0.0):.2f}% | 4-й гол: {probs.get('Четвертый гол', 0.0):.2f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Гол после 70' (при 0:0 к 70')</div>
        <div class="kpi-value" style="color: #d97706;">{p_late}</div>
        <div class="kpi-sub">Матчей со счетом 0:0 на 70': {stats.matches_0_0_at_70:,}</div>
      </div>
    </section>

    <!-- Goal Minute Distributions -->
    <section id="halves">
      <div class="chart-card">
        {head_first_half}
        {div_first_half}
      </div>

      <div class="chart-card">
        {head_second_half}
        {div_second_half}
      </div>

      <div class="callout">
        <b>💡 Почему наблюдаются пики на 45-й и 90-й минутах?</b><br>
        1. <b>Агрегация в протоколах:</b> Во многих лигах и исторических данных компенсированное время не разделялось на «+1, +2» — все голы после 45:00 и 90:00 записывались строго на 45-ю или 90-ю минуту.<br>
        2. <b>Разметка компенсированного времени:</b> Столбцы <code>45+X</code> и <code>90+X</code> отображают матчи с подробным хронометражем. Плавный спад обусловлен тем, что 1–2 минуты добавляются регулярно, а 5+ минут — редко.<br>
        3. <b>Игровая динамика:</b> Под занавес таймов накапливается физическая усталость игроков обороны, а уступающая команда идет на риск, повышая общую результативность.
      </div>
    </section>

    <!-- Goal Time Differences -->
    <section id="differences" style="margin-top: 2rem;">
      <div class="grid-2">
        <div class="chart-card">
          {head_diff}
          {div_diff}
        </div>
        <div class="chart-card">
          {head_fh_diff}
          {div_fh_diff}
        </div>
      </div>
    </section>

    <!-- Probabilities & Top Scores -->
    <section id="probabilities" style="margin-top: 2rem;">
      <div class="grid-2">
        <div class="chart-card">
          {head_probs}
          {div_probs}
        </div>
        <div id="scores" class="chart-card">
          {head_scores}
          {div_scores}
        </div>
      </div>
    </section>

    <!-- Yearly Stats -->
    <section id="yearly" style="margin-top: 2rem;">
      <div class="chart-card">
        {head_yearly}
        {div_yearly}
      </div>
    </section>

  </main>

  <footer>
    Футбольная статистика &copy; {datetime.datetime.now().year}. Построено на Plotly & Python. Все права защищены.
  </footer>

</body>
</html>
"""
    return html_template


def export_separate_charts(
    stats: AggregatedStats,
    output_folder: str,
    offline: bool = False,
    top_n: int = TOP_SCORES_COUNT,
) -> None:
    """Exports each chart as an individual standalone HTML file in output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    mode = True if offline else "cdn"

    build_first_half_chart(stats).write_html(
        os.path.join(output_folder, "first_half_goals_minutes.html"),
        include_plotlyjs=mode,
    )
    build_second_half_chart(stats).write_html(
        os.path.join(output_folder, "second_half_goals_minutes.html"),
        include_plotlyjs=mode,
    )
    build_goal_diff_chart(stats).write_html(
        os.path.join(output_folder, "goal_difference.html"),
        include_plotlyjs=mode,
    )
    build_first_half_goal_diff_chart(stats).write_html(
        os.path.join(output_folder, "first_half_goal_difference.html"),
        include_plotlyjs=mode,
    )
    build_probabilities_chart(stats).write_html(
        os.path.join(output_folder, "goal_probabilities.html"),
        include_plotlyjs=mode,
    )
    build_top_scores_chart(stats, top_n=top_n).write_html(
        os.path.join(output_folder, "top_scores.html"), include_plotlyjs=mode
    )
    build_yearly_chart(stats).write_html(
        os.path.join(output_folder, "yearly_stats.html"), include_plotlyjs=mode
    )


def generate_all_reports(
    json_folder: str = DEFAULT_JSON_FOLDER,
    plots_folder: str = DEFAULT_PLOTS_FOLDER,
    offline: bool = False,
    separate_files: bool = True,
    top_n: int = TOP_SCORES_COUNT,
) -> AggregatedStats:
    """Aggregates match data and generates the HTML dashboard plus optional separate charts."""
    os.makedirs(plots_folder, exist_ok=True)

    stats = aggregate_dataset(json_folder)
    print(format_console_summary(stats, top_n=top_n))

    # Build figures
    figures = {
        "first_half": build_first_half_chart(stats),
        "second_half": build_second_half_chart(stats),
        "diff": build_goal_diff_chart(stats),
        "fh_diff": build_first_half_goal_diff_chart(stats),
        "probs": build_probabilities_chart(stats),
        "scores": build_top_scores_chart(stats, top_n=top_n),
        "yearly": build_yearly_chart(stats),
    }

    # Generate single master dashboard
    report_html = render_dashboard_html(
        stats, figures, offline=offline, top_n=top_n
    )
    report_path = os.path.join(plots_folder, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    logger.info(f"Сгенерирован единый отчёт-дашборд: {report_path}")

    # Generate standalone files if requested
    if separate_files:
        export_separate_charts(
            stats, plots_folder, offline=offline, top_n=top_n
        )
        logger.info(
            f"Экспортированы отдельные интерактивные графики в папку: {plots_folder}"
        )

    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Генератор интерактивной футбольной статистики и HTML-дашборда."
    )
    parser.add_argument(
        "--json-folder",
        default=DEFAULT_JSON_FOLDER,
        help="Путь к папке с JSON файлами матчей (по умолчанию: JSON)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_PLOTS_FOLDER,
        help="Путь к выходной папке для HTML отчетов (по умолчанию: plots)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Встроить plotly.js внутрь файлов для автономного просмотра без интернета",
    )
    parser.add_argument(
        "--separate-files",
        action="store_true",
        default=True,
        help="Также генерировать отдельные HTML файлы для каждого графика (по умолчанию: True)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_SCORES_COUNT,
        help="Количество позиций в рейтинге частых счетов (по умолчанию: 7)",
    )

    args = parser.parse_args()
    generate_all_reports(
        json_folder=args.json_folder,
        plots_folder=args.out,
        offline=args.offline,
        separate_files=args.separate_files,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()