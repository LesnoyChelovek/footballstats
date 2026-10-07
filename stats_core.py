"""
Core business logic, data models, parsing, validation, and aggregations for football match statistics.
Shared by both CLI static report generator (main.py) and interactive HTML report generator (stats_html.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

# --- Configuration Constants ---
FIRST_HALF_MAX_MINUTE = 45
SECOND_HALF_MAX_MINUTE = 90
MINUTE_THRESHOLD_FOR_LATE_GOAL = 70
DEFAULT_JSON_FOLDER = "JSON"
DEFAULT_PLOTS_FOLDER = "plots"
TOP_SCORES_COUNT = 7

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoalTime:
    """Represents the parsed time of a goal in a match."""

    period: str  # "Первый тайм", "Второй тайм", "Экстра-тайм", "Не определено"
    period_order: int  # 1: 1st half, 2: 2nd half, 3: extra time 1st, 4: extra time 2nd, 99: undefined
    main_minute: int  # e.g., 45 in "45+2", 15 in "15", 90 in "90+3", 105 in "105+1"
    added_minute: int  # e.g., 2 in "45+2", 0 in "15"
    raw_str: str

    @property
    def is_stoppage(self) -> bool:
        """True if the goal occurred in stoppage/added time."""
        return self.added_minute > 0

    @property
    def label(self) -> str:
        """Formatted string representation, e.g. '45+2', '15', '90+3'."""
        if self.is_stoppage:
            return f"{self.main_minute}+{self.added_minute}"
        return str(self.main_minute)

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Chronological sort key guaranteeing correct match sequence."""
        return (self.period_order, self.main_minute, self.added_minute)


def normalize_score(score_str: Any) -> tuple[str, bool]:
    """
    Normalizes a match score string.
    Trims whitespace and removes annotation characters like '*' (e.g., '0 - 3*' -> '0 - 3').
    Standardizes format to 'H - A'.
    Returns (normalized_score, was_modified).
    """
    if not isinstance(score_str, str):
        score_str = str(score_str)
    cleaned = score_str.strip()
    was_modified = False

    if "*" in cleaned:
        cleaned = cleaned.replace("*", "").strip()
        was_modified = True

    m = re.match(r"^(\d+)\s*-\s*(\d+)$", cleaned)
    if m:
        std = f"{m.group(1)} - {m.group(2)}"
        if std != score_str:
            was_modified = True
        return std, was_modified

    return cleaned, was_modified


def _as_minute_list(value: Any) -> list[str]:
    """Normalizes a minutes field to a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _is_na_minute(minute_string: str) -> bool:
    """Checks if a minute string denotes missing/unavailable time ('NA', 'N/A')."""
    return minute_string.strip().upper() in {"NA", "N/A"}


def _minutes_unavailable(home_goals: list[str], away_goals: list[str]) -> bool:
    """Returns True if any goal minute is missing/NA."""
    return any(_is_na_minute(minute) for minute in home_goals + away_goals)


def parse_minute_string(minute_string: str) -> Optional[GoalTime]:
    """
    Parses a goal minute string into a GoalTime object.
    Supports formats:
      - '15', '45', '78'' -> regular match minute
      - '45+2', '90+3'' -> stoppage time
      - '93', '105', '118' -> extra time in cup/playoff matches (>=91 without '+')
      - '105+1', '120+2' -> extra time stoppage
    Returns None if missing/NA or cannot be parsed.
    """
    cleaned = str(minute_string).rstrip("'’").strip()
    if _is_na_minute(cleaned):
        return None

    try:
        if "+" in cleaned:
            parts = cleaned.split("+")
            if len(parts) != 2:
                raise ValueError(f"Too many plus signs in {cleaned}")
            main_m = int(parts[0])
            add_m = int(parts[1])

            if main_m <= FIRST_HALF_MAX_MINUTE:
                period = "Первый тайм"
                period_order = 1
            elif main_m <= SECOND_HALF_MAX_MINUTE:
                period = "Второй тайм"
                period_order = 2
            elif main_m <= 105:
                period = "Экстра-тайм"
                period_order = 3
            else:
                period = "Экстра-тайм"
                period_order = 4

            return GoalTime(
                period=period,
                period_order=period_order,
                main_minute=main_m,
                added_minute=add_m,
                raw_str=cleaned,
            )
        else:
            minute_int = int(cleaned)
            if 1 <= minute_int <= FIRST_HALF_MAX_MINUTE:
                period = "Первый тайм"
                period_order = 1
            elif 46 <= minute_int <= SECOND_HALF_MAX_MINUTE:
                period = "Второй тайм"
                period_order = 2
            elif minute_int >= 91:
                # Documented rule: minutes >= 91 without '+' are genuine extra-time in knockout matches.
                period = "Экстра-тайм"
                period_order = 3 if minute_int <= 105 else 4
            else:
                # 0 or negative
                logger.warning(
                    f"Недопустимая минута гола: '{minute_string}'. Пропускается."
                )
                return None

            return GoalTime(
                period=period,
                period_order=period_order,
                main_minute=minute_int,
                added_minute=0,
                raw_str=cleaned,
            )

    except ValueError:
        if not _is_na_minute(cleaned):
            logger.warning(
                f"Не удалось разобрать строку минуты: '{minute_string}'. Возвращается None"
            )
        return None


def calculate_goal_difference(g1: GoalTime, g2: GoalTime) -> int:
    """
    Calculates the playing-time difference in minutes between first goal (g1) and second goal (g2).
    Precondition: g1 and g2 are sorted chronologically (g1.sort_key <= g2.sort_key).
    Returns an integer >= 0.
    """
    if g1.period_order == g2.period_order:
        if not g1.is_stoppage and not g2.is_stoppage:
            return max(0, g2.main_minute - g1.main_minute)
        if not g1.is_stoppage and g2.is_stoppage:
            base_minute = 45 if g1.period_order == 1 else 90
            return max(0, (base_minute + g2.added_minute) - g1.main_minute)
        if g1.is_stoppage and g2.is_stoppage:
            return max(0, g2.added_minute - g1.added_minute)
        # Stoppage followed by regular in same half shouldn't occur chronologically
        return 0

    if g1.period_order == 1 and g2.period_order == 2:
        # Playing time elapsed in 2nd half before g2:
        sec_half_elapsed = max(0, g2.main_minute - 45)
        # Playing time in 1st half remaining after g1:
        first_half_remaining = max(0, 45 - min(g1.main_minute, 45))
        return sec_half_elapsed + first_half_remaining

    if g1.period_order <= 2 and g2.period_order >= 3:
        # 1st/2nd half to extra time: g2 happened in extra time
        extra_elapsed = max(0, g2.main_minute - 90)
        reg_remaining = max(0, 90 - min(g2.main_minute, 90))
        return extra_elapsed + reg_remaining

    return max(0, g2.main_minute - g1.main_minute)


def extract_year_from_filename(filename: str) -> str:
    """Extracts 4-digit year from filename like 'match_data2024.json', or returns 'unknown'."""
    m = re.search(r"(\d{4})", os.path.basename(filename))
    if m:
        return m.group(1)
    logger.warning(
        f"Имя файла '{filename}' не содержит 4-значный год. Присвоена группа 'unknown'."
    )
    return "unknown"


@dataclass
class YearlyStats:
    """Aggregated statistics for a single season/year."""

    year: str
    total_matches: int = 0
    minute_matches_count: int = 0
    total_goals: int = 0
    score_counts: Counter[str] = field(default_factory=Counter)

    @property
    def percentage_with_minutes(self) -> float:
        if self.total_matches == 0:
            return 0.0
        return (self.minute_matches_count / self.total_matches) * 100.0

    @property
    def average_goals_per_minute_match(self) -> float:
        if self.minute_matches_count == 0:
            return 0.0
        return self.total_goals / self.minute_matches_count


@dataclass
class AggregatedStats:
    """Container for all aggregated match and goal metrics across the dataset."""

    total_matches: int = 0
    minute_matches_count: int = 0
    score_only_matches_count: int = 0
    normalized_scores_count: int = 0

    score_counts: Counter[str] = field(default_factory=Counter)

    # First Half goal counts:
    # regular minutes: 1..45 -> count
    first_half_regular: Counter[int] = field(default_factory=Counter)
    # stoppage minutes: added minute X (for 45+X) -> count
    first_half_stoppage: Counter[int] = field(default_factory=Counter)

    # Second Half goal counts:
    # regular minutes: 46..90 -> count
    second_half_regular: Counter[int] = field(default_factory=Counter)
    # stoppage minutes: added minute X (for 90+X) -> count
    second_half_stoppage: Counter[int] = field(default_factory=Counter)

    # Extra time goals (main_minute >= 91 or extra time stoppage):
    extra_time_goals_count: int = 0
    extra_time_breakdown: Counter[str] = field(default_factory=Counter)

    # Goal differences between 1st and 2nd goals
    goal_differences: list[int] = field(default_factory=list)
    first_half_goal_differences: list[int] = field(default_factory=list)

    # Subsequent goal probabilities
    matches_with_first_goal: int = 0
    matches_with_second_goal: int = 0
    matches_with_third_goal: int = 0
    matches_with_fourth_goal: int = 0

    # Late goal (0-0 at 70)
    matches_0_0_at_70: int = 0
    matches_0_0_at_70_goal_after_70: int = 0

    # Breakdown by year
    yearly_stats: dict[str, YearlyStats] = field(default_factory=dict)

    @property
    def total_first_half_goals(self) -> int:
        return sum(self.first_half_regular.values()) + sum(
            self.first_half_stoppage.values()
        )

    @property
    def total_second_half_goals(self) -> int:
        return sum(self.second_half_regular.values()) + sum(
            self.second_half_stoppage.values()
        )

    @property
    def total_goals_in_minute_matches(self) -> int:
        return (
            self.total_first_half_goals
            + self.total_second_half_goals
            + self.extra_time_goals_count
        )

    @property
    def goal_probabilities(self) -> dict[str, float]:
        """Probabilities of scoring 2nd, 3rd, 4th goal given the 1st goal."""
        res: dict[str, float] = {}
        if self.matches_with_first_goal > 0:
            res["Второй гол"] = (
                self.matches_with_second_goal / self.matches_with_first_goal
            ) * 100.0
            res["Третий гол"] = (
                self.matches_with_third_goal / self.matches_with_first_goal
            ) * 100.0
            res["Четвертый гол"] = (
                self.matches_with_fourth_goal / self.matches_with_first_goal
            ) * 100.0
        return res

    @property
    def late_goal_probability(self) -> Optional[float]:
        """Probability of a goal after 70th min given 0-0 at 70th min."""
        if self.matches_0_0_at_70 > 0:
            return (
                self.matches_0_0_at_70_goal_after_70 / self.matches_0_0_at_70
            ) * 100.0
        return None

    def top_scores(self, top_n: int = TOP_SCORES_COUNT) -> list[tuple[str, int]]:
        """Returns the top N most frequent match scores (most frequent first)."""
        return self.score_counts.most_common(top_n)


def process_match_dict(
    match_data: dict[str, Any], stats: AggregatedStats, year: str = "unknown"
) -> bool:
    """
    Processes a single match dictionary and updates AggregatedStats.
    Returns True if match had valid goal minutes, False if score-only (NA minutes).
    """
    home_goals_raw = _as_minute_list(match_data.get("home_goals_minutes", []))
    away_goals_raw = _as_minute_list(match_data.get("away_goals_minutes", []))
    raw_score = match_data.get("score", "N/A")

    norm_score, was_norm = normalize_score(raw_score)
    if was_norm:
        stats.normalized_scores_count += 1

    stats.score_counts[norm_score] += 1
    stats.total_matches += 1

    # Yearly aggregation tracking
    if year not in stats.yearly_stats:
        stats.yearly_stats[year] = YearlyStats(year=year)
    y_stat = stats.yearly_stats[year]
    y_stat.total_matches += 1
    y_stat.score_counts[norm_score] += 1

    if _minutes_unavailable(home_goals_raw, away_goals_raw):
        stats.score_only_matches_count += 1
        return False

    # Parse all goals in match
    all_goals: list[GoalTime] = []
    for g_str in home_goals_raw + away_goals_raw:
        gt = parse_minute_string(g_str)
        if gt is not None:
            all_goals.append(gt)

    stats.minute_matches_count += 1
    y_stat.minute_matches_count += 1
    y_stat.total_goals += len(all_goals)

    # Sort goals chronologically
    all_goals.sort(key=lambda g: g.sort_key)

    has_goal_before_70 = False
    has_goal_after_70 = False

    for gt in all_goals:
        if gt.period == "Первый тайм":
            if gt.is_stoppage:
                stats.first_half_stoppage[gt.added_minute] += 1
            else:
                stats.first_half_regular[gt.main_minute] += 1
        elif gt.period == "Второй тайм":
            if gt.is_stoppage:
                stats.second_half_stoppage[gt.added_minute] += 1
            else:
                stats.second_half_regular[gt.main_minute] += 1
        elif gt.period == "Экстра-тайм":
            stats.extra_time_goals_count += 1
            stats.extra_time_breakdown[gt.label] += 1

        # Check late goal conditions (70th minute threshold)
        # Note: 1st half goals are always <= 45 <= 70.
        # 2nd half goals with main_minute <= 70 are before 70.
        if gt.period_order == 1:
            has_goal_before_70 = True
        elif gt.period_order == 2:
            if gt.main_minute <= MINUTE_THRESHOLD_FOR_LATE_GOAL:
                has_goal_before_70 = True
            else:
                has_goal_after_70 = True
        elif gt.period_order >= 3:
            # Extra time is after 90 > 70
            has_goal_after_70 = True

    # Subsequent goal progression
    num_goals = len(all_goals)
    if num_goals > 0:
        stats.matches_with_first_goal += 1
        if num_goals > 1:
            stats.matches_with_second_goal += 1
            if num_goals > 2:
                stats.matches_with_third_goal += 1
                if num_goals > 3:
                    stats.matches_with_fourth_goal += 1

    # Goal differences between 1st and 2nd goal
    if num_goals >= 2:
        g1, g2 = all_goals[0], all_goals[1]
        diff = calculate_goal_difference(g1, g2)
        stats.goal_differences.append(diff)
        if g1.period == "Первый тайм" and g2.period == "Первый тайм":
            diff_fh = calculate_goal_difference(g1, g2)
            stats.first_half_goal_differences.append(diff_fh)

    # 0-0 at 70 minutes check
    if not has_goal_before_70:
        stats.matches_0_0_at_70 += 1
        if has_goal_after_70:
            stats.matches_0_0_at_70_goal_after_70 += 1

    return True


def load_json_file(filepath: str) -> list[dict[str, Any]]:
    """Loads and normalizes JSON content into a flat list of match dicts."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)
        if isinstance(content, list):
            if content and isinstance(content[0], list):
                return [m for sublist in content for m in sublist if isinstance(m, dict)]
            return [m for m in content if isinstance(m, dict)]
        elif isinstance(content, dict):
            if "matches" in content and isinstance(content["matches"], list):
                return [m for m in content["matches"] if isinstance(m, dict)]
            return [content]
        return []
    except Exception as e:
        logger.exception(f"Ошибка загрузки файла {filepath}: {e}")
        return []


def aggregate_dataset(
    json_folder: str = DEFAULT_JSON_FOLDER,
) -> AggregatedStats:
    """Scans all JSON files in json_folder and compiles the full AggregatedStats."""
    stats = AggregatedStats()
    if not os.path.exists(json_folder):
        logger.warning(f"Папка не найдена: {json_folder}")
        return stats

    json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])
    if not json_files:
        logger.warning(f"В папке не найдено JSON файлов: {json_folder}")
        return stats

    for filename in json_files:
        filepath = os.path.join(json_folder, filename)
        year = extract_year_from_filename(filename)
        matches = load_json_file(filepath)
        for m in matches:
            process_match_dict(m, stats, year=year)

    logger.info(
        f"Обработано матчей для статистики: {stats.total_matches} "
        f"(с минутами голов: {stats.minute_matches_count}, только счет: {stats.score_only_matches_count}, "
        f"нормализовано счетов: {stats.normalized_scores_count})"
    )
    logger.info(
        f"Всего голов в матчах с минутами: {stats.total_goals_in_minute_matches} "
        f"(1-й тайм: {stats.total_first_half_goals}, 2-й тайм: {stats.total_second_half_goals}, "
        f"экстра-тайм: {stats.extra_time_goals_count}). Расхождение: 0."
    )

    return stats


def format_console_summary(
    stats: AggregatedStats, top_n: int = TOP_SCORES_COUNT
) -> str:
    """Formats standard text output matching original console summaries."""
    lines: list[str] = []

    probs = stats.goal_probabilities
    if probs:
        lines.append("\nВероятность, что после первого гола будет забит:")
        for order, prob in probs.items():
            lines.append(f"- {order}: {prob:.2f}%")
    else:
        lines.append(
            "\nНет матчей с первым голом для расчета вероятности последующих голов."
        )

    top_scores = stats.top_scores(top_n=top_n)
    lines.append(
        f"\nТоп-{len(top_scores)} самых часто встречающихся счетов матчей:"
    )
    for score, count in top_scores:
        lines.append(f"- Счет '{score}': {count} матчей")

    late_prob = stats.late_goal_probability
    if late_prob is not None:
        lines.append(
            f"\nВероятность гола после {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуты в матчах, "
            f"где счет был 0-0 к {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуте: {late_prob:.2f}%"
        )
    else:
        lines.append(
            f"\nНет матчей со счетом 0-0 к {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуте "
            f"для расчета вероятности гола после {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуты."
        )

    return "\n".join(lines)
