import json
import os
import matplotlib.pyplot as plt
from collections import Counter
import logging
import matplotlib

# --- Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
PLOT_STYLE = 'ggplot'
FONT_SIZE = 12
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
XTICK_FONT_SIZE = 10
YTICK_FONT_SIZE = 10
COLOR_PALETTE = ['#5cb85c', '#5bc0de', '#d9534f', '#f0ad4e', '#428bca', '#9467bd', '#e377c2']
DEFAULT_JSON_FOLDER = 'JSON'
DEFAULT_PLOTS_FOLDER = 'plots'
TOP_SCORES_COUNT = 7
TOP_GOAL_MINUTES_COUNT = 5
MINUTE_THRESHOLD_FOR_LATE_GOAL = 70
GOAL_BINS_FIRST_HALF = range(0, 70)
GOAL_BINS_SECOND_HALF = range(46, 130)
GOAL_DIFF_BINS = range(0, 91, 5)
FIRST_HALF_GOAL_DIFF_BINS = range(0, 46, 2)
HISTOGRAM_X_TICKS_FIRST_HALF = range(0, 71, 5)
HISTOGRAM_X_TICKS_SECOND_HALF = range(45, 131, 5)
HISTOGRAM_X_TICKS_GOAL_DIFF = range(0, 91, 10)
HISTOGRAM_X_TICKS_FIRST_HALF_GOAL_DIFF = range(0, 46, 5)
FIRST_HALF_MAX_MINUTE = 45


# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# --- Matplotlib Style Setup ---
plt.style.use(PLOT_STYLE)
matplotlib.rcParams['font.size'] = FONT_SIZE
matplotlib.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
matplotlib.rcParams['axes.labelsize'] = LABEL_FONT_SIZE
matplotlib.rcParams['xtick.labelsize'] = XTICK_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = YTICK_FONT_SIZE


def parse_minute_string(minute_string: str) -> tuple[str, int]:
    """Parses a minute string to extract time period and minute."""
    minute_string = minute_string.rstrip("'’")
    try:
        if '+' in minute_string:
            main_minute, added_time = minute_string.split('+')
            minute = int(main_minute) + int(added_time)
            time_period = "Первый тайм" if int(main_minute) <= FIRST_HALF_MAX_MINUTE else "Второй тайм" if int(main_minute) <= 90 else "Экстра-тайм"
        else:
            minute = int(minute_string)
            time_period = "Первый тайм" if 1 <= minute <= FIRST_HALF_MAX_MINUTE else "Второй тайм" if 46 <= minute <= 90 else "Экстра-тайм" if minute >= 91 else "Не определено"
        return time_period, minute
    except ValueError:
        logging.warning(f"Could not parse minute string: '{minute_string}'. Returning 'Не определено', 0")
        return "Не определено", 0


def _extract_goal_data(match_data: dict) -> tuple[list[str], list[str], str]:
    """Extracts goal minutes and score from match data."""
    home_goals = match_data.get('home_goals_minutes', [])
    away_goals = match_data.get('away_goals_minutes', [])
    score = match_data.get('score', 'N/A')
    return home_goals, away_goals, score


def _update_goal_stats(home_goals: list[str], away_goals: list[str], first_half_minutes: list[int],
                       second_half_minutes: list[int], all_goal_minutes: Counter,
                       goals_with_time_period: list[tuple[int, str]], matches_first_goal_after_70: list[int],
                       goal_before_70: list[bool], goal_after_70: list[bool]) -> None:
    """Updates statistics related to goal times and periods."""
    for goal_minute_str in home_goals + away_goals:
        time_period, minute = parse_minute_string(goal_minute_str)
        if time_period != "Не определено":
            all_goal_minutes[minute] += 1
            goals_with_time_period.append((minute, time_period))
            if time_period == "Первый тайм":
                first_half_minutes.append(minute)
            elif time_period == "Второй тайм":
                second_half_minutes.append(minute)
            if minute <= MINUTE_THRESHOLD_FOR_LATE_GOAL:
                goal_before_70[0] = True
            if minute > MINUTE_THRESHOLD_FOR_LATE_GOAL:
                goal_after_70[0] = True


def _update_match_goal_counts(num_goals_in_match: int, matches_with_first_goal: list[int],
                              matches_with_second_goal: list[int], matches_with_third_goal: list[int],
                              matches_with_fourth_goal: list[int]) -> None:
    """Updates counters for matches with first, second, third, and fourth goals."""
    if num_goals_in_match > 0:
        matches_with_first_goal[0] += 1
        if num_goals_in_match > 1:
            matches_with_second_goal[0] += 1
            if num_goals_in_match > 2:
                matches_with_third_goal[0] += 1
                if num_goals_in_match > 3:
                    matches_with_fourth_goal[0] += 1


def _update_goal_differences(goals_with_time_period: list[tuple[int, str]], goal_differences: list[int],
                             first_half_goal_differences: list[int]) -> None:
    """Updates lists of goal differences, including first half specific differences."""
    if len(goals_with_time_period) >= 2:
        sorted_goals_minutes = sorted(goals_with_time_period, key=lambda x: x[0])
        first_goal_data = sorted_goals_minutes[0]
        second_goal_data = sorted_goals_minutes[1]
        goal_differences.append(second_goal_data[0] - first_goal_data[0])
        if first_goal_data[1] == "Первый тайм" and second_goal_data[1] == "Первый тайм":
            first_half_goal_differences.append(second_goal_data[0] - first_goal_data[0])


def _update_zero_zero_stats(goal_before_70: list[bool], goal_after_70: list[bool],
                            matches_0_0_at_70: list[int], matches_0_0_at_70_goal_after_70: list[int]) -> None:
    """Updates counters for matches that are 0-0 at 70 minutes and if they get a goal after."""
    if not goal_before_70[0]:
        matches_0_0_at_70[0] += 1
        if goal_after_70[0]:
            matches_0_0_at_70_goal_after_70[0] += 1


def process_match(match_data: dict, first_half_minutes: list[int], second_half_minutes: list[int],
                  goal_differences: list[int], matches_with_first_goal: list[int],
                  matches_with_second_goal: list[int], matches_with_third_goal: list[int],
                  matches_with_fourth_goal: list[int], all_goal_minutes: Counter,
                  score_counts: Counter, first_half_goal_differences: list[int],
                  matches_first_goal_after_70: list[int], matches_0_0_at_70: list[int],
                  matches_0_0_at_70_goal_after_70: list[int]) -> None:
    """Processes data for a single match and updates statistics."""

    home_goals, away_goals, score = _extract_goal_data(match_data)
    score_counts[score] += 1

    all_goals_minutes_match = []
    goals_with_time_period = []
    goal_before_70 = [False] # Use list to be mutable in helper functions
    goal_after_70 = [False]   # Use list to be mutable in helper functions

    _update_goal_stats(home_goals, away_goals, first_half_minutes, second_half_minutes,
                       all_goal_minutes, goals_with_time_period, matches_first_goal_after_70,
                       goal_before_70, goal_after_70)

    num_goals_in_match = len(all_goals_minutes_match) # Not used, should use len(goals_with_time_period) or recalculate from home_goals/away_goals if needed. Actually, `goals_with_time_period` is used for goal counts and differences, so using its length is correct.
    num_goals_in_match = len(goals_with_time_period)

    _update_match_goal_counts(num_goals_in_match, matches_with_first_goal, matches_with_second_goal,
                               matches_with_third_goal, matches_with_fourth_goal)

    if num_goals_in_match >= 2: # Only calculate goal differences if at least 2 goals
        _update_goal_differences(goals_with_time_period, goal_differences, first_half_goal_differences)

    _update_zero_zero_stats(goal_before_70, goal_after_70, matches_0_0_at_70, matches_0_0_at_70_goal_after_70)


def calculate_probability_stats(matches_with_first_goal: list[int], matches_with_second_goal: list[int],
                                 matches_with_third_goal: list[int], matches_with_fourth_goal: list[int]) -> dict[str, float]:
    """Calculates probabilities of subsequent goals after the first goal."""
    probability_data = {}
    first_goal_matches = matches_with_first_goal[0]
    if first_goal_matches > 0:
        probability_data["Второй гол"] = (matches_with_second_goal[0] / first_goal_matches) * 100
        probability_data["Третий гол"] = (matches_with_third_goal[0] / first_goal_matches) * 100
        probability_data["Четвертый гол"] = (matches_with_fourth_goal[0] / first_goal_matches) * 100
    return probability_data


def get_top_scores(score_counts: Counter, top_n: int = TOP_SCORES_COUNT) -> list[tuple[str, int]]:
    """Returns the top N most frequent match scores."""
    return score_counts.most_common(top_n)


def plot_histogram(data: list[int], bins: range, title: str, xlabel: str, ylabel: str,
                   plot_filename: str, processed_matches_count: int, x_ticks: range = None, x_range: tuple[int, int] = None) -> None:
    """Generates and saves a histogram plot."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color=COLOR_PALETTE[0], edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if x_ticks:
        plt.xticks(x_ticks)
    if x_range:
        plt.xlim(x_range)
    plt.grid(axis='y', linestyle='--')
    _annotate_processed_matches(processed_matches_count)
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_bar_chart(data: dict[str, float], title: str, xlabel: str, ylabel: str,
                   plot_filename: str, processed_matches_count: int, y_range: tuple[int, int] = None) -> None:
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
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    _annotate_processed_matches(processed_matches_count)
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_top_scores_chart(top_scores: list[tuple[str, int]], plot_folder: str, processed_matches_count: int) -> None:
    """Generates and saves a horizontal bar chart for top match scores."""
    scores = [score for score, count in top_scores]
    counts = [count for score, count in top_scores]

    plt.figure(figsize=(10, 7))
    plt.barh(scores[::-1], counts[::-1], color=COLOR_PALETTE[4], alpha=0.7)
    plt.title('Топ-7 самых часто встречающихся счетов матчей')
    plt.xlabel('Количество матчей')
    plt.ylabel('Счет матча')
    for index, value in enumerate(counts[::-1]):
        plt.text(value + 10, index, str(value), va='center', fontsize=10)

    _annotate_processed_matches(processed_matches_count)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_filename = os.path.join(plot_folder, 'top_scores.png')
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()


def _annotate_processed_matches(processed_matches_count):
    """Adds annotation about the number of processed matches to the plot."""
    plt.text(0.95, 0.95, f'Обработано матчей: {processed_matches_count}',
             transform=plt.gca().transAxes, ha='right', va='top', fontsize=9, color='gray')


def _load_json_file(filepath: str) -> list[dict]:
    """Loads and parses a JSON file, handling different structures."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
            if isinstance(file_content, list):
                if file_content and isinstance(file_content[0], list): # Check for list of lists
                    return [match for inner_list in file_content for match in inner_list] # Flatten list of lists
                return file_content
            elif isinstance(file_content, dict) and 'matches' in file_content and isinstance(file_content['matches'], list):
                return file_content['matches']
            elif isinstance(file_content, dict): # Assume it's a single match dict
                return [file_content]
            else:
                logging.warning(f"Unexpected JSON format in {filepath}. Skipping file content.")
                return [] # Return empty list to avoid further processing errors
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON in {filepath}. Check file format.")
        return []
    except Exception as e:
        logging.exception(f"Error processing file {filepath}")
        return []


def analyze_football_stats(json_folder: str = DEFAULT_JSON_FOLDER, plots_folder: str = DEFAULT_PLOTS_FOLDER) -> None:
    """
    Analyzes football statistics from JSON files, generates plots, and prints results.
    """

    first_half_minutes = []
    second_half_minutes = []
    goal_differences = []
    matches_with_first_goal = [0]
    matches_with_second_goal = [0]
    matches_with_third_goal = [0]
    matches_with_fourth_goal = [0]
    all_goal_minutes = Counter()
    score_counts = Counter()
    processed_matches_count = 0
    first_half_goal_differences = []
    matches_first_goal_after_70 = [0]
    matches_0_0_at_70 = [0]
    matches_0_0_at_70_goal_after_70 = [0]

    os.makedirs(plots_folder, exist_ok=True) # Create plots folder if it doesn't exist

    json_files = [filename for filename in os.listdir(json_folder) if filename.endswith(".json")]
    if not json_files:
        logging.warning(f"No JSON files found in folder: {json_folder}")
        return

    for filename in json_files:
        filepath = os.path.join(json_folder, filename)
        match_list = _load_json_file(filepath)
        if not match_list: # Skip to next file if loading failed or content was unexpected
            continue

        for match_data in match_list:
            if not isinstance(match_data, dict):
                logging.warning(f"Expected match dictionary, but got {type(match_data)} in {filename}. Skipping element.")
                continue
            process_match(match_data, first_half_minutes, second_half_minutes, goal_differences,
                          matches_with_first_goal, matches_with_second_goal, matches_with_third_goal,
                          matches_with_fourth_goal, all_goal_minutes, score_counts,
                          first_half_goal_differences, matches_first_goal_after_70,
                          matches_0_0_at_70, matches_0_0_at_70_goal_after_70)
            processed_matches_count += 1

    logging.info(f"Processed matches for statistics: {processed_matches_count}")

    # --- Plotting ---
    plot_histogram(first_half_minutes, bins=GOAL_BINS_FIRST_HALF,
                 title='Распределение минут голов в первом тайме',
                 xlabel='Минута', ylabel='Количество голов',
                 plot_filename=os.path.join(plots_folder, 'first_half_goals_minutes.png'),
                 processed_matches_count=processed_matches_count,
                 x_ticks=HISTOGRAM_X_TICKS_FIRST_HALF)

    plot_histogram(second_half_minutes, bins=GOAL_BINS_SECOND_HALF,
                 title='Распределение минут голов во втором тайме',
                 xlabel='Минута', ylabel='Количество голов',
                 plot_filename=os.path.join(plots_folder, 'second_half_goals_minutes.png'),
                 processed_matches_count=processed_matches_count,
                 x_ticks=HISTOGRAM_X_TICKS_SECOND_HALF)

    plot_histogram(goal_differences, bins=GOAL_DIFF_BINS,
                 title='Распределение разницы в минутах между первым и вторым голом',
                 xlabel='Разница в минутах', ylabel='Количество матчей',
                 plot_filename=os.path.join(plots_folder, 'goal_difference.png'),
                 processed_matches_count=processed_matches_count,
                 x_ticks=HISTOGRAM_X_TICKS_GOAL_DIFF)

    plot_histogram(first_half_goal_differences, bins=FIRST_HALF_GOAL_DIFF_BINS,
                 title='Распределение разницы в минутах между первым и вторым голом (Первый тайм)',
                 xlabel='Разница в минутах', ylabel='Количество матчей',
                 plot_filename=os.path.join(plots_folder, 'first_half_goal_difference.png'),
                 processed_matches_count=processed_matches_count,
                 x_ticks=HISTOGRAM_X_TICKS_FIRST_HALF_GOAL_DIFF, x_range=(0, FIRST_HALF_MAX_MINUTE))

    probability_data = calculate_probability_stats(matches_with_first_goal, matches_with_second_goal,
                                                     matches_with_third_goal, matches_with_fourth_goal)
    if probability_data:
        print("\nВероятность, что после первого гола будет забит:")
        for goal_order, probability in probability_data.items():
            print(f"- {goal_order}: {probability:.2f}%")
        plot_bar_chart(probability_data,
                         title='Вероятность забития последующих голов после первого',
                         xlabel='Следующий гол', ylabel='Вероятность (%)',
                         plot_filename=os.path.join(plots_folder, 'goal_probabilities.png'),
                         processed_matches_count=processed_matches_count,
                         y_range=(0, 100))
    else:
        print("\nНет матчей с первым голом для расчета вероятности последующих голов.")


    top_7_scores = get_top_scores(score_counts, top_n=TOP_SCORES_COUNT)
    print(f"\nТоп-{TOP_SCORES_COUNT} самых часто встречающихся счетов матчей:")
    for score, count in top_7_scores:
        print(f"- Счет '{score}': {count} матчей")

    plot_top_scores_chart(top_7_scores, plots_folder, processed_matches_count)

    # --- Late Goal Probability ---
    if matches_0_0_at_70[0] > 0:
        probability_goal_after_70_if_0_0 = (matches_0_0_at_70_goal_after_70[0] / matches_0_0_at_70[0]) * 100
        print(f"\nВероятность гола после {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуты в матчах, где счет был 0-0 к {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуте: {probability_goal_after_70_if_0_0:.2f}%")
    else:
        print(f"\nНет матчей со счетом 0-0 к {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуте для расчета вероятности гола после {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуты.")


if __name__ == "__main__":
    analyze_football_stats()