import json
import os
from collections import Counter
import logging
import matplotlib
import matplotlib.pyplot as plt  # Оставляем для стилей matplotlib
import plotly.graph_objects as go
from plotly.offline import plot  # Для сохранения HTML файлов

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
FIRST_HALF_MAX_MINUTE = 45

# --- Bins and Ticks Configuration for Histograms ---
GOAL_BINS_FIRST_HALF = range(0, FIRST_HALF_MAX_MINUTE + 25, 5)
GOAL_BINS_SECOND_HALF = range(46, 130, 5)
GOAL_DIFF_BINS = range(0, 95, 5)
FIRST_HALF_GOAL_DIFF_BINS = range(0, FIRST_HALF_MAX_MINUTE + 5, 2)

HISTOGRAM_X_TICKS_FIRST_HALF = range(0, FIRST_HALF_MAX_MINUTE + 1, 5)
HISTOGRAM_X_TICKS_SECOND_HALF = range(45, 131, 10)
HISTOGRAM_X_TICKS_GOAL_DIFF = range(0, 91, 10)
HISTOGRAM_X_TICKS_FIRST_HALF_GOAL_DIFF = range(0, FIRST_HALF_MAX_MINUTE + 1, 5)


# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# --- Matplotlib Style Setup (для общих стилей, если нужно) ---
plt.style.use(PLOT_STYLE) # Можно оставить для общей стилизации, хотя Plotly имеет свои стили
matplotlib.rcParams['font.size'] = FONT_SIZE
matplotlib.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
matplotlib.rcParams['axes.labelsize'] = LABEL_FONT_SIZE
matplotlib.rcParams['xtick.labelsize'] = XTICK_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = YTICK_FONT_SIZE
matplotlib.rcParams['axes.facecolor'] = 'whitesmoke'


def parse_minute_string(minute_string: str) -> tuple[str, int]:
    """Разбирает строку минуты, извлекая период времени и минуту."""
    minute_string = minute_string.rstrip("'’")
    try:
        if '+' in minute_string:
            main_minute, added_time = minute_string.split('+')
            minute = int(main_minute) + int(added_time)
            main_minute_int = int(main_minute)
        else:
            minute = int(minute_string)
            main_minute_int = minute

        if 1 <= main_minute_int <= FIRST_HALF_MAX_MINUTE:
            time_period = "Первый тайм"
        elif 46 <= main_minute_int <= 90:
            time_period = "Второй тайм"
        elif main_minute_int >= 91:
            time_period = "Экстра-тайм"
        else:
            time_period = "Не определено"
        return time_period, minute
    except ValueError:
        logging.warning(f"Не удалось разобрать строку минуты: '{minute_string}'. Возвращается 'Не определено', 0")
        return "Не определено", 0

def _extract_goal_data(match_data: dict) -> tuple[list[str], list[str], str]:
    """Извлекает минуты голов и счет из данных матча."""
    home_goals = match_data.get('home_goals_minutes', [])
    away_goals = match_data.get('away_goals_minutes', [])
    score = match_data.get('score', 'N/A')
    return home_goals, away_goals, score

def _update_goal_stats(home_goals: list[str], away_goals: list[str], first_half_minutes: list[int],
                       second_half_minutes: list[int], all_goal_minutes: Counter,
                       goals_with_time_period: list[tuple[int, str]],
                       goal_before_70: list[bool], goal_after_70: list[bool]) -> None:
    """Обновляет статистику, связанную со временем и периодами голов."""
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
    """Обновляет счетчики для матчей с первым, вторым, третьим и четвертым голами."""
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
    """Обновляет списки разниц в минутах между голами, включая разницы для первого тайма."""
    if len(goals_with_time_period) >= 2:
        sorted_goals_minutes = sorted(goals_with_time_period, key=lambda x: x[0])
        first_goal_data = sorted_goals_minutes[0]
        second_goal_data = sorted_goals_minutes[1]
        goal_differences.append(second_goal_data[0] - first_goal_data[0])
        if first_goal_data[1] == "Первый тайм" and second_goal_data[1] == "Первый тайм":
            first_half_goal_differences.append(second_goal_data[0] - first_goal_data[0])

def _update_zero_zero_stats(goal_before_70: list[bool], goal_after_70: list[bool],
                           matches_0_0_at_70: list[int], matches_0_0_at_70_goal_after_70: list[int]) -> None:
    """Обновляет счетчики для матчей 0-0 к 70-й минуте и голов после 70-й минуты в таких матчах."""
    if not goal_before_70[0]:
        matches_0_0_at_70[0] += 1
        if goal_after_70[0]:
            matches_0_0_at_70_goal_after_70[0] += 1

def process_match(match_data: dict, first_half_minutes: list[int], second_half_minutes: list[int],
                  goal_differences: list[int], matches_with_first_goal: list[int],
                  matches_with_second_goal: list[int], matches_with_third_goal: list[int],
                  matches_with_fourth_goal: list[int], all_goal_minutes: Counter,
                  score_counts: Counter, first_half_goal_differences: list[int],
                  goal_before_70_global: list[bool], goal_after_70_global: list[bool],
                  matches_0_0_at_70: list[int], matches_0_0_at_70_goal_after_70: list[int]) -> None:
    """Обрабатывает данные одного матча и обновляет статистику."""

    home_goals, away_goals, score = _extract_goal_data(match_data)
    score_counts[score] += 1

    goals_with_time_period = []
    goal_before_70 = [False]
    goal_after_70 = [False]

    _update_goal_stats(home_goals, away_goals, first_half_minutes, second_half_minutes,
                       all_goal_minutes, goals_with_time_period,
                       goal_before_70, goal_after_70)

    num_goals_in_match = len(goals_with_time_period)

    _update_match_goal_counts(num_goals_in_match, matches_with_first_goal, matches_with_second_goal,
                               matches_with_third_goal, matches_with_fourth_goal)

    if num_goals_in_match >= 2:
        _update_goal_differences(goals_with_time_period, goal_differences, first_half_goal_differences)

    _update_zero_zero_stats(goal_before_70, goal_after_70, matches_0_0_at_70, matches_0_0_at_70_goal_after_70)

    if goal_before_70[0]:
        goal_before_70_global[0] += 1
    if goal_after_70[0]:
        goal_after_70_global[0] += 1


def calculate_probability_stats(matches_with_first_goal: list[int], matches_with_second_goal: list[int],
                                 matches_with_third_goal: list[int], matches_with_fourth_goal: list[int]) -> dict[str, float]:
    """Вычисляет вероятности последующих голов после первого гола."""
    probability_data = {}
    first_goal_matches = matches_with_first_goal[0]
    if first_goal_matches > 0:
        probability_data["Второй гол"] = (matches_with_second_goal[0] / first_goal_matches) * 100
        probability_data["Третий гол"] = (matches_with_third_goal[0] / first_goal_matches) * 100
        probability_data["Четвертый гол"] = (matches_with_fourth_goal[0] / first_goal_matches) * 100
    return probability_data

def get_top_scores(score_counts: Counter, top_n: int = TOP_SCORES_COUNT) -> list[tuple[str, int]]:
    """Возвращает топ N самых частых счетов матчей."""
    return score_counts.most_common(top_n)

def plot_histogram_plotly(data: list[int], bins: range, title: str, xlabel: str, ylabel: str,
                          plot_filename: str, processed_matches_count: int, x_ticks: range = None, x_range: tuple[int, int] = None) -> None:
    """Создает и сохраняет интерактивную гистограмму с использованием Plotly."""
    fig = go.Figure(data=[go.Histogram(x=data, xbins=dict(start=bins[0], end=bins[-1], size=bins[1]-bins[0]),
                                     marker_color=COLOR_PALETTE[0], opacity=0.7)])

    fig.update_layout(
        title={
            'text': title,
            'x':0.5, # Центрирование заголовка
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': TITLE_FONT_SIZE + 2}
        },
        xaxis_title={
            'text': xlabel,
            'font': {'size': LABEL_FONT_SIZE + 1}
        },
        yaxis_title={
            'text': ylabel,
            'font': {'size': LABEL_FONT_SIZE + 1}
        },
        xaxis=dict(
            tickvals=list(x_ticks) if x_ticks else None,
            range=x_range,
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        bargap=0.1, # Небольшой зазор между столбцами
        plot_bgcolor='whitesmoke',
        margin=dict(l=50, r=50, b=50, t=100, pad=4) # Настройка отступов
    )

    _annotate_processed_matches_plotly(fig, processed_matches_count)

    plot(fig, filename=plot_filename, auto_open=False) # Сохраняем в HTML и отключаем автооткрытие

def plot_bar_chart_plotly(data: dict[str, float], title: str, xlabel: str, ylabel: str,
                         plot_filename: str, processed_matches_count: int, y_range: tuple[int, int] = None) -> None:
    """Создает и сохраняет интерактивную столбчатую диаграмму с использованием Plotly."""
    labels = list(data.keys())
    values = list(data.values())

    fig = go.Figure(data=[go.Bar(x=labels, y=values,
                                marker_color=COLOR_PALETTE[1:4], opacity=0.7,
                                text=[f'{val:.2f}%' for val in values], # Текст на столбцах
                                textposition='outside')]) # Позиция текста

    fig.update_layout(
        title={
            'text': title,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': TITLE_FONT_SIZE + 2}
        },
        xaxis_title={
            'text': xlabel,
            'font': {'size': LABEL_FONT_SIZE + 1}
        },
        yaxis_title={
            'text': ylabel,
            'font': {'size': LABEL_FONT_SIZE + 1}
        },
        yaxis=dict(
            range=y_range,
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='whitesmoke',
        margin=dict(l=50, r=50, b=50, t=100, pad=4)
    )
    _annotate_processed_matches_plotly(fig, processed_matches_count)
    plot(fig, filename=plot_filename, auto_open=False)

def plot_top_scores_chart_plotly(top_scores: list[tuple[str, int]], plot_folder: str, processed_matches_count: int) -> None:
    """Создает и сохраняет интерактивную горизонтальную столбчатую диаграмму для топ счетов с использованием Plotly."""
    scores = [score for score, count in top_scores]
    counts = [count for score, count in top_scores]

    fig = go.Figure(data=[go.Bar(y=scores[::-1], x=counts[::-1], orientation='h',
                                marker_color=COLOR_PALETTE[4], opacity=0.7,
                                text=counts[::-1], # Текст на столбцах
                                textposition='outside')]) # Позиция текста

    fig.update_layout(
        title={
            'text': 'Топ-7 самых часто встречающихся счетов матчей',
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': TITLE_FONT_SIZE + 2}
        },
        xaxis_title={
            'text': 'Количество матчей',
            'font': {'size': LABEL_FONT_SIZE + 1}
        },
        yaxis_title={
            'text': 'Счет матча',
            'font': {'size': LABEL_FONT_SIZE + 1}
        },
        xaxis=dict(
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            autorange="reversed", # Инвертируем ось Y
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='whitesmoke',
        margin=dict(l=50, r=50, b=50, t=100, pad=4)
    )
    _annotate_processed_matches_plotly(fig, processed_matches_count)
    plot_filename = os.path.join(plot_folder, 'top_scores.html') # Сохраняем как HTML
    plot(fig, filename=plot_filename, auto_open=False)

def _annotate_processed_matches_plotly(fig, processed_matches_count):
    """Добавляет аннотацию о количестве обработанных матчей на график Plotly."""
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.98, # Позиция в правом верхнем углу
        xanchor='right', yanchor='top',
        text=f'Обработано матчей: {processed_matches_count}',
        showarrow=False,
        font=dict(size=9, color='gray')
    )


def _load_json_file(filepath: str) -> list[dict]:
    """Загружает и разбирает JSON файл, обрабатывая различные структуры."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
        if isinstance(file_content, list):
            if file_content and isinstance(file_content[0], list):
                return [match for inner_list in file_content for match in inner_list]
            return file_content
        elif isinstance(file_content, dict) and 'matches' in file_content and isinstance(file_content['matches'], list):
            return file_content['matches']
        elif isinstance(file_content, dict):
            return [file_content]
        else:
            logging.warning(f"Неожиданный формат JSON в {filepath}. Содержимое файла пропускается.")
            return []
    except FileNotFoundError:
        logging.error(f"Файл не найден: {filepath}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Ошибка декодирования JSON в {filepath}. Проверьте формат файла.")
        return []
    except Exception as e:
        logging.exception(f"Ошибка при обработке файла {filepath}")
        return []

def analyze_football_stats(json_folder: str = DEFAULT_JSON_FOLDER, plots_folder: str = DEFAULT_PLOTS_FOLDER) -> None:
    """
    Анализирует футбольную статистику из JSON файлов, генерирует интерактивные графики и выводит результаты.
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
    goal_before_70_global = [0]
    goal_after_70_global = [0]
    matches_0_0_at_70 = [0]
    matches_0_0_at_70_goal_after_70 = [0]

    os.makedirs(plots_folder, exist_ok=True)

    json_files = [filename for filename in os.listdir(json_folder) if filename.endswith(".json")]
    if not json_files:
        logging.warning(f"В папке не найдено JSON файлов: {json_folder}")
        return

    for filename in json_files:
        filepath = os.path.join(json_folder, filename)
        match_list = _load_json_file(filepath)
        if not match_list:
            continue

        for match_data in match_list:
            if not isinstance(match_data, dict):
                logging.warning(f"Ожидался словарь матча, но получен {type(match_data)} в файле {filename}. Элемент пропускается.")
                continue
            process_match(match_data, first_half_minutes, second_half_minutes, goal_differences,
                          matches_with_first_goal, matches_with_second_goal, matches_with_third_goal,
                          matches_with_fourth_goal, all_goal_minutes, score_counts,
                          first_half_goal_differences, goal_before_70_global, goal_after_70_global,
                          matches_0_0_at_70, matches_0_0_at_70_goal_after_70)
            processed_matches_count += 1

    logging.info(f"Обработано матчей для статистики: {processed_matches_count}")

    # --- Построение интерактивных графиков Plotly ---
    plot_histogram_plotly(first_half_minutes, bins=GOAL_BINS_FIRST_HALF,
                 title='Распределение минут голов в первом тайме',
                 xlabel='Минута', ylabel='Количество голов',
                 plot_filename=os.path.join(plots_folder, 'first_half_goals_minutes.html'), # Сохраняем как HTML
                 processed_matches_count=processed_matches_count,
                 x_ticks=HISTOGRAM_X_TICKS_FIRST_HALF)

    plot_histogram_plotly(second_half_minutes, bins=GOAL_BINS_SECOND_HALF,
                 title='Распределение минут голов во втором тайме',
                 xlabel='Минута', ylabel='Количество голов',
                 plot_filename=os.path.join(plots_folder, 'second_half_goals_minutes.html'), # Сохраняем как HTML
                 processed_matches_count=processed_matches_count,
                 x_ticks=HISTOGRAM_X_TICKS_SECOND_HALF)

    plot_histogram_plotly(goal_differences, bins=GOAL_DIFF_BINS,
                 title='Распределение разницы в минутах между первым и вторым голом',
                 xlabel='Разница в минутах', ylabel='Количество матчей',
                 plot_filename=os.path.join(plots_folder, 'goal_difference.html'), # Сохраняем как HTML
                 processed_matches_count=processed_matches_count,
                 x_ticks=HISTOGRAM_X_TICKS_GOAL_DIFF)

    plot_histogram_plotly(first_half_goal_differences, bins=FIRST_HALF_GOAL_DIFF_BINS,
                 title='Распределение разницы в минутах между первым и вторым голом (Первый тайм)',
                 xlabel='Разница в минутах', ylabel='Количество матчей',
                 plot_filename=os.path.join(plots_folder, 'first_half_goal_difference.html'), # Сохраняем как HTML
                 processed_matches_count=processed_matches_count,
                 x_ticks=HISTOGRAM_X_TICKS_FIRST_HALF_GOAL_DIFF, x_range=(0, FIRST_HALF_MAX_MINUTE))

    probability_data = calculate_probability_stats(matches_with_first_goal, matches_with_second_goal,
                                                     matches_with_third_goal, matches_with_fourth_goal)
    if probability_data:
        print("\nВероятность, что после первого гола будет забит:")
        for goal_order, probability in probability_data.items():
            print(f"- {goal_order}: {probability:.2f}%")
        plot_bar_chart_plotly(probability_data,
                         title='Вероятность забития последующих голов после первого',
                         xlabel='Следующий гол', ylabel='Вероятность (%)',
                         plot_filename=os.path.join(plots_folder, 'goal_probabilities.html'), # Сохраняем как HTML
                         processed_matches_count=processed_matches_count,
                         y_range=(0, 100))
    else:
        print("\nНет матчей с первым голом для расчета вероятности последующих голов.")


    top_7_scores = get_top_scores(score_counts, top_n=TOP_SCORES_COUNT)
    print(f"\nТоп-{TOP_SCORES_COUNT} самых часто встречающихся счетов матчей:")
    for score, count in top_7_scores:
        print(f"- Счет '{score}': {count} матчей")

    plot_top_scores_chart_plotly(top_7_scores, plots_folder, processed_matches_count) # Используем Plotly версию

    # --- Вероятность позднего гола ---
    if matches_0_0_at_70[0] > 0:
        probability_goal_after_70_if_0_0 = (matches_0_0_at_70_goal_after_70[0] / matches_0_0_at_70[0]) * 100
        print(f"\nВероятность гола после {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуты в матчах, где счет был 0-0 к {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуте: {probability_goal_after_70_if_0_0:.2f}%")

        total_matches_with_goals = goal_before_70_global[0] + goal_after_70_global[0]
        if total_matches_with_goals > 0:
            probability_goal_before_70 = (goal_before_70_global[0] / total_matches_with_goals) * 100
            probability_goal_after_70 = (goal_after_70_global[0] / total_matches_with_goals) * 100
            print(f"\nОбщая вероятность гола до {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуты (среди всех матчей с голами): {probability_goal_before_70:.2f}%")
            print(f"Общая вероятность гола после {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуты (среди всех матчей с голами): {probability_goal_after_70:.2f}%")
    else:
        print(f"\nНет матчей со счетом 0-0 к {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуте для расчета вероятности гола после {MINUTE_THRESHOLD_FOR_LATE_GOAL}-й минуты.")


if __name__ == "__main__":
    analyze_football_stats()