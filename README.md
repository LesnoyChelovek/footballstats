# footballstats
=======
# Football Match Statistics Analyzer

This project consists of two Python scripts designed to analyze football match data from JSON files and generate insightful statistics and visualizations. The scripts calculate various metrics related to goal distribution, match scores, and probabilities, providing a comprehensive overview of football match patterns.

## Scripts Overview

This project includes two main scripts:

*   **`main.py`**: Analyzes football match data and generates static plots (PNG format) using `matplotlib`. It also prints statistical summaries to the console.
*   **`stats_html.py`**:  Similar to `main.py`, but utilizes `plotly` to create interactive plots (HTML format).  Like `main.py`, it also outputs statistical summaries to the console.

Both scripts perform the same statistical analysis but differ in their plotting libraries and output format. `main.py` is suitable for generating static images, while `stats_html.py` is ideal for creating interactive visualizations that can be easily explored in a web browser.

## Features

Both scripts provide the following functionalities:

*   **Data Parsing**: Reads football match data from JSON files, supporting various JSON structures (list of matches, dictionary with a 'matches' key, or a single match dictionary).
*   **Goal Time Analysis**:
    *   Calculates and visualizes the distribution of goals scored in the first and second halves of matches.
    *   Identifies the most frequent goal minutes.
    *   Analyzes the time difference between the first and second goals in matches.
*   **Match Score Analysis**:
    *   Counts the occurrences of different match scores.
    *   Identifies and visualizes the most frequent match scores.
*   **Goal Probability Analysis**:
    *   Calculates the probability of scoring subsequent goals (2nd, 3rd, 4th) after the first goal in a match.
    *   Determines the probability of a goal being scored after the 70th minute in matches that are 0-0 at the 70th minute.
*   **Data Visualization**:
    *   Generates histograms for goal minute distributions and goal time differences.
    *   Creates bar charts for goal scoring probabilities.
    *   Produces horizontal bar charts for visualizing the top most frequent match scores.
*   **Output Formats**:
    *   **`main.py`**: Saves plots as static PNG image files in the `plots` folder and prints text-based statistical results to the console.
    *   **`stats_html.py`**: Saves interactive plots as HTML files in the `plots` folder and prints text-based statistical results to the console.

## Getting Started

### Prerequisites

*   **Python 3.x**:  Ensure you have Python 3 or a later version installed on your system.
*   **Python Libraries**: You need to install the following Python libraries. You can install them using pip:

    ```bash
    pip install matplotlib plotly collections logging
    ```

    Specifically:
    *   `matplotlib`: For static plotting (`main.py`).
    *   `plotly`: For interactive plotting (`stats_html.py`).
    *   `collections`:  Built-in Python library (specifically `Counter`).
    *   `logging`: Built-in Python library for logging.

### Installation

1.  **Clone the repository** (if you are accessing this code from GitHub):
    ```bash
    git clone [repository-url]
    cd [repository-directory]
    ```
2.  **Prepare your JSON data**: Place your JSON files containing football match data into a folder named `JSON` in the same directory as the scripts. If you prefer a different folder name, you can configure this in the script (see Configuration section).

### JSON Data Format

The scripts are designed to handle JSON files containing football match data.  The expected format for each match within the JSON file is a dictionary that should include the following keys:

*   **`home_goals_minutes`**: A list of strings representing the minutes when the home team scored goals.  Minutes can be in the format like `"15"`, `"45+2"`, `"78'"` or `"90+5’"`.
*   **`away_goals_minutes`**: A list of strings representing the minutes when the away team scored goals, in the same format as `home_goals_minutes`.
*   **`score`**: A string representing the final score of the match, e.g., `"2-1"`, `"0-0"`, `"3-2"`.

The JSON file itself can be structured in a few ways:

*   **List of Matches**: A JSON array where each element is a match dictionary as described above.
    ```json
    [
        {
            "home_goals_minutes": ["25", "67"],
            "away_goals_minutes": ["40", "88", "90+3"],
            "score": "2-3"
        },
        {
            "home_goals_minutes": ["12"],
            "away_goals_minutes": [],
            "score": "1-0"
        },
        ...
    ]
    ```
*   **Dictionary with 'matches' Key**: A JSON object with a key named `"matches"` whose value is a list of match dictionaries.
    ```json
    {
        "matches": [
            {
                "home_goals_minutes": ["30"],
                "away_goals_minutes": ["55", "72"],
                "score": "1-2"
            },
            ...
        ]
    }
    ```
*   **Single Match Dictionary**: A JSON object representing a single match (useful for testing or processing one file at a time).
    ```json
    {
        "home_goals_minutes": ["10", "80"],
        "away_goals_minutes": ["45"],
        "score": "2-1"
    }
    ```
*   **List of Lists of Matches**: A JSON array where each element is a list of match dictionaries (for more complex file structures). The script will flatten this structure automatically.

The scripts are designed to be flexible and will attempt to parse these different JSON structures.

## Usage

1.  **Place your JSON files** in the `JSON` folder (or the folder you configured).
2.  **Run the scripts**:

    *   For static plots (PNG, using `matplotlib`):
        ```bash
        python main.py
        ```
    *   For interactive plots (HTML, using `plotly`):
        ```bash
        python stats_html.py
        ```

3.  **Check the output**:
    *   Plots will be saved in the `plots` folder created in the same directory as the scripts.
    *   Statistical summaries, including probabilities and top scores, will be printed to your console.

## Configuration

You can configure various parameters at the beginning of both `main.py` and `stats_html.py` scripts to customize the analysis and output. Here are the key configuration variables:

*   **`LOG_LEVEL`**: Sets the logging level (e.g., `logging.INFO`, `logging.WARNING`, `logging.ERROR`). Default is `logging.INFO`.
*   **`LOG_FORMAT`**: Defines the format of log messages.
*   **`PLOT_STYLE`**: Sets the `matplotlib` plot style. Default is `'ggplot'`.
*   **`FONT_SIZE`, `TITLE_FONT_SIZE`, `LABEL_FONT_SIZE`, `XTICK_FONT_SIZE`, `YTICK_FONT_SIZE`**:  Control font sizes for plot elements.
*   **`COLOR_PALETTE`**:  A list of colors used for plots.
*   **`DEFAULT_JSON_FOLDER`**:  The name of the folder where the script looks for JSON files. Default is `'JSON'`.
*   **`DEFAULT_PLOTS_FOLDER`**: The name of the folder where plots will be saved. Default is `'plots'`.
*   **`TOP_SCORES_COUNT`**:  Number of top scores to display in the top scores chart. Default is `7`.
*   **`TOP_GOAL_MINUTES_COUNT`**:  (Currently not used in the provided code, but might be intended for future features).
*   **`MINUTE_THRESHOLD_FOR_LATE_GOAL`**: The minute threshold to define a "late goal". Default is `70`.
*   **`FIRST_HALF_MAX_MINUTE`**:  The maximum minute considered to be in the first half. Default is `45`.
*   **Bins and Ticks for Histograms (`GOAL_BINS_FIRST_HALF`, `GOAL_BINS_SECOND_HALF`, `GOAL_DIFF_BINS`, `FIRST_HALF_GOAL_DIFF_BINS`, `HISTOGRAM_X_TICKS_FIRST_HALF`, `HISTOGRAM_X_TICKS_SECOND_HALF`, `HISTOGRAM_X_TICKS_GOAL_DIFF`, `HISTOGRAM_X_TICKS_FIRST_HALF_GOAL_DIFF`)**: These variables define the bins and x-axis ticks for the histograms generated by the scripts. You can adjust these to customize the granularity and appearance of the histograms.

To modify these settings, simply open the script (`main.py` or `stats_html.py`) in a text editor and change the values of the configuration variables at the beginning of the file.

## Output

After running the scripts, you will find:

*   **Plots Folder (`plots`)**: This folder will contain the generated plots.
    *   For `main.py`: PNG image files (e.g., `first_half_goals_minutes.png`, `top_scores.png`, etc.).
    *   For `stats_html.py`: HTML files (e.g., `first_half_goals_minutes.html`, `top_scores.html`, etc.). Open these HTML files in a web browser to view the interactive plots.
*   **Console Output**: The scripts will print statistical results to the console, including:
    *   Probabilities of subsequent goals.
    *   Top most frequent match scores with their counts.
    *   Probability of a late goal in 0-0 matches.
    *   Overall probabilities of goals before and after the `MINUTE_THRESHOLD_FOR_LATE_GOAL` minute.
    *   Information about the number of matches processed.

## Contributing

Contributions to this project are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or fix.
3.  **Make your changes** and commit them.
4.  **Submit a pull request** to the main repository.

## License

[Specify License here, e.g., MIT License]

This project is open-source and available under the [License Name] License.

---

**Example Output (Console):**
2023-10-27 10:30:00 - INFO - Processed matches for statistics: 150
Вероятность, что после первого гола будет забит:

Второй гол: 65.22%

Третий гол: 42.39%

Четвертый гол: 25.00%

Топ-7 самых часто встречающихся счетов матчей:

Счет '1-0': 25 матчей

Счет '1-1': 20 матчей

Счет '2-1': 18 матчей

Счет '0-0': 15 матчей

Счет '2-0': 12 матчей

Счет '0-1': 10 матчей

Счет '3-1': 8 матчей

Вероятность гола после 70-й минуты в матчах, где счет был 0-0 к 70-й минуте: 35.71%

Общая вероятность гола до 70-й минуты (среди всех матчей с голами): 72.50%
Общая вероятность гола после 70-й минуты (среди всех матчей с голами): 27.50%

This README provides a comprehensive guide to understanding and using the Football Match Statistics Analyzer scripts.  Enjoy exploring your football data!
