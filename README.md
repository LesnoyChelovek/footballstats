# Football Match Statistics Analyzer

Python scripts that read football match JSON, aggregate goal-minute statistics, and write charts.

`main.py` and `stats_html.py` both use the calculations in `stats_core.py`. Run the scripts from the project root so the relative paths `JSON/` and `plots/` resolve.

## Scripts

* **`stats_core.py`** loads the match files and computes the statistics. It parses a goal minute into `GoalTime`, normalizes the score, sorts the goals of a match, and keeps the totals in `AggregatedStats` and `YearlyStats`.
* **`stats_html.py`** writes an interactive Plotly dashboard to `plots/report.html` and one HTML file per chart. The half-by-half charts can switch between goal counts and percentages in the browser. `--offline` embeds Plotly so the pages open without a network.
* **`main.py`** prints the same text summary and writes static matplotlib PNGs into `plots/`. It has no command-line options.

## Statistics

* **Minutes.** The first half is `1–45` plus stoppage `45+1…45+N`. The second half is `46–90` plus `90+1…90+N`. A minute of 91 or later without `+` (`93`, `105`, `118`) and extra-time stoppage (`105+1`, `120+2`) count as extra time.
* **Order.** Goals in a match are sorted by period, base minute, then added minute, so `45+3` stays before `47`. The gap between the first and second goal is zero or positive.
* **Scores.** A string such as `0 - 3*` becomes `0 - 3`. The final score is counted for every loaded match.
* **Distributions.** Stoppage time is counted apart from the regular minute. On the HTML charts, the hover text for minutes 45 and 90 says those bars include stoppage time that the source recorded as 45 or 90. The dashboard explains the spikes at the end of each half.
* **Intervals and probabilities.** Gap between the first and second goal for the whole match, and again when both of those goals are in the first half. Share of matches that reach a second, third, and fourth goal after the first. Share of matches that were 0-0 at minute 70 and scored afterwards.
* **Seasons.** The four-digit year in the filename is the season. For each season the scripts keep the number of matches, the share that has goal minutes, and the average goals per match that has minutes. That season chart is only in the HTML dashboard.

The charts are built from these aggregated counts. `plots/report.html` is about 100 KB, and each standalone chart file is about 12–24 KB.

## Installation

Python 3.11 or newer. The virtualenv in this checkout uses Python 3.14, Plotly 7.1.0, and matplotlib 3.11.2.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists Plotly and matplotlib.

## Interactive dashboard

```bash
python stats_html.py
```

Open `plots/report.html`.

```text
--json-folder FOLDER   Match JSON directory (default: JSON)
--out FOLDER           Output directory (default: plots)
--offline              Embed plotly.js for offline viewing
--top-n N              How many frequent scores to show (default: 7)
```

The script also writes one HTML file per chart into the output folder. `--separate-files` is accepted and does not turn those files off.

```bash
python stats_html.py --offline --out my_report
python stats_html.py --top-n 10
```

HTML files:

* `report.html`
* `first_half_goals_minutes.html`
* `second_half_goals_minutes.html`
* `goal_difference.html`
* `first_half_goal_difference.html`
* `goal_probabilities.html`
* `top_scores.html`
* `yearly_stats.html`

## Static charts

```bash
python main.py
```

`main.py` always reads `JSON/` and writes `plots/`. The console summary uses the top 7 scores.

PNG files:

* `first_half_goals_minutes.png`
* `second_half_goals_minutes.png`
* `goal_difference.png`
* `first_half_goal_difference.png`
* `goal_probabilities.png`
* `top_scores.png`

## JSON data

Match files live in `JSON/` and are named `match_data2000.json` through `match_data2026.json`. Any other `.json` file in that folder is loaded too. A filename without a four-digit year is grouped as `unknown`.

A file may be a list of match objects, a list of lists of match objects (one level is flattened), an object with a `matches` list, or a single match object.

```json
{
  "home_team": "Brentford",
  "away_team": "Arsenal",
  "score": "1 - 3",
  "home_goals_minutes": ["13'"],
  "away_goals_minutes": ["29'", "50'", "53'"]
}
```

The scripts use `score`, `home_goals_minutes`, and `away_goals_minutes`. Team names stay in the files and are not part of the aggregates. A minute field may be a list or one string. Accepted forms are `15`, `45+2`, `78'`, and `90+5'`. In these files a minute usually ends with `’'` (a curly apostrophe followed by a straight one). The parser strips both. `NA` and `N/A` mean the minute is missing.

If any goal minute in a match is missing, the score is still counted and the match is left out of the minute charts. An empty minute list means no goals. A non-zero score with empty minute lists still enters the minute statistics as a match with no timed goals.

## Generated files

The scripts create `plots/`. That directory is listed in `.gitignore`.
