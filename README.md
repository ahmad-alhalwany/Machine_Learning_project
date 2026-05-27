# Machine Learning Project — Sports Match Prediction

A collection of **machine learning notebooks** that predict sports match outcomes using real historical data. Includes two projects: **NBA game prediction** (Ridge Classifier + feature selection) and **Premier League football prediction** (Random Forest + rolling stats).

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data-green)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red?logo=jupyter)](https://jupyter.org)

---

## Overview

This repository demonstrates end-to-end **sports analytics ML pipelines** — from data loading and feature engineering to model training, backtesting, and evaluation. Both projects use real datasets and scikit-learn classifiers with time-aware train/test splits to avoid data leakage.

**Why this project matters:** Sports prediction is a practical ML use case that combines feature engineering (rolling averages, opponent stats), proper temporal validation, and real-world data scraping — skills directly transferable to forecasting problems in any domain.

---

## Projects

| Project | Sport | Model | Target | Dataset |
|---------|-------|-------|--------|---------|
| [NBA_matches_Predict](#1-nba-match-prediction) | Basketball (NBA) | Ridge Classifier | Will team win next game? | `nba_games.csv` (~12MB) |
| [football_matches_predict](#2-premier-league-football-prediction) | Football (EPL) | Random Forest | Will team win the match? | `matches.csv` (~1,500 rows) |

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                  Machine Learning Pipeline                   │
├──────────────────────────┬──────────────────────────────────┤
│   NBA_matches_Predict    │   football_matches_predict        │
│                          │                                   │
│  nba_games.csv           │  matches.csv (FBref EPL data)     │
│       ↓                  │       ↓                           │
│  Feature engineering   │  Category encoding + rolling avg  │
│  Rolling 10-game stats   │  3-game rolling (gf, ga, shots…)  │
│  Opponent stat merge     │  Train/test date split            │
│       ↓                  │       ↓                           │
│  RidgeClassifier         │  RandomForestClassifier           │
│  SequentialFeatureSelect │  50 estimators                    │
│  TimeSeriesSplit CV      │                                   │
│       ↓                  │       ↓                           │
│  Season-based backtest   │  Accuracy + Precision metrics     │
└──────────────────────────┴──────────────────────────────────┘
```

---

## 1. NBA Match Prediction

**Folder:** `NBA_matches_Predict/`  
**Notebook:** `NBA_predict.ipynb`  
**Data:** `nba_games.csv`

### Goal

Predict whether an NBA team will **win their next game** based on historical team stats (points, rebounds, assists, etc.).

### Pipeline

1. Load and sort game data by date
2. Create target: shift `won` column by -1 per team (`target = next game result`)
3. Remove columns with null values
4. **MinMaxScaler** normalization
5. **Sequential Feature Selector** — pick top 30 features using `TimeSeriesSplit` CV
6. **Rolling 10-game averages** per team per season
7. **Merge opponent stats** for full matchup features
8. **Season-based backtest** — train on past seasons, predict current season
9. Evaluate with `accuracy_score`

### Model

```python
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector

rr = RidgeClassifier(alpha=1)
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, cv=TimeSeriesSplit(n_splits=3))
```

### Data scraping (optional section in notebook)

Uses **Playwright + BeautifulSoup** to scrape [Basketball-Reference.com](https://www.basketball-reference.com) for seasons 2016–2022. Requires:

```bash
pip install playwright beautifulsoup4
playwright install
```

Create directories before scraping:

```bash
mkdir -p data/standings data/scores
```

---

## 2. Premier League Football Prediction

**Folder:** `football_matches_predict/`  
**Notebook:** `Untitled.ipynb`  
**Data:** `matches.csv`

### Goal

Predict whether a Premier League team will **win a match** (binary: Win = 1, Loss/Draw = 0).

### Pipeline

1. Load `matches.csv` (FBref Premier League match data)
2. Parse dates and encode features:
   - `venue_code` — Home/Away category encoding
   - `opp_code` — opponent category encoding
   - `hour` — match kickoff hour
   - `day_code` — day of week
3. Create target: `target = (result == "W")`
4. **Train/test split by date:**
   - Train: before `2023-01-01`
   - Test: after `2023-01-01`
5. **Rolling 3-game averages** for: goals for/against, shots, shots on target, distance, free kicks, penalties
6. Train **RandomForestClassifier** and evaluate accuracy + precision
7. **Merge dual-team predictions** — combine home and away model outputs for match-level analysis

### Model

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
predictors = ["venue_code", "opp_code", "hour", "day_code"]
# Extended with rolling features: gf_rolling, ga_rolling, sh_rolling, etc.
```

### Results (baseline)

| Metric | Value |
|--------|-------|
| Accuracy (basic features) | ~61.4% |
| Precision | Evaluated via `precision_score` |

### Data scraping (optional section in notebook)

Scrapes [FBref Premier League Stats](https://fbref.com/en/comps/9/Premier-League-Stats) using `requests` + `BeautifulSoup` + `pd.read_html`:

```python
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
data = requests.get(standings_url)
matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| ML | scikit-learn (RidgeClassifier, RandomForestClassifier, SequentialFeatureSelector) |
| Data | Pandas, NumPy |
| Scraping | Playwright, BeautifulSoup, requests |
| Environment | Jupyter Notebook |

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| **Python** | 3.8+ (3.11 recommended) | `python --version` |
| **pip** | Latest | `pip --version` |
| **Jupyter** | Latest | `jupyter --version` |

---

## How to Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/ahmad-alhalwany/Machine_Learning_project.git
cd Machine_Learning_project
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install pandas numpy scikit-learn jupyter ipykernel matplotlib
```

**For NBA web scraping section (optional):**

```bash
pip install playwright beautifulsoup4 lxml html5lib
playwright install
```

**For football web scraping section (optional):**

```bash
pip install requests beautifulsoup4 lxml html5lib
```

### Step 4 — Run NBA prediction notebook

```bash
cd NBA_matches_Predict
jupyter notebook NBA_predict.ipynb
```

Run all cells sequentially. The dataset `nba_games.csv` is included — no scraping required for the ML pipeline.

### Step 5 — Run football prediction notebook

```bash
cd ../football_matches_predict
jupyter notebook Untitled.ipynb
```

Run all cells sequentially. The dataset `matches.csv` is included.

---

## Project Structure

```text
Machine_Learning_project/
├── NBA_matches_Predict/
│   ├── NBA_predict.ipynb      # NBA win prediction pipeline
│   └── nba_games.csv          # NBA historical game stats (~12MB)
├── football_matches_predict/
│   ├── Untitled.ipynb         # Premier League win prediction pipeline
│   └── matches.csv            # EPL match data (~1,500 rows)
└── README.md
```

---

## Key ML Concepts Demonstrated

| Concept | NBA Project | Football Project |
|---------|-------------|------------------|
| Feature selection | SequentialFeatureSelector (30 features) | Manual + rolling features |
| Normalization | MinMaxScaler | — |
| Rolling averages | 10-game window | 3-game window |
| Opponent features | Merge opponent rolling stats | Dual-team prediction merge |
| Time-aware validation | Season-based backtest | Date-based train/test split |
| Cross-validation | TimeSeriesSplit (3 folds) | — |
| Metrics | accuracy_score | accuracy_score + precision_score |
| Web scraping | Playwright + Basketball-Reference | requests + FBref |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: sklearn` | Run `pip install scikit-learn` |
| NBA notebook slow on feature selection | SequentialFeatureSelector is compute-heavy — reduce `n_features_to_select` or dataset size |
| `FileNotFoundError: nba_games.csv` | Run notebook from inside `NBA_matches_Predict/` directory |
| `FileNotFoundError: matches.csv` | Run notebook from inside `football_matches_predict/` directory |
| NBA scraping fails | Create `data/standings/` and `data/scores/` folders first |
| FBref scraping blocked | FBref may rate-limit — use included `matches.csv` instead |
| Football date parsing error | Ensure `matches["date"]` is converted: `pd.to_datetime(matches["date"])` |

---

## Screenshots

| NBA Feature Selection | Football Predictions | Football Rolling Stats |
|-----------------------|---------------------|------------------------|
| _Add screenshot_ | _Add screenshot_ | _Add screenshot_ |

---

## Roadmap

- [ ] Rename `Untitled.ipynb` → `football_predict.ipynb`
- [ ] Add root `requirements.txt`
- [ ] Unify scraping scripts into reusable modules
- [ ] Add XGBoost / LightGBM comparison
- [ ] Build FastAPI prediction endpoint
- [ ] Add model persistence (`joblib.dump`)

---

## Author

**Ahmad Alhalwany**

- GitHub: [@ahmad-alhalwany](https://github.com/ahmad-alhalwany)
- Repository: [Machine_Learning_project](https://github.com/ahmad-alhalwany/Machine_Learning_project)

---

## License

MIT — free to use for learning and reference.
