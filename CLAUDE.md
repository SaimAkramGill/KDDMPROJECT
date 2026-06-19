# KDDM Ice Hockey Scouting Project

> **Course:** Knowledge Discovery and Data Mining (KDDM)
> **Semester:** Summer 2026
> **Phase I Deadline:** 2026-05-03 (upload only)
> **Phase II Deadline:** 2026-06-19 (live presentation, 10 min max)
> **Full reference:** `docs/PROJECT_UNDERSTANDING.md`

## Team

| Member | Matriculation # | Responsibility |
|--------|----------------|----------------|
| Muhammad Saim Akram | 12344542 | Phase I (preprocessing, cleaning, EDA) |
| Zamarak Sufyan | — | Phase II (modeling, evaluation) |

Both names + matriculation numbers must appear on the title slide of every submission.

---

## 1. What This Project Is

Build a "talent scout in a box" that predicts whether an ice hockey player would be selected for a successful Austrian ice hockey team. The dataset is **synthetic but intentionally messy** — handling that mess IS the assignment.

- **Phase I (current):** Preprocessing, cleaning, EDA. **No target variable available.**
- **Phase II:** Modeling and evaluation. Target variable will be provided.

There are **10,000 players** spread across **7 files** in **5 different formats**.

---

## 2. Phase I Deliverables (Slides + Code)

Exact slide structure required for Phase I upload:

| Slide # | Topic | Count |
|---------|-------|-------|
| 0 | Title slide | Optional |
| 1 | Approach | Exactly 1 |
| 2-3 | Preprocessing | Exactly 2 |
| 4-5 | Data relationships | Exactly 2 |
| 6-7 | Data cleaning / imputation | Exactly 2 |
| 8 | Most important findings | Exactly 1 |

**Total: 8 required slides + optional title (9 max).**

Phase II later requires: 1 feature selection, 1 method, 1-2 evaluation, 1 discussion (4-5 slides).

---

## 3. The 7 Data Files (Single Source of Truth)

| File | Format | Rows | Cols | Join Key | Difficulty |
|------|--------|------|------|----------|-----------|
| `data/performance.tsv` | TSV | 10,000 | 29 | `international_id` (int) | Easy |
| `data/contracts_competition.pkl` | Pickle | 10,000 | 8 | `international_id` (int) | Easy |
| `data/hockey_scouting_notes.csv` | CSV | 10,000 | 7 | `international_id` (int) | Easy |
| `data/identity_card_1.csv` | CSV | 9,112 | 8 | `international_id` + `medical_id` | Easy |
| `data/identity_card_0.tsv` | TSV | 888 | 8 | Roman numeral + UUID | **Hard** |
| `data/medical_information.xlsx` | Excel | 10,000 | 11 | `medical_id` (UUID) | Medium |
| `data/moms_notes.json` | JSON | 10,000 | 13 | **NONE** (name match) | **Hard** |

### Join architecture

```
performance.tsv ---------------+
contracts_competition.pkl -----+
hockey_scouting_notes.csv -----+-- international_id (integer, 1-10000)
identity_card_1.csv -----------+
                               |
  (identity_card_1 has both international_id AND medical_id)
                               |
identity_card_0.tsv ------------- numerus_internationalis_ad_identitatem (ROMAN NUMERAL!)
                                  numerus_identificationis_ad_medicinam (UUID)
                               |
                               +-- medical_id (UUID) -- medical_information.xlsx

moms_notes.json --------------- NO ID! Must join by first_name + last_name via identity cards
```

### Standard join strategy

1. Convert `identity_card_0` Roman numeral IDs to integers
2. Rename `identity_card_0` Latin columns to English
3. Concatenate `identity_card_0` (888) + `identity_card_1` (9,112) = 10,000 unique identity rows
4. Left-join performance, contracts, scouting onto identity via `international_id`
5. Left-join medical onto identity via `medical_id`
6. Join `moms_notes` via name matching (strip Latin suffixes first)

Verify final shape is **10,000 rows, no duplicates on `international_id`**.

---

## 4. File Loading Cheat Sheet

```python
import pandas as pd
import json

# Easy ones
performance = pd.read_csv('data/performance.tsv', sep='\t')
contracts = pd.read_pickle('data/contracts_competition.pkl')
scouting = pd.read_csv('data/hockey_scouting_notes.csv')
identity_1 = pd.read_csv('data/identity_card_1.csv')

# Identity card 0 — Latin + Roman numerals
identity_0 = pd.read_csv('data/identity_card_0.tsv', sep='\t')
# Then: rename columns, convert Roman numerals, optionally strip Latin name suffixes

# Medical — non-standard Excel header
medical = pd.read_excel(
    'data/medical_information.xlsx',
    header=None,
    skiprows=3
)
medical.columns = [
    'medical_id', 'height', 'weight', 'age_in_years', 'shoe_size',
    'body_fat_percentage', 'fitness_level', 'sprint_time',
    'medical_information', 'return_date', 'physician_signature'
]

# Mom's notes — no ID column
with open('data/moms_notes.json', 'r', encoding='utf-8') as f:
    moms_data = json.load(f)
moms_notes = pd.DataFrame(moms_data)
```

---

## 5. Column Name Mapping — identity_card_0 (Latin → English)

| Latin (identity_card_0.tsv) | English (identity_card_1.csv) |
|-----------------------------|------------------------------|
| `numerus_internationalis_ad_identitatem` | `international_id` |
| `numerus_identificationis_ad_medicinam` | `medical_id` |
| `praenomen` | `first_name` |
| `cognomen` | `last_name` |
| `sexus` | `gender` |
| `aetas_annorum` | `age` |
| `urbs_natalis` | `birth_city` |
| `natio` | `nationality` |

Roman numeral examples to expect: `CCLXXV` = 275, `MMDLX` = 2560, `MMMMMMMDCCVIII` = 7708. Use a converter or the `roman` PyPI package.

Latinized name suffixes to strip when joining with `moms_notes`: `-ius`, `-us` (e.g., `Keithus` → `Keith`, `Alleeius` → `Allee`, `Kohlius` → `Kohl`).

---

## 6. Feature Groups (~67 total features)

### Group 1: Performance / Game Statistics (28 features)
**Source:** `performance.tsv` — likely the most predictive set.

`goals`, `assists`, `num_of_shots`, `shot_speed`, `shot_attempts`, `shooting_percentage`, `high_danger_shots`, `medium_danger_shots`, `low_danger_shots`, `save_percentage` (goalie), `winning_goals`, `power_play_time`, `power_play_goals`, `time_on_ice`, `faceoff_win_percentage`, `puck_touches`, `puck_recoveries`, `puck_possession_time`, `penalty_kill_time`, **`penality_minutes` (typo — rename to `penalty_minutes`)**, `penalties_taken`, `time_between_penalties`, `goals_against_total` (goalie), `goals_against_average` (goalie), `passes_attempted`, `passes_completed`, `pass_completion_rate`, `games_missed_due_to_injury`.

### Group 2: Personal / Biographical (13 features) — mostly distractors
**Source:** `moms_notes.json`

Useful: `age` (compare to `age_in_years`), maybe `school_grade`, maybe `years_in_usa`, maybe `personal_notes` (text).
**Distractors to drop:** `eye_color`, `stuffed_animal_name`, `favourite_board_game`, `favourite_child_info`, `favourite_tv_show` (100% empty), `favourite_food` (100% empty).

### Group 3: Physical / Medical (10 features)
**Source:** `medical_information.xlsx`

`height` (mixed m/cm), `weight` (mixed kg/bare), `age_in_years` (dup of `age`), `shoe_size`, `body_fat_percentage` (mixed %/bare), `fitness_level` (mixed casing + literal string `Nan`), `sprint_time`, `medical_information` (85% null, free text), `return_date` (3 date formats), **`physician_signature` (joke values — DROP)**.

### Group 4: Role / Experience (6 features)
**Source:** `hockey_scouting_notes.csv`

`position`, `dominant_hand` (651 missing), `experience_level`, `years_played`, `years_pro`, `scout_notes` (1,038 missing, free text).

### Group 5: Contract / Career (7 features)
**Source:** `contracts_competition.pkl`

`contracts_signed` (122 missing), `salary`, `captain` (object dtype), `won_championship` (object dtype), `jersey_number` (probably noise), `draft_year`, `number_of_previous_teams` (206 missing).

---

## 7. Critical Watch-Outs

### 7.1 The 37 systematic missing rows
Exactly **37 rows** are missing across nearly every column in every file. Likely the same 37 players have systematically incomplete records. Investigate and discuss as a finding.

### 7.2 Column name typo
`performance.tsv` has `penality_minutes` (misspelled). Rename to `penalty_minutes` after load.

### 7.3 Mixed units / formats requiring parsing

| Field | Inconsistencies |
|-------|----------------|
| `gender` / `sexus` | `male`, `m`, `female`, `f`, `other` |
| `nationality` / `natio` | `austria`, `Austria`, `AUSTRIA`, `fRANCE`, `sWITZERLAND` |
| `fitness_level` | `Average`, `average`, `Good`, `good`, `Elite`, `elite`, string `Nan` |
| `height` | `1.79m` vs `180.1135cm` |
| `weight` | `82.4772kg` vs `93.8985` (no unit) |
| `body_fat_percentage` | `20.87%` vs `23.38` (no symbol) |
| `return_date` | `2025-01-12` vs `2025/01/05` vs `01-02-2025` |

### 7.4 Object-typed booleans
`captain` and `won_championship` in `contracts_competition.pkl` are stored as object with mixed True/False/NaN. Convert to clean boolean (or nullable boolean).

### 7.5 Duplicate / redundant features
- `age` (moms_notes) vs `age_in_years` (medical) — compare; pick one.
- `pass_completion_rate` should ≈ `passes_completed / passes_attempted` — verify.
- `shooting_percentage` should relate to `goals / num_of_shots` — verify.

### 7.6 Features to DROP (distractors / noise)

| Feature | Reason |
|---------|--------|
| `eye_color` | No hockey relevance |
| `stuffed_animal_name` | Obvious noise |
| `favourite_board_game` | Obvious noise |
| `favourite_child_info` | Obvious noise, PII-adjacent |
| `favourite_tv_show` | 100% empty strings |
| `favourite_food` | 100% empty strings |
| `physician_signature` | Joke values (`Dr. House`, `Doctor Strange`) |
| `first_name`, `last_name` | Identifiers — keep only for joining, then drop |

### 7.7 Goalie-specific features
`save_percentage`, `goals_against_total`, `goals_against_average` only make sense for goalies. Don't treat goalie zeros as outliers for skaters and vice versa — segment analysis by `position` when needed.

### 7.8 No target variable in Phase I
Phase I is explicitly preprocessing/EDA **without** the target. Do **not** invent a proxy target. The target will be released for Phase II.

### 7.9 Intentional outliers
Some "outliers" are intentional synthetic data points. Still detect/handle as taught, but document your reasoning rather than silently dropping.

---

## 8. Missing Value Inventory

| File | Column | Missing | % | Notes |
|------|--------|---------|---|-------|
| All files | (systematic 37) | 37 | 0.37% | Same players everywhere |
| `performance.tsv` | All feature cols | 37 each | 0.37% | Systematic |
| `hockey_scouting_notes.csv` | `dominant_hand` | 651 | 6.51% | Beyond baseline |
| `hockey_scouting_notes.csv` | `scout_notes` | 1,038 | 10.38% | Significant |
| `contracts_competition.pkl` | `contracts_signed` | 122 | 1.22% | Beyond baseline |
| `contracts_competition.pkl` | `number_of_previous_teams` | 206 | 2.06% | Beyond baseline |
| `medical_information.xlsx` | `medical_information` | 8,518 | 85.18% | Most players healthy |
| `medical_information.xlsx` | `fitness_level` | 358 | 3.58% | Beyond baseline |
| `moms_notes.json` | `personal_notes` | 2,527 | 25.27% | Quarter missing |
| `moms_notes.json` | `favourite_board_game` | 431 | 4.31% | |
| `moms_notes.json` | `eye_color` | 297 | 2.97% | Drop column anyway |
| `moms_notes.json` | `favourite_tv_show` | 0 (null) | 0% | 100% empty strings |
| `moms_notes.json` | `favourite_food` | 0 (null) | 0% | 100% empty strings |
| `identity_card_1.csv` | All except IDs | 37 each | 0.41% | Systematic |
| `identity_card_0.tsv` | (none) | 0 | 0% | Clean |

---

## 9. Phase I Roadmap (Step Checklist)

### Step 1 — Environment
- Jupyter notebook or Python script under `src/` or notebook in project root.
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl`, `json` (stdlib).

### Step 2 — Inspect each file individually
Load each file, print `.shape`, `.dtypes`, `.head()`, `.describe()`, `.isnull().sum()`. Document findings (feeds preprocessing slides).

### Step 3 — Handle identity_card_0
- Roman numeral → integer converter.
- Rename Latin columns per Section 5 table.
- Optionally strip Latin name suffixes for join.

### Step 4 — Handle medical_information.xlsx header
- `pd.read_excel(..., header=None, skiprows=3)`.
- Assign columns manually (see Section 4).

### Step 5 — Merge identity cards
- Concat cleaned `identity_card_0` (888) + `identity_card_1` (9,112).
- Verify 10,000 unique `international_id` values.

### Step 6 — Build master DataFrame
- Left-joins as described in Section 3.
- Verify final shape: 10,000 rows.

### Step 7 — Standardize inconsistent values
- Normalize `gender`, `nationality`, `fitness_level` (lowercase, fix typos, handle string `Nan`).
- Parse `height`, `weight`, `body_fat_percentage` (strip units, convert to common unit).
- Parse `return_date` (3 formats → datetime).
- Rename `penality_minutes` → `penalty_minutes`.

### Step 8 — Handle missing values
- Confirm the systematic 37 rows are the same players.
- Choose imputation per column type:
  - Numeric → mean/median/mode (or keep NaN).
  - Categorical → mode or "Unknown".
  - Text → empty string or NaN.
- Document reasoning (slides 6-7).

### Step 9 — Drop irrelevant features
See Section 7.6. Document what was dropped and why.

### Step 10 — Detect outliers
- Box plots, z-scores, or IQR.
- Position-aware analysis (goalies vs skaters).
- Some outliers are intentional — document, don't silently remove.

### Step 11 — EDA
- Correlation matrix (heatmap) for numeric features.
- Distribution plots for key features.
- Scatter / pair plots.
- Patterns by position.
- Verify `age` vs `age_in_years` consistency.
- Verify computed features (shooting %, pass completion).

### Step 12 — Build slides (exactly 8 required)
1. Approach
2-3. Preprocessing
4-5. Data relationships
6-7. Cleaning / imputation
8. Findings

### Step 13 — Package and submit
- Code runs end-to-end from raw → clean.
- Upload slides + code to TeachCenter before **2026-05-03**.

---

## 10. Working Conventions for This Repo

### Code style
- Use pandas idioms; avoid manual row loops.
- When parsing mixed-unit strings, use regex with named groups; document the regex.
- Prefer `.copy()` after slicing to silence `SettingWithCopyWarning` cleanly.
- Save intermediate cleaned data under `data/newdata/` (already established — see `master_merged.csv`).

### Notebook hygiene
- The two notebooks in use: `dataCleaning.ipynb` and `dataClean_Manual.ipynb`.
- Restart-and-run-all should produce the same output. No hidden state.
- Clear large outputs before committing to keep diffs reviewable.

### Source files
- Helper Python under `src/`. The presentation builder is `src/build_presentation.py`.
- Figures go in `docs/figures/`.

### Reproducibility checklist before submission
1. Restart kernel, run all cells, confirm no errors.
2. Final cleaned DataFrame is exactly 10,000 rows.
3. Every dropped column has a one-line justification in the notebook.
4. Every imputation choice has a one-line justification.
5. The 37-row systematic missing pattern is explicitly addressed.

---

## 11. What NOT to Do

- Do **not** fabricate a target variable for Phase I. The instructors withhold it deliberately.
- Do **not** drop the 37 systematic rows without first investigating whether they share traits (could be a finding worth a slide).
- Do **not** treat goalie-only zeros as outliers for skaters (or vice versa).
- Do **not** silently keep `physician_signature`, `eye_color`, or the 100%-empty columns — they are tests.
- Do **not** use `--no-verify`, `git reset --hard`, or other destructive shortcuts without explicit user approval.
- Do **not** add features, abstractions, or "flexibility" beyond what the assignment requires (Karpathy simplicity rule).

---

## 12. References

- **Full project brief:** `docs/PROJECT_UNDERSTANDING.md` (671 lines — single source of truth)
- **Phase I slides (in progress):** `docs/Phase1_Presentation.pptx`
- **Figures:** `docs/figures/`
- **Notebooks:** `dataCleaning.ipynb`, `dataClean_Manual.ipynb`
- **Merged output:** `data/newdata/master_merged.csv`
- **Presentation builder:** `src/build_presentation.py`

When project context is unclear, default to `docs/PROJECT_UNDERSTANDING.md`.
