# KDDM Project Understanding: Ice Hockey Talent Scouting

> **Course:** Knowledge Discovery and Data Mining (KDDM)
> **Semester:** Summer 2026
> **Last Updated:** 2026-04-27
> **Phase I Deadline:** 2026-05-03
> **Phase II Deadline:** 2026-06-19

---

## Table of Contents

1. [What This Project Is Actually About](#1-what-this-project-is-actually-about)
2. [The Two Phases and Their Deliverables](#2-the-two-phases-and-their-deliverables)
3. [Exact Slide Requirements](#3-exact-slide-requirements)
4. [Deadlines](#4-deadlines)
5. [Dataset Overview](#5-dataset-overview)
6. [Features Grouped by Category](#6-features-grouped-by-category)
7. [Key Things to Watch Out For](#7-key-things-to-watch-out-for)
8. [Data Quality Issues Found During Inspection](#8-data-quality-issues-found-during-inspection)
9. [Suggested Development Roadmap for Phase I](#9-suggested-development-roadmap-for-phase-i)

---

## 1. What This Project Is Actually About

**In plain language:** You are building a system that predicts whether an ice hockey player would be selected for a successful Austrian ice hockey team. Think of it as a "talent scout in a box" -- given a bunch of data about a player (their stats, physical attributes, career history, etc.), can you predict whether they would make the cut?

The project is split into two parts:

- **Phase I** is about understanding and cleaning the data. You do NOT have access to the answer key (the target variable) yet. Your job is to explore, clean, and prepare the data so it is ready for modeling.
- **Phase II** is about building and evaluating machine learning models that actually make the prediction. You will receive the target variable for this phase.

The dataset is **synthetic** (the instructors generated it), but it intentionally mimics real-world messiness: missing values, inconsistent formats, outliers, irrelevant features, and data spread across multiple files in different formats. This is by design -- handling this mess IS the assignment.

---

## 2. The Two Phases and Their Deliverables

### Phase I: Preprocessing, Cleaning, and EDA

**Goal:** Understand the data, clean it, find patterns -- all WITHOUT knowing the target variable.

**Deliverables:**
- A presentation (uploaded to TeachCenter, NOT presented live)
- Your code (uploaded to TeachCenter)

**What you need to demonstrate:**
- What preprocessing steps you took
- What interesting things you found (missing values, outliers)
- What relationships exist in the data
- How you approached unknown/messy data

### Phase II: Modeling and Evaluation

**Goal:** Train models to predict the target variable. Evaluate how well they work.

**Deliverables:**
- A presentation (presented live, max 10 minutes)
- Your code (uploaded to TeachCenter)

**What you need to demonstrate:**
- Your training strategy
- How you selected features (which columns to use and why)
- How you split your data (train/test/validation)
- What tools and approaches you used
- How you evaluated your models (you choose the evaluation metrics)

---

## 3. Exact Slide Requirements

### Phase I Presentation (Upload Only)

| Slide # | Topic | Count |
|---------|-------|-------|
| 0 | Title slide | Optional |
| 1 | Approach slide | Exactly 1 |
| 2-3 | Preprocessing slides | Exactly 2 |
| 4-5 | Data relationship slides | Exactly 2 |
| 6-7 | Data cleaning / imputation slides | Exactly 2 |
| 8 | Most important findings / discussion | Exactly 1 |

**Total:** 8 required slides (+ optional title = 9 max)

**Tips for each slide:**
- **Approach slide:** Describe your overall strategy. How did you decide what to do first? What tools did you use? How did you handle the multi-file, multi-format situation?
- **Preprocessing slides (2):** Show what you did to load, merge, and transform the raw data. Include specifics: unit conversions, Roman numeral decoding, column renaming, etc.
- **Data relationship slides (2):** Correlations, scatter plots, distributions. Which features relate to each other? Are there clusters? Use visualizations.
- **Data cleaning slides (2):** How did you handle missing values? What imputation methods did you use? How did you detect and handle outliers? Show before/after.
- **Findings slide:** What are the most interesting or surprising things you discovered? What would you want to investigate further in Phase II?

### Phase II Presentation (Live, Max 10 Minutes)

| Slide # | Topic | Count |
|---------|-------|-------|
| 0 | Title slide | Optional |
| 1 | Feature selection slide | Exactly 1 |
| 2 | Method slide (splitting, approaches) | Exactly 1 |
| 3-4 | Evaluation slides | 1 or 2 |
| 5 | Discussion slide | Exactly 1 |

**Total:** 4-5 required slides (+ optional title)

**Important:** You have a hard 10-minute limit for the live presentation. Practice timing.

---

## 4. Deadlines

| Phase | Deadline | What to Submit | Presentation |
|-------|----------|----------------|-------------|
| Phase I | **2026-05-03** | Slides + code via TeachCenter | Upload only |
| Phase II | **2026-06-19** | Slides + code via TeachCenter | Live (10 min max) |

**Time available from today (2026-04-27):**
- Phase I: **6 days remaining** -- this is urgent
- Phase II: ~53 days remaining (but depends on Phase I being done)

---

## 5. Dataset Overview

You have **7 data files** across **5 different formats** (CSV, TSV, JSON, XLSX, PKL). All files describe the same pool of **10,000 ice hockey players**.

### File-by-File Breakdown

#### `data/performance.tsv` (1.7 MB)
- **Format:** Tab-separated values
- **Rows:** 10,000 | **Columns:** 29
- **Join key:** `international_id` (integer, 1-10000, all unique)
- **Contains:** All performance/game statistics (goals, assists, shots, penalties, etc.)
- **Why it matters:** This is likely the most predictive data. On-ice performance stats are the core of any scouting evaluation.
- **Watch out for:** Column named `penality_minutes` (typo -- should be `penalty_minutes`). 37 rows have ALL columns null (systematic missing data).

#### `data/contracts_competition.pkl` (501 KB)
- **Format:** Python pickle (pandas DataFrame)
- **Rows:** 10,000 | **Columns:** 8
- **Join key:** `international_id` (integer, all 10,000 unique)
- **Contains:** Contract and career info (salary, captain status, championships, draft year, etc.)
- **Why it matters:** Career trajectory and contract history reflect a player's proven value.
- **Watch out for:** `captain` and `won_championship` are stored as object type (mixed True/False/NaN), not clean booleans. 37 rows systematically missing. `contracts_signed` has 122 missing, `number_of_previous_teams` has 206 missing (more than the baseline 37).

#### `data/hockey_scouting_notes.csv` (866 KB)
- **Format:** Comma-separated values
- **Rows:** 10,000 | **Columns:** 7
- **Join key:** `international_id` (integer, all 10,000 unique)
- **Contains:** Position, dominant hand, experience level, years played/pro, scout notes
- **Why it matters:** Position and experience context; scout notes are free-text evaluations.
- **Watch out for:** `dominant_hand` has 651 missing values (not just the baseline 37). `scout_notes` has 1,038 missing. These are text fields that need NLP or encoding.

#### `data/identity_card_0.tsv` (83 KB)
- **Format:** Tab-separated values
- **Rows:** 888 | **Columns:** 8
- **Join key:** `numerus_internationalis_ad_identitatem` (ROMAN NUMERALS -- must be converted)
- **Secondary key:** `numerus_identificationis_ad_medicinam` (UUID, links to medical data)
- **Contains:** Player identity: name, gender, age, birth city, nationality
- **Why it matters:** This is the smaller split of identity data. Combined with `identity_card_1.csv`, it covers all 10,000 players.
- **CRITICAL WATCH OUT:**
  - Column names are in **Latin** (e.g., `praenomen` = first_name, `cognomen` = last_name)
  - International IDs are **Roman numerals** (e.g., `CCLXXV` = 275, `MMMMMMMDCCVIII` = 7708)
  - Last names have Latin suffixes (e.g., `Alleeius`, `Kohlius` -- need to strip `-ius`/`-us`)
  - First names are sometimes Latinized (e.g., `Keithus`, `Donaldus`)
  - Gender values are inconsistent (`male`, `female`, `m`, `f`, `other`)
  - Nationality has inconsistent casing (`austria`, `AUSTRIA`, `Austria`, `fRANCE`)

#### `data/identity_card_1.csv` (743 KB)
- **Format:** Comma-separated values
- **Rows:** 9,112 | **Columns:** 8
- **Join key:** `international_id` (integer)
- **Secondary key:** `medical_id` (UUID, links to medical data)
- **Contains:** Same structure as identity_card_0 but with normal English column names
- **Why it matters:** The larger split of identity data. Together with identity_card_0, covers all 10,000 players with zero overlap.
- **Watch out for:** Gender values are still inconsistent (`male`, `female`, `m`, `f`, `other`). 37 rows have systematic nulls across all columns.

#### `data/medical_information.xlsx` (917 KB)
- **Format:** Excel spreadsheet with non-standard header layout
- **Rows:** 10,000 | **Columns:** 11
- **Join key:** `medical_id` (UUID -- links through identity cards, NOT directly to international_id)
- **Contains:** Physical measurements, fitness data, medical conditions, return dates
- **Why it matters:** Physical attributes and fitness metrics are relevant to scouting.
- **CRITICAL WATCH OUT:**
  - The Excel file has a **messy header**: Row 0 is metadata/date, Row 1 has column names, Row 2 has `medical_id` label. You need `skiprows=3` or manual header handling.
  - Height values have **mixed units**: some are `1.79m` (meters with suffix), others are `180.1135cm` (centimeters with suffix). You must parse and normalize.
  - Weight values have **mixed formats**: some include `kg` suffix, others are bare numbers.
  - `body_fat_percentage` has mixed formats: some include `%` suffix, others are bare numbers.
  - `fitness_level` has **inconsistent casing**: `Average`, `average`, `good`, `Good`, `excellent`, `Excellent`, `elite`, `Elite`, `poor`, `Poor`. Also contains the string `Nan` (not a real NaN -- it is the literal text "Nan").
  - `medical_information` is 85% null (8,518 out of 10,000 missing). Non-null values are injury descriptions like "bruises", "lower back pain", "hip injury".
  - `return_date` has **mixed date formats**: `2025-01-12`, `2025/01/05`, `01-02-2025`.
  - `physician_signature` contains joke values: `Dr. sth., MD`, `unintelligible handwriting`, `Doctor Who, MD`, `Dr. House`, `Doctor Strange`. This is NOT a real feature.

#### `data/moms_notes.json` (4.4 MB)
- **Format:** JSON (array of 10,000 objects)
- **Rows:** 10,000 | **Columns:** 13
- **Join key:** **NONE** -- no ID column! Must join by `first_name` + `last_name` (or similar fuzzy matching through identity cards)
- **Contains:** Personal/biographical info including many distractor features
- **Why it matters:** Contains `school_grade` and `years_in_usa` which could be real features. Most other fields are intentional noise.
- **CRITICAL WATCH OUT:**
  - **No direct ID column** -- this is the hardest file to join. You must match by name through the identity card files.
  - `favourite_tv_show` and `favourite_food` are **100% empty strings** (all 10,000 records). These columns are completely useless.
  - `personal_notes` has 2,527 missing values.
  - `favourite_board_game` has 431 missing values.
  - `eye_color` has 297 missing values.
  - Many fields here are **intentional distractors** designed to test whether you can identify irrelevant features (see Section 7).

### Join Key Architecture

The files connect like this:

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

**Join strategy:**
1. Convert identity_card_0 Roman numeral IDs to integers
2. Rename identity_card_0 Latin columns to English
3. Concatenate identity_card_0 + identity_card_1 to get full 10,000-row identity table
4. Join performance, contracts, scouting onto identity via `international_id`
5. Join medical onto identity via `medical_id`
6. Join moms_notes via name matching (be careful with Latinized names in identity_card_0)

---

## 6. Features Grouped by Category

### Group 1: Performance / Game Statistics (28 features)
**Source:** `data/performance.tsv`

These are on-ice performance metrics. Likely the most important features for prediction.

| Feature | Description | Notes |
|---------|------------|-------|
| `goals` | Number of goals scored | |
| `assists` | Number of assists | |
| `num_of_shots` | Number of shots taken | |
| `shot_speed` | Speed of shot (km/h) | |
| `shot_attempts` | Number of shot attempts | |
| `shooting_percentage` | Goals / shots ratio | Check if consistent with goals/num_of_shots |
| `high_danger_shots` | Shots from close/central areas | |
| `medium_danger_shots` | Shots from moderate scoring areas | |
| `low_danger_shots` | Shots from far away / poor angles | |
| `save_percentage` | Goalie stat: shots stopped / total | Goalie-specific |
| `winning_goals` | Goals that gave team the winning lead | |
| `power_play_time` | Time on ice during power plays | |
| `power_play_goals` | Goals scored during power plays | |
| `time_on_ice` | Total time spent on ice | |
| `faceoff_win_percentage` | Faceoff wins / total faceoffs | Center-specific mostly |
| `puck_touches` | Number of puck contacts | |
| `puck_recoveries` | Times player regained loose puck | |
| `puck_possession_time` | Total time controlling puck | |
| `penalty_kill_time` | Time on ice while team is shorthanded | |
| `penality_minutes` | Minutes in penalty box | **TYPO in column name** |
| `penalties_taken` | Number of infractions committed | |
| `time_between_penalties` | Average time between penalties | |
| `goals_against_total` | Goals allowed (goalie stat) | Goalie-specific |
| `goals_against_average` | Avg goals allowed per game (goalie) | Goalie-specific |
| `passes_attempted` | Total pass attempts | |
| `passes_completed` | Successful passes | |
| `pass_completion_rate` | Completed / attempted passes | Check consistency with above two |
| `games_missed_due_to_injury` | Games missed from injury | |

### Group 2: Personal / Biographical (13 features)
**Source:** `data/moms_notes.json`

Most of these are **intentional distractors**. The instructors put them there to test whether you can identify irrelevant features.

| Feature | Description | Likely Relevant? | Notes |
|---------|------------|-----------------|-------|
| `first_name` | First name | No (identifier) | Needed for joining only |
| `last_name` | Last name | No (identifier) | Needed for joining only |
| `age` | Player age | **Maybe** | Duplicated as `age_in_years` in medical data |
| `birth_city` | Birth city | Unlikely | |
| `school_grade` | Average school grade | Possibly | Could correlate with discipline |
| `years_in_usa` | Years spent in USA | Possibly | Could relate to training background |
| `eye_color` | Eye color | **No** | Obvious distractor |
| `stuffed_animal_name` | Favorite stuffed animal | **No** | Obvious distractor |
| `favourite_board_game` | Favorite board game | **No** | Obvious distractor |
| `personal_notes` | Notes about player | Maybe | Free text, needs NLP |
| `favourite_child_info` | Info about favorite child | **No** | Obvious distractor |
| `favourite_tv_show` | Favorite TV show | **No** | 100% empty strings |
| `favourite_food` | Favorite food | **No** | 100% empty strings |

### Group 3: Physical / Medical (10 features)
**Source:** `data/medical_information.xlsx`

Physical measurements and fitness data. Relevant for scouting.

| Feature | Description | Notes |
|---------|------------|-------|
| `height` | Player height | Mixed units: meters and centimeters |
| `weight` | Player weight | Mixed formats: with/without `kg` suffix |
| `age_in_years` | Player age | **Duplicate of `age` in moms_notes** |
| `shoe_size` | Shoe size | Marginal relevance at best |
| `body_fat_percentage` | Body fat % | Mixed formats: with/without `%` suffix |
| `fitness_level` | Fitness category | Inconsistent casing + string "Nan" |
| `sprint_time` | Time to reach top speed | Relevant for speed assessment |
| `medical_information` | Medical conditions/injuries | 85% null. Free text when present |
| `return_date` | Clearance return date | Mixed date formats |
| `physician_signature` | Doctor's signature | **NOT a feature** -- drop it |

### Group 4: Role / Experience (6 features)
**Source:** `data/hockey_scouting_notes.csv`

Position and experience context. Important for understanding player roles.

| Feature | Description | Notes |
|---------|------------|-------|
| `position` | Playing position | Categorical (defense, right wing, etc.) |
| `dominant_hand` | Left/right hand | 651 missing values (6.5%) |
| `experience_level` | Experience category | Categorical (veteran, etc.) |
| `years_played` | Total years playing | |
| `years_pro` | Years with pro contract | |
| `scout_notes` | Free-text scout evaluation | 1,038 missing (10.4%) |

### Group 5: Contract / Career (7 features)
**Source:** `data/contracts_competition.pkl`

Career and financial information.

| Feature | Description | Notes |
|---------|------------|-------|
| `contracts_signed` | Number of contracts | 122 missing values |
| `salary` | Current salary | |
| `captain` | Whether player is captain | Boolean as object type |
| `won_championship` | Whether player won championship | Boolean as object type |
| `jersey_number` | Preferred jersey number | Probably not predictive |
| `draft_year` | Year player was drafted | |
| `number_of_previous_teams` | Teams played for before current | 206 missing values |

---

## 7. Key Things to Watch Out For

### 7.1 The Data is Synthetic with Intentional Traps

The instructors explicitly state they generated this data and "incorporated many of the characteristics of various real-world datasets." This means:

- **Intentional outliers** are embedded in the data. You should still detect and handle them as taught in lectures, but some "outliers" may be intentional data points.
- **Distractor features** (eye_color, stuffed_animal_name, favourite_board_game, etc.) are included to test whether you can identify and drop irrelevant variables.
- **Data quality issues** are intentional teaching moments, not accidents.

### 7.2 Multiple File Formats Require Different Loading Strategies

| File | Format | Loading Method |
|------|--------|---------------|
| `performance.tsv` | TSV | `pd.read_csv(..., sep='\t')` |
| `hockey_scouting_notes.csv` | CSV | `pd.read_csv(...)` |
| `identity_card_0.tsv` | TSV | `pd.read_csv(..., sep='\t')` + Roman numeral conversion |
| `identity_card_1.csv` | CSV | `pd.read_csv(...)` |
| `medical_information.xlsx` | Excel | `pd.read_excel(..., header=None, skiprows=3)` |
| `moms_notes.json` | JSON | `json.load()` then `pd.DataFrame()` |
| `contracts_competition.pkl` | Pickle | `pd.read_pickle(...)` |

### 7.3 The Join/Merge Challenge

This is NOT a single-table dataset. You must merge 7 files into one unified DataFrame. The join architecture has three levels of difficulty:

1. **Easy joins:** `performance.tsv`, `contracts_competition.pkl`, and `hockey_scouting_notes.csv` all share `international_id` as an integer key.
2. **Medium joins:** `identity_card_1.csv` also has `international_id` (integer) and `medical_id` (UUID). `medical_information.xlsx` joins through `medical_id`.
3. **Hard joins:**
   - `identity_card_0.tsv` has `international_id` as **Roman numerals** in a Latin-named column. Must decode Roman numerals to integers.
   - `moms_notes.json` has **no ID at all**. Must join by name matching through identity cards.

### 7.4 The Systematic 37 Missing Values

Exactly **37 rows** are missing across nearly every column in every file. This strongly suggests that 37 specific players have systematically incomplete records across all data sources. This is likely an intentional pattern for you to discover and discuss.

### 7.5 Duplicate / Redundant Features

- `age` (in moms_notes.json) vs. `age_in_years` (in medical_information.xlsx) -- these should represent the same thing. Compare them after joining. If they differ, decide which to keep and document why.
- `pass_completion_rate` should theoretically equal `passes_completed / passes_attempted`. Verify this. Discrepancies indicate data quality issues.
- `shooting_percentage` should relate to `goals / num_of_shots` or `goals / shot_attempts`. Verify this relationship.

### 7.6 Features That Should Definitely Be Dropped

These features are almost certainly not predictive and should be dropped:

- `eye_color` -- no plausible connection to hockey performance
- `stuffed_animal_name` -- obvious noise
- `favourite_board_game` -- obvious noise
- `favourite_child_info` -- obvious noise (and likely PII-like)
- `favourite_tv_show` -- 100% empty strings
- `favourite_food` -- 100% empty strings
- `physician_signature` -- contains joke values (`Doctor Strange`, `Dr. House`, etc.)
- `first_name`, `last_name` -- identifiers, not features (keep only for joining)

### 7.7 The identity_card_0 Latin Encoding Problem

The smaller identity card file (888 rows) uses:
- **Latin column names:** `praenomen` (first_name), `cognomen` (last_name), `numerus_internationalis_ad_identitatem` (international_id), etc.
- **Roman numeral IDs:** `CCLXXV` = 275, `MMDLX` = 2560
- **Latinized names:** `Keithus`, `Donaldus`, `Alleeius`, `Kohlius`

You need to:
1. Write (or find) a Roman numeral to integer converter
2. Rename all columns to match identity_card_1
3. Strip Latin suffixes from names before joining with moms_notes

### 7.8 Inconsistent Value Encoding

Examples of values that represent the same thing but are encoded differently:

| Field | Inconsistencies |
|-------|----------------|
| `gender` / `sexus` | `male`, `m`, `female`, `f`, `other` |
| `nationality` / `natio` | `austria`, `Austria`, `AUSTRIA`, `fRANCE`, `sWITZERLAND` |
| `fitness_level` | `Average`, `average`, `Good`, `good`, `Elite`, `elite`, `Nan` (string) |
| `height` | `1.79m` vs `180.1135cm` |
| `weight` | `82.4772kg` vs `93.8985` (no unit) |
| `body_fat_percentage` | `20.87%` vs `23.38` (no symbol) |
| `return_date` | `2025-01-12` vs `2025/01/05` vs `01-02-2025` |

### 7.9 Goalie-Specific Features

Some stats only make sense for goalies:
- `save_percentage` -- a goalie metric
- `goals_against_total` -- a goalie metric
- `goals_against_average` -- a goalie metric

For non-goalies, these will likely be 0 or NaN. Keep this in mind when analyzing distributions -- do not treat goalie zeros as outliers for skaters and vice versa.

### 7.10 No Target Variable in Phase I

The `feature_description.txt` does not include a target variable. This is intentional. Phase I is explicitly about preprocessing and EDA **without** the target. You will receive the target for Phase II. Do not try to guess or create a proxy target.

---

## 8. Data Quality Issues Found During Inspection

This section documents specific issues found by inspecting the actual data files.

### 8.1 Missing Value Summary

| File | Column | Missing Count | % Missing | Notes |
|------|--------|--------------|-----------|-------|
| All files | (systematic) | 37 | 0.37% | Same 37 players missing everywhere |
| `performance.tsv` | All 28 feature columns | 37 each | 0.37% | Systematic |
| `hockey_scouting_notes.csv` | `dominant_hand` | 651 | 6.51% | Beyond systematic 37 |
| `hockey_scouting_notes.csv` | `scout_notes` | 1,038 | 10.38% | Significant chunk |
| `contracts_competition.pkl` | `contracts_signed` | 122 | 1.22% | Beyond systematic 37 |
| `contracts_competition.pkl` | `number_of_previous_teams` | 206 | 2.06% | Beyond systematic 37 |
| `medical_information.xlsx` | `medical_information` | 8,518 | 85.18% | Most players healthy |
| `medical_information.xlsx` | `fitness_level` | 358 | 3.58% | Beyond systematic 37 |
| `medical_information.xlsx` | `weight` | 34 | 0.34% | Slightly less than 37 |
| `medical_information.xlsx` | `body_fat_percentage` | 30 | 0.30% | Slightly less than 37 |
| `moms_notes.json` | `personal_notes` | 2,527 | 25.27% | Quarter of data missing |
| `moms_notes.json` | `favourite_board_game` | 431 | 4.31% | |
| `moms_notes.json` | `eye_color` | 297 | 2.97% | Feature should be dropped anyway |
| `moms_notes.json` | `favourite_tv_show` | 0 (null) | 0% | But 100% empty strings |
| `moms_notes.json` | `favourite_food` | 0 (null) | 0% | But 100% empty strings |
| `identity_card_1.csv` | All except IDs | 37 each | 0.41% | Systematic |
| `identity_card_0.tsv` | (none) | 0 | 0% | Clean |

### 8.2 Type and Format Issues

| File | Column | Issue |
|------|--------|-------|
| `identity_card_0.tsv` | ID column | Roman numerals instead of integers |
| `identity_card_0.tsv` | All column names | Latin instead of English |
| `identity_card_0.tsv` | `praenomen`, `cognomen` | Some names Latinized |
| `medical_information.xlsx` | Header rows | Non-standard layout, needs skiprows=3 |
| `medical_information.xlsx` | `height` | Mixed meters/centimeters with unit suffixes |
| `medical_information.xlsx` | `weight` | Mixed with/without `kg` suffix |
| `medical_information.xlsx` | `body_fat_percentage` | Mixed with/without `%` suffix |
| `medical_information.xlsx` | `fitness_level` | Inconsistent casing, string `Nan` |
| `medical_information.xlsx` | `return_date` | Three different date formats |
| `contracts_competition.pkl` | `captain`, `won_championship` | Object dtype, not boolean |
| `performance.tsv` | `penality_minutes` | Typo in column name |

### 8.3 Potential Distractor / Noise Columns

These columns are almost certainly included as traps to test feature selection:

1. `eye_color` -- no hockey relevance
2. `stuffed_animal_name` -- no hockey relevance
3. `favourite_board_game` -- no hockey relevance
4. `favourite_child_info` -- no hockey relevance, PII-adjacent
5. `favourite_tv_show` -- 100% empty
6. `favourite_food` -- 100% empty
7. `physician_signature` -- joke values, identifier not feature
8. `jersey_number` -- superstition, not performance
9. `shoe_size` -- marginal at best, probably correlates with height

---

## 9. Suggested Development Roadmap for Phase I

Use this as a checklist. Each step builds on the previous one.

### Week 1 (Now through 2026-05-01): Data Loading and Integration

- [ ] **Step 1: Set up your environment**
  - Create a Jupyter notebook or Python script
  - Install required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl` (for xlsx), `json` (built-in)
  - Set up a `src/` or `notebooks/` directory for your code

- [ ] **Step 2: Load each file individually and inspect**
  - Load each of the 7 files with the correct reader
  - Print `.shape`, `.dtypes`, `.head()`, `.describe()`, `.isnull().sum()` for each
  - Document what you find (this feeds directly into your presentation)

- [ ] **Step 3: Handle identity_card_0 special formatting**
  - Write a Roman numeral to integer converter (or use a library like `roman`)
  - Rename Latin columns to match identity_card_1 column names:
    - `numerus_internationalis_ad_identitatem` -> `international_id`
    - `numerus_identificationis_ad_medicinam` -> `medical_id`
    - `praenomen` -> `first_name`
    - `cognomen` -> `last_name`
    - `sexus` -> `gender`
    - `aetas_annorum` -> `age`
    - `urbs_natalis` -> `birth_city`
    - `natio` -> `nationality`
  - Handle Latinized names (strip `-ius`/`-us` suffixes where appropriate)

- [ ] **Step 4: Handle medical_information.xlsx header**
  - Load with `pd.read_excel(..., header=None, skiprows=3)`
  - Manually assign column names: `['medical_id', 'height', 'weight', 'age_in_years', 'shoe_size', 'body_fat_percentage', 'fitness_level', 'sprint_time', 'medical_information', 'return_date', 'physician_signature']`

- [ ] **Step 5: Merge identity card files**
  - Concatenate the cleaned identity_card_0 (888 rows) with identity_card_1 (9,112 rows)
  - Verify you get exactly 10,000 rows with no duplicate `international_id` values

- [ ] **Step 6: Build the master DataFrame**
  - Start with identity table (10,000 rows)
  - Left join `performance.tsv` on `international_id`
  - Left join `contracts_competition.pkl` on `international_id`
  - Left join `hockey_scouting_notes.csv` on `international_id`
  - Left join `medical_information.xlsx` on `medical_id`
  - Join `moms_notes.json` by matching `first_name` + `last_name` through identity cards
  - Verify final shape is 10,000 rows

### Week 1 (continued): Data Cleaning

- [ ] **Step 7: Standardize inconsistent values**
  - Normalize `gender`: map all to `male`, `female`, `other`
  - Normalize `nationality`: lowercase all, fix typos
  - Normalize `fitness_level`: lowercase all, handle string `Nan` -> real NaN
  - Parse `height`: extract numeric value, convert all to same unit (cm or m)
  - Parse `weight`: strip `kg` suffix where present, ensure all numeric
  - Parse `body_fat_percentage`: strip `%` suffix where present
  - Parse `return_date`: handle all three date formats, convert to datetime
  - Fix `penality_minutes` column name typo -> `penalty_minutes`

- [ ] **Step 8: Handle missing values**
  - Investigate the 37 systematically missing rows -- are these the same players?
  - Decide imputation strategy for each column based on its type:
    - Numeric: mean, median, or mode imputation (or leave as NaN if appropriate)
    - Categorical: mode imputation or "Unknown" category
    - Text: leave as NaN or fill with empty string
  - Document your reasoning (this goes on slides 6-7)

- [ ] **Step 9: Drop irrelevant features**
  - Drop obvious distractors: `eye_color`, `stuffed_animal_name`, `favourite_board_game`, `favourite_child_info`, `favourite_tv_show`, `favourite_food`
  - Drop identifiers: `first_name`, `last_name`, `physician_signature`
  - Drop 100%-empty columns
  - Document which columns you dropped and why

- [ ] **Step 10: Detect and handle outliers**
  - Use box plots, z-scores, or IQR method as taught in lectures
  - Be aware that some outliers are intentional (synthetic data)
  - Consider position-specific analysis (goalie stats vs skater stats)
  - Document outliers found and your handling strategy

### Week 1 Final Push (2026-05-01 to 2026-05-03): EDA and Presentation

- [ ] **Step 11: Exploratory Data Analysis**
  - Compute correlation matrix for numeric features (heatmap)
  - Create distribution plots for key features
  - Explore relationships between features (scatter plots, pair plots)
  - Analyze patterns by position (goalies vs defenders vs forwards)
  - Check the `age` vs `age_in_years` discrepancy
  - Verify computed features (shooting_percentage, pass_completion_rate)
  - Look for interesting clusters or groupings

- [ ] **Step 12: Build the Phase I presentation**
  - Slide 1 (Approach): Your strategy, tools, how you handled the multi-file challenge
  - Slides 2-3 (Preprocessing): File loading, merging, format conversions, type fixes
  - Slides 4-5 (Data Relationships): Correlations, distributions, interesting patterns
  - Slides 6-7 (Cleaning/Imputation): Missing values strategy, outlier handling
  - Slide 8 (Findings): Most surprising discoveries, what you would investigate in Phase II

- [ ] **Step 13: Package and submit**
  - Clean up your code (remove dead code, add comments)
  - Make sure the code runs end-to-end from raw data to clean DataFrame
  - Upload presentation AND code to TeachCenter before 2026-05-03

---

## Appendix A: Quick Reference -- Column Name Mapping for identity_card_0

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

## Appendix B: Quick Reference -- File Loading Cheat Sheet

```python
import pandas as pd
import json

# Performance stats (easy)
performance = pd.read_csv('data/performance.tsv', sep='\t')

# Contracts (easy)
contracts = pd.read_pickle('data/contracts_competition.pkl')

# Scouting notes (easy)
scouting = pd.read_csv('data/hockey_scouting_notes.csv')

# Identity card 1 (easy)
identity_1 = pd.read_csv('data/identity_card_1.csv')

# Identity card 0 (needs work)
identity_0 = pd.read_csv('data/identity_card_0.tsv', sep='\t')
# Then: rename columns, convert Roman numerals, clean names

# Medical (needs work)
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

# Mom's notes (needs work)
with open('data/moms_notes.json', 'r', encoding='utf-8') as f:
    moms_data = json.load(f)
moms_notes = pd.DataFrame(moms_data)
# Then: join by name matching -- no ID column!
```

## Appendix C: Summary Statistics at a Glance

| File | Rows | Columns | Join Key | Format Difficulty |
|------|------|---------|----------|-------------------|
| `performance.tsv` | 10,000 | 29 | `international_id` (int) | Easy |
| `contracts_competition.pkl` | 10,000 | 8 | `international_id` (int) | Easy |
| `hockey_scouting_notes.csv` | 10,000 | 7 | `international_id` (int) | Easy |
| `identity_card_1.csv` | 9,112 | 8 | `international_id` + `medical_id` | Easy |
| `identity_card_0.tsv` | 888 | 8 | Roman numeral + UUID | Hard |
| `medical_information.xlsx` | 10,000 | 11 | `medical_id` (UUID) | Medium |
| `moms_notes.json` | 10,000 | 13 | None (name matching) | Hard |
| **Total unique features** | | **~67** | | |
| **Total players** | | **10,000** | | |

---

*This document was generated on 2026-04-27 based on inspection of the actual project files and the project brief. Refer back to it throughout the project as your single source of truth for what needs to be done and what to watch out for.*
