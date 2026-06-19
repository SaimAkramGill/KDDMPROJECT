---
name: kddm-hockey-data
description: Activates when working with the KDDM ice hockey scouting dataset (7 source files in data/, merging, cleaning, imputation, EDA, Phase I presentation). Provides loading recipes, the Roman-numeral/Latin column map, the 37-row systematic missing pattern, mixed-unit parsing rules, and the distractor feature list.
---

# KDDM Hockey Data Skill

Quick-reference context for the KDDM Ice Hockey Scouting project. Full brief lives in `docs/PROJECT_UNDERSTANDING.md` and `CLAUDE.md`.

## When this matters

- Loading any of the 7 files in `data/`
- Merging into a 10,000-row master DataFrame
- Cleaning mixed units, casing, date formats, Latin/Roman numerals
- Handling the systematic 37-row missing pattern
- Imputation, outlier detection, EDA
- Building Phase I slides (deadline 2026-05-03)
- Working with `dataCleaning.ipynb`, `dataClean_Manual.ipynb`, or `src/build_presentation.py`

## The 7 files (one-line each)

1. `data/performance.tsv` — 10,000×29, join on `international_id`. **Has typo `penality_minutes`.**
2. `data/contracts_competition.pkl` — 10,000×8, join on `international_id`. `captain`/`won_championship` object dtype.
3. `data/hockey_scouting_notes.csv` — 10,000×7, join on `international_id`. `dominant_hand` 651 missing, `scout_notes` 1,038 missing.
4. `data/identity_card_1.csv` — 9,112×8, join on `international_id` + `medical_id`. English names.
5. `data/identity_card_0.tsv` — 888×8. **Latin column names, Roman numeral IDs, Latinized names.**
6. `data/medical_information.xlsx` — 10,000×11. **Needs `skiprows=3`, mixed units everywhere.**
7. `data/moms_notes.json` — 10,000×13. **No ID column — join by name.**

## Loading cheat sheet

```python
import pandas as pd, json

performance = pd.read_csv('data/performance.tsv', sep='\t')
contracts   = pd.read_pickle('data/contracts_competition.pkl')
scouting    = pd.read_csv('data/hockey_scouting_notes.csv')
identity_1  = pd.read_csv('data/identity_card_1.csv')
identity_0  = pd.read_csv('data/identity_card_0.tsv', sep='\t')  # needs rename + Roman conversion

medical = pd.read_excel('data/medical_information.xlsx', header=None, skiprows=3)
medical.columns = [
    'medical_id', 'height', 'weight', 'age_in_years', 'shoe_size',
    'body_fat_percentage', 'fitness_level', 'sprint_time',
    'medical_information', 'return_date', 'physician_signature',
]

with open('data/moms_notes.json', 'r', encoding='utf-8') as f:
    moms_notes = pd.DataFrame(json.load(f))
```

## Latin → English column map (identity_card_0)

| Latin | English |
|-------|---------|
| `numerus_internationalis_ad_identitatem` | `international_id` |
| `numerus_identificationis_ad_medicinam` | `medical_id` |
| `praenomen` | `first_name` |
| `cognomen` | `last_name` |
| `sexus` | `gender` |
| `aetas_annorum` | `age` |
| `urbs_natalis` | `birth_city` |
| `natio` | `nationality` |

Roman numeral examples: `CCLXXV`=275, `MMDLX`=2560. Latin name suffix to strip: `-ius`/`-us` (e.g., `Keithus`→`Keith`).

## Join order

1. Convert Roman IDs → int in `identity_card_0`, rename Latin columns.
2. Concat both identity cards → 10,000 unique `international_id`.
3. Left-join `performance`, `contracts`, `scouting` on `international_id`.
4. Left-join `medical` on `medical_id`.
5. Join `moms_notes` by name (strip Latin suffixes first).
6. Verify final shape is **(10000, ~67)**.

## Mandatory cleanup checklist

- [ ] Rename `penality_minutes` → `penalty_minutes`
- [ ] `captain`, `won_championship` → nullable boolean
- [ ] `height`: parse `1.79m` / `180.1135cm` → single unit (cm or m)
- [ ] `weight`: strip `kg`, ensure float
- [ ] `body_fat_percentage`: strip `%`, ensure float
- [ ] `return_date`: handle `YYYY-MM-DD`, `YYYY/MM/DD`, `DD-MM-YYYY` → datetime
- [ ] `gender`: `m`/`male`, `f`/`female`, `other` → 3 canonical values
- [ ] `nationality`: lowercase, fix casing typos (`fRANCE`, `sWITZERLAND`)
- [ ] `fitness_level`: lowercase, literal `"Nan"` → real `NaN`

## DROP list (distractors)

`eye_color`, `stuffed_animal_name`, `favourite_board_game`, `favourite_child_info`, `favourite_tv_show` (100% empty), `favourite_food` (100% empty), `physician_signature` (joke values), and `first_name`/`last_name` after joining.

## Watch-outs

- **37 systematic missing rows** across nearly every column in every file — likely the same players. Investigate; don't silently drop.
- **Goalie-only features:** `save_percentage`, `goals_against_total`, `goals_against_average`. Segment by `position` before treating zeros as outliers.
- **Duplicate features:** `age` (moms_notes) vs `age_in_years` (medical) — compare; pick one.
- **Verify computed features:** `pass_completion_rate ≈ passes_completed / passes_attempted`; `shooting_percentage` vs `goals/num_of_shots`.
- **Intentional outliers** exist — document handling, don't silently remove.
- **Phase I has NO target variable.** Do not fabricate one.

## Phase I slide structure (must match exactly)

| Slide | Topic | Count |
|-------|-------|-------|
| 0 | Title | Optional |
| 1 | Approach | Exactly 1 |
| 2-3 | Preprocessing | Exactly 2 |
| 4-5 | Data relationships | Exactly 2 |
| 6-7 | Cleaning / imputation | Exactly 2 |
| 8 | Findings | Exactly 1 |

## Working conventions

- Save intermediate outputs under `data/newdata/` (`master_merged.csv` is established).
- Notebooks: restart-and-run-all reproducibility required. Clear large outputs before commit.
- Helper Python lives under `src/`. Presentation builder: `src/build_presentation.py`.
- Figures: `docs/figures/`.

## When delegating heavy cleaning work

Use the `kddm-data-cleaner` subagent (in `.claude/agents/`) — it has the full project system prompt baked in.
