"""Build Phase1_DataCleaning.ipynb in a beginner / student style.

Run from the project root:  python src/_build_phase1_notebook.py
Output: Phase1_DataCleaning.ipynb
"""
import json
import os

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def co(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    })


# =============================================================== TITLE
md("""\
# KDDM Phase 1 — Ice Hockey Talent Scouting

**Group 174**

| Student | Matriculation Number |
| --- | --- |
| Muhammad Saim Akram | 12344542 |
| Zamarak Sufyan |  |

This notebook does the Phase 1 part of the project: loading the 7 data files, cleaning them up, joining them into one big table, and making some plots.

The dataset has information about 10,000 ice hockey players. The goal of Phase 1 is just to clean the data — we don't have the answer column yet (that comes in Phase 2). The data is split across 7 different files in 5 different formats, and many of the columns are messy on purpose, so we have to figure out how to fix all of that.
""")

# =============================================================== STEP 1
md("""\
## Step 1: Importing the libraries we need

We are going to use:

- **pandas** to load and work with the tables
- **numpy** for some math things like NaN
- **matplotlib** for making the plots
- **os** so we can build file paths

We also import `display` from IPython so that tables print nicely in the notebook.
""")

co("""\
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# This makes pandas show all columns when we print a table
pd.set_option("display.max_columns", None)

# Where the data files live, and where to save things
data_folder = "data"
output_folder = "data/newdata"
figures_folder = "docs/figures"

# Make sure the output folders exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

print("Libraries loaded. Data folder:", data_folder)
""")

# =============================================================== STEP 2
md("""\
## Step 2: Loading all 7 data files

Each file has a different format, so we need a different pandas function for each one:

- `.tsv` files are tab-separated, so we use `pd.read_csv` with `sep="\\t"`
- `.csv` files are comma-separated, so we use plain `pd.read_csv`
- `.pkl` files are pandas pickle files, so we use `pd.read_pickle`
- `.xlsx` files are Excel files, so we use `pd.read_excel`
- `.json` files are JSON, so we use `pd.read_json`
""")

co("""\
performance = pd.read_csv("data/performance.tsv", sep="\\t")
contracts = pd.read_pickle("data/contracts_competition.pkl")
scouting = pd.read_csv("data/hockey_scouting_notes.csv")
identity_1 = pd.read_csv("data/identity_card_1.csv")
identity_0 = pd.read_csv("data/identity_card_0.tsv", sep="\\t")
medical = pd.read_excel("data/medical_information.xlsx")
moms_notes = pd.read_json("data/moms_notes.json")

print("All 7 files loaded.")
""")

# =============================================================== STEP 3
md("""\
## Step 3: Taking a first look at each file

Before we change anything, we want to see what we are working with. For each file we will:

1. Print its shape (how many rows and columns)
2. Show the first few rows with `.head()`
3. Check how many missing values it has with `.isnull().sum()`

This way we know what problems we need to fix later.
""")

md("### Performance file")
co("""\
print("Shape:", performance.shape)
display(performance.head())
print("\\nMissing values per column:")
print(performance.isnull().sum())
""")

md("### Contracts file")
co("""\
print("Shape:", contracts.shape)
display(contracts.head())
print("\\nMissing values per column:")
print(contracts.isnull().sum())
""")

md("### Identity card 1 (the normal one)")
co("""\
print("Shape:", identity_1.shape)
display(identity_1.head())
print("\\nMissing values per column:")
print(identity_1.isnull().sum())
""")

md("### Identity card 0 (the weird one with Latin words)")
co("""\
print("Shape:", identity_0.shape)
display(identity_0.head())
print("\\nThe column names are in Latin and the ID column is in Roman numerals!")
""")

md("### Medical file (the Excel file with weird header)")
co("""\
print("Shape:", medical.shape)
display(medical.head())
print("\\nNotice that the column names look strange - the first row contains the real headers, not the column index.")
""")

md("### Scouting notes file")
co("""\
print("Shape:", scouting.shape)
display(scouting.head())
print("\\nMissing values per column:")
print(scouting.isnull().sum())
""")

md("### Mom's notes file")
co("""\
print("Shape:", moms_notes.shape)
display(moms_notes.head())
print("\\nNote: this file does NOT have an international_id column, so we will need a different way to join it later.")
""")

md("""\
### What we learned from looking at the files

- `performance.tsv`, `contracts_competition.pkl`, `hockey_scouting_notes.csv` and `identity_card_1.csv` all have a column called `international_id` that we can use to join them.
- `identity_card_0.tsv` is the same kind of data as `identity_card_1`, but the column names are in Latin and the IDs are Roman numerals. We need to fix it.
- `medical_information.xlsx` has a broken header. We need to re-read it.
- `moms_notes.json` has no ID column, so we will have to join by first name and last name.
- Lots of the columns have missing values. We will handle them later.
""")

# =============================================================== STEP 4
md("""\
## Step 4: Fixing identity_card_0 (Latin words + Roman numerals)

The columns are in Latin (`praenomen` instead of `first_name`, etc.), the IDs are Roman numerals like `CCLXXV`, and some of the names have Latin endings (`Keithus` instead of `Keith`). We have to fix all three things.
""")

md("### 4a: Renaming the Latin column names to English")
co("""\
# A dictionary that says what each Latin name should become in English
latin_to_english = {
    "numerus_internationalis_ad_identitatem": "international_id",
    "numerus_identificationis_ad_medicinam": "medical_id",
    "praenomen": "first_name",
    "cognomen": "last_name",
    "sexus": "gender",
    "aetas_annorum": "age",
    "urbs_natalis": "birth_city",
    "natio": "nationality",
}

identity_0 = identity_0.rename(columns=latin_to_english)
print("Columns after renaming:")
print(identity_0.columns.tolist())
""")

md("""\
### 4b: Converting the Roman numerals to normal numbers

For example, `CCLXXV` should become `275`.

The rules of Roman numerals are: I = 1, V = 5, X = 10, L = 50, C = 100, D = 500, M = 1000. When a smaller letter comes BEFORE a bigger one, you subtract it (so IV = 4, IX = 9). Otherwise you add.

We will write a small function that does this, and then apply it to every value in the column.
""")

co("""\
def roman_to_number(roman_string):
    # If the value is missing, return missing
    if pd.isna(roman_string):
        return np.nan

    # The values of each letter
    letter_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    total = 0
    previous_value = 0

    # We read the Roman number from right to left.
    # If the current letter is smaller than the one to its right, we subtract.
    # Otherwise we add.
    for letter in reversed(str(roman_string).strip().upper()):
        if letter not in letter_values:
            return np.nan
        current_value = letter_values[letter]
        if current_value < previous_value:
            total = total - current_value
        else:
            total = total + current_value
        previous_value = current_value

    return total


# Apply the function to every row in the international_id column
identity_0["international_id"] = identity_0["international_id"].apply(roman_to_number)

print("First few converted IDs:")
print(identity_0[["international_id", "first_name", "last_name"]].head())
""")

md("""\
### 4c: Removing the Latin endings from the names

Some names end with `-ius` (like `Kohlius` → `Kohl`) or `-us` (like `Keithus` → `Keith`). We need to remove those endings so the names match the ones in `moms_notes.json`.
""")

co("""\
def strip_latin_ending(name):
    if pd.isna(name):
        return name
    name = str(name)
    # Check the longer ending first ('ius'), then the shorter one ('us')
    if name.endswith("ius"):
        return name[:-3]
    if name.endswith("us"):
        return name[:-2]
    return name


identity_0["first_name"] = identity_0["first_name"].apply(strip_latin_ending)
identity_0["last_name"] = identity_0["last_name"].apply(strip_latin_ending)

print("Names after stripping Latin endings:")
print(identity_0[["first_name", "last_name"]].head(10))
""")

# =============================================================== STEP 5
md("""\
## Step 5: Combining the two identity tables

Now that `identity_0` has the same column names and the same kind of IDs as `identity_1`, we can stack them on top of each other with `pd.concat`. We expect the result to have 10,000 rows (888 + 9,112).
""")

co("""\
identity = pd.concat([identity_0, identity_1], ignore_index=True)

print("Combined identity table shape:", identity.shape)
print("Number of unique international_ids:", identity["international_id"].nunique())
""")

# =============================================================== STEP 6
md("""\
## Step 6: Cleaning the medical Excel file

The Excel file has a strange header layout. The real column names are in row 0, the label `medical_id` sits alone in row 1 column 0, and row 2 is some metadata. So we need to re-read the file with `header=None` and rebuild the column names ourselves.
""")

md("### 6a: Re-reading with a fixed header")
co("""\
# Re-read the medical file WITHOUT pandas guessing the header
medical_raw = pd.read_excel("data/medical_information.xlsx", header=None)

# Look at what's in there
print("First 4 rows of the raw file:")
display(medical_raw.head(4))
""")

co("""\
# Build the real column names: 'medical_id' for the first column,
# and the names from row 1 for the rest
column_names = ["medical_id"] + medical_raw.iloc[1, 1:].tolist()
print("Column names will be:")
print(column_names)

# Now keep only the actual data rows (starting from row 3) and use our column names
medical = medical_raw.iloc[3:].reset_index(drop=True).copy()
medical.columns = column_names

print("\\nMedical table after fixing the header:")
display(medical.head())
""")

md("""\
### 6b: Fixing the height column

The height values are written in two different ways: some have `cm` at the end (like `180.1135cm`) and some have `m` (like `1.79m`). We will convert everything to centimeters.
""")

co("""\
def fix_height(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    # If the text contains 'nan', treat as missing
    if "nan" in text:
        return np.nan
    # If it ends with 'cm', remove the 'cm' and convert to a number
    if text.endswith("cm"):
        return float(text[:-2])
    # If it ends with 'm', remove 'm' and multiply by 100 to get cm
    if text.endswith("m"):
        return float(text[:-1]) * 100
    # Otherwise just convert to a number
    return float(text)


medical["height"] = medical["height"].apply(fix_height)
print("Height after cleaning (in cm):")
print(medical["height"].describe())
""")

md("""\
### 6c: Fixing the weight column

Some weight values have `kg` at the end. We just remove it.
""")

co("""\
def fix_weight(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "nan" in text:
        return np.nan
    if text.endswith("kg"):
        return float(text[:-2])
    return float(text)


medical["weight"] = medical["weight"].apply(fix_weight)
print("Weight after cleaning (in kg):")
print(medical["weight"].describe())
""")

md("""\
### 6d: Fixing the body fat percentage column

Some values have `%` at the end, some don't.
""")

co("""\
def fix_body_fat(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1])
    return float(text)


medical["body_fat_percentage"] = medical["body_fat_percentage"].apply(fix_body_fat)
print("Body fat % after cleaning:")
print(medical["body_fat_percentage"].describe())
""")

md("""\
### 6e: Converting return_date to real dates

The dates come in three different formats. Luckily `pd.to_datetime` can usually figure them out if we set `errors="coerce"` (which turns anything it can't parse into NaT).
""")

co("""\
medical["return_date"] = pd.to_datetime(medical["return_date"], errors="coerce")
print("Some return_date values after conversion:")
print(medical["return_date"].dropna().head(5))
""")

md("""\
### 6f: Fixing the fitness_level column

This column has values like `Average`, `average`, `Good`, `good` (different cases of the same word). It also has the literal string `Nan`, which we want to turn into a real missing value.
""")

co("""\
# Replace the string 'Nan' with a real missing value
medical["fitness_level"] = medical["fitness_level"].replace("Nan", np.nan)

# Lowercase everything so 'Good' and 'good' become the same
medical["fitness_level"] = medical["fitness_level"].str.lower().str.strip()

print("Fitness level after cleaning:")
print(medical["fitness_level"].value_counts(dropna=False))
""")

md("""\
### 6g: Dropping the physician_signature column

This column has joke values like `Dr. House`, `Doctor Strange`, `unintelligible handwriting`. It is not a real feature, so we drop it.
""")

co("""\
medical = medical.drop(columns=["physician_signature"])
print("Medical columns now:")
print(medical.columns.tolist())
""")

co("""\
# Also convert the rest of the numeric columns properly
medical["age_in_years"] = pd.to_numeric(medical["age_in_years"], errors="coerce")
medical["shoe_size"] = pd.to_numeric(medical["shoe_size"], errors="coerce")
medical["sprint_time"] = pd.to_numeric(medical["sprint_time"], errors="coerce")

print("Medical types after cleaning:")
print(medical.dtypes)
""")

# =============================================================== STEP 7
md("""\
## Step 7: Small fixes on the other files
""")

md("""\
### 7a: Fixing the typo in performance.tsv

The column is spelled `penality_minutes` but it should be `penalty_minutes`.
""")

co("""\
performance = performance.rename(columns={"penality_minutes": "penalty_minutes"})
print("performance has penalty_minutes:", "penalty_minutes" in performance.columns)
""")

md("""\
### 7b: Converting captain and won_championship to real booleans

These columns are stored as `object` type with a mix of `True`, `False`, and strings like `"True"`, `"False"`. We map them to real booleans.
""")

co("""\
# Build a small lookup that turns anything True-ish into True and False-ish into False
true_false_map = {True: True, False: False, "True": True, "False": False, "true": True, "false": False}

contracts["captain"] = contracts["captain"].map(true_false_map)
contracts["won_championship"] = contracts["won_championship"].map(true_false_map)

print("captain values:")
print(contracts["captain"].value_counts(dropna=False))
""")

md("""\
### 7c: Fixing implausible draft years

The minimum draft_year in the raw file is 1850, but the NHL draft only started in 1963. Anything before 1960 has to be wrong. We turn those into NaN.
""")

co("""\
print("Before fix: min draft_year =", contracts["draft_year"].min())

# Find the bad rows and replace them with NaN
bad_draft_years = contracts["draft_year"] < 1960
print("Number of bad draft years:", bad_draft_years.sum())
contracts.loc[bad_draft_years, "draft_year"] = np.nan

print("After fix: min draft_year =", contracts["draft_year"].min())
""")

md("""\
### 7d: Lowercasing the category columns in hockey_scouting_notes

We do `.str.lower().str.strip()` so that 'Defense', 'defense', and ' defense' all become the same.
""")

co("""\
scouting["position"] = scouting["position"].str.lower().str.strip()
scouting["dominant_hand"] = scouting["dominant_hand"].str.lower().str.strip()
scouting["experience_level"] = scouting["experience_level"].str.lower().str.strip()

print("Positions:", scouting["position"].unique())
print("Dominant hands:", scouting["dominant_hand"].unique())
""")

md("""\
### 7e: Dropping the empty columns in moms_notes

Some columns in mom's notes are completely empty (100% empty strings or NaN). Let's find them and drop them.
""")

co("""\
# Find columns where every value is the same (or missing)
constant_columns = []
for column_name in moms_notes.columns:
    unique_values = moms_notes[column_name].nunique(dropna=False)
    if unique_values <= 1:
        constant_columns.append(column_name)

print("Constant / empty columns in moms_notes:", constant_columns)

moms_notes = moms_notes.drop(columns=constant_columns)
moms_notes = moms_notes.drop_duplicates().reset_index(drop=True)
print("moms_notes shape after dropping:", moms_notes.shape)
""")

# =============================================================== STEP 8
md("""\
## Step 8: Making the categorical values consistent across files
""")

md("""\
### 8a: Cleaning gender

In some files gender is written as `male`, `m`, `M`. We map them all to either `male`, `female`, or `other`.
""")

co("""\
gender_map = {"m": "male", "male": "male", "f": "female", "female": "female", "other": "other"}

def fix_gender(value):
    if pd.isna(value):
        return np.nan
    return gender_map.get(str(value).strip().lower(), np.nan)

identity_0["gender"] = identity_0["gender"].apply(fix_gender)
identity_1["gender"] = identity_1["gender"].apply(fix_gender)

# Need to redo identity after the fix
identity = pd.concat([identity_0, identity_1], ignore_index=True)

print("Gender values:")
print(identity["gender"].value_counts(dropna=False))
""")

md("""\
### 8b: Cleaning nationality

Nationality has values like `austria`, `Austria`, `AUSTRIA`, `fRANCE`. We just lowercase everything.
""")

co("""\
identity["nationality"] = identity["nationality"].str.lower().str.strip()
print("Top nationalities:")
print(identity["nationality"].value_counts().head(10))
""")

md("""\
### 8c: Cleaning impossible ages

In the raw data, the maximum age was around 455 years (clearly wrong). Real hockey players are between roughly 16 and 50. Anything over 60 is set to NaN.
""")

co("""\
print("Age before fix: max =", identity["age"].max())
identity.loc[identity["age"] > 60, "age"] = np.nan
print("Age after fix: max =", identity["age"].max())
""")

# =============================================================== STEP 9
md("""\
## Step 9: Joining everything together into one master table

Now we have all the pieces cleaned up. Time to put them together.

We will use **left joins** which means: keep every row in the identity table, and pull in matching rows from the other tables. This way we don't lose any players.

The order of the joins:

1. Start with `identity` (10,000 rows)
2. Join `performance` on `international_id`
3. Join `contracts` on `international_id`
4. Join `scouting` on `international_id`
5. Join `medical` on `medical_id`
6. Join `moms_notes` on first_name + last_name (this is the tricky one)
""")

co("""\
master = identity.copy()
master = master.merge(performance, on="international_id", how="left")
master = master.merge(contracts, on="international_id", how="left")
master = master.merge(scouting, on="international_id", how="left")
master = master.merge(medical, on="medical_id", how="left")

print("Master shape after ID-based joins:", master.shape)
""")

md("""\
### 9b: Joining mom's notes using the names

Mom's notes has no `international_id`! We have to build a join key from `first_name + last_name`. We lowercase both names so capitalization differences don't matter.
""")

co("""\
master["name_key"] = master["first_name"].str.lower().str.strip() + "_" + master["last_name"].str.lower().str.strip()
moms_notes["name_key"] = moms_notes["first_name"].str.lower().str.strip() + "_" + moms_notes["last_name"].str.lower().str.strip()

# How many will match?
master_names = set(master["name_key"])
moms_names = set(moms_notes["name_key"])
matched_names = master_names & moms_names
print("Names that match:", len(matched_names))
print("Master names without a match:", len(master_names - moms_names))
""")

co("""\
# Drop any duplicate names in moms_notes so the join doesn't double rows
moms_for_join = moms_notes.drop_duplicates(subset="name_key", keep="first")

master = master.merge(moms_for_join, on="name_key", how="left", suffixes=("", "_moms"))

# Drop the duplicated columns that came from moms_notes
columns_to_drop = []
for c in master.columns:
    if c.endswith("_moms"):
        columns_to_drop.append(c)
columns_to_drop.append("name_key")
master = master.drop(columns=columns_to_drop)

# Make sure we still have 10,000 unique players
master = master.drop_duplicates(subset="international_id", keep="first").reset_index(drop=True)
print("Master shape after moms join:", master.shape)
""")

md("""\
### 9c: Dropping the distractor columns

The brief warns us that some columns are there just to test if we can spot useless features. These are:

- `eye_color`, `stuffed_animal_name`, `favourite_board_game`, `favourite_child_info`: obvious noise
- `favourite_tv_show`, `favourite_food`: 100% empty strings

(The last two might already be gone from Step 7e, but it doesn't hurt to drop them again with `errors="ignore"`.)
""")

co("""\
distractor_columns = [
    "eye_color",
    "stuffed_animal_name",
    "favourite_board_game",
    "favourite_child_info",
    "favourite_tv_show",
    "favourite_food",
]
master = master.drop(columns=distractor_columns, errors="ignore")
print("Master shape after dropping distractors:", master.shape)
""")

# =============================================================== STEP 10
md("""\
## Step 10: Investigating the mysterious 37 missing rows

When we looked at the files earlier, we saw that every file has about 37 rows that are completely empty in every column (except for the ID). The brief hinted that this is a pattern worth investigating.

Let's check if it's the SAME 37 players missing in every file. If yes, that means the missingness is structural (those players just have no data anywhere), not random.
""")

co("""\
# Find players that are fully missing in each file
fully_null_in_performance = performance[performance.drop(columns=["international_id"]).isnull().all(axis=1)]
fully_null_in_contracts = contracts[contracts.drop(columns=["international_id"]).isnull().all(axis=1)]
fully_null_in_scouting = scouting[scouting.drop(columns=["international_id"]).isnull().all(axis=1)]

ids_perf = set(fully_null_in_performance["international_id"])
ids_contracts = set(fully_null_in_contracts["international_id"])
ids_scouting = set(fully_null_in_scouting["international_id"])

print("Fully empty in performance:", len(ids_perf))
print("Fully empty in contracts:", len(ids_contracts))
print("Fully empty in scouting:", len(ids_scouting))

overlap = ids_perf & ids_contracts & ids_scouting
print("\\nSame players empty in ALL three files:", len(overlap))
print("Some of their IDs:", sorted(list(overlap))[:10])
""")

md("""\
So yes — the SAME 37 players are missing in every file. This is a finding for the presentation.

We keep these rows in the master table (so the total stays at 10,000) but we will not try to make up data for them.
""")

# =============================================================== STEP 11
md("""\
## Step 11: How much is missing in the master table?

Let's see which columns still have NaN and how many.
""")

co("""\
missing_counts = master.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
missing_percent = (missing_counts / len(master) * 100).round(2)

missing_table = pd.DataFrame({"missing": missing_counts, "percent": missing_percent})
display(missing_table)
""")

co("""\
# Make a bar chart of the missing percentages
fig, ax = plt.subplots(figsize=(9, max(4, len(missing_table) * 0.25)))
ax.barh(missing_table.index, missing_table["percent"], color="#0B2E4F")
ax.invert_yaxis()
ax.set_xlabel("Percent missing")
ax.set_title("Missing values per column")
plt.tight_layout()
plt.savefig("docs/figures/fig05_missing_per_column.png", dpi=150, bbox_inches="tight")
plt.show()
""")

co("""\
# Also save the null density per file (raw) for slide 2 reference if needed
raw_frames = {
    "performance.tsv": performance,
    "contracts_competition.pkl": contracts,
    "hockey_scouting_notes.csv": scouting,
    "identity_card_1.csv": identity_1,
    "identity_card_0.tsv": identity_0,
    "moms_notes.json": moms_notes,
}
densities = {}
for name, df in raw_frames.items():
    densities[name] = df.isnull().mean().mean() * 100

density_series = pd.Series(densities).sort_values()

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(density_series.index, density_series.values, color="#0B2E4F")
ax.set_xlabel("% missing (averaged across columns)")
ax.set_title("How much is missing in each file")
plt.tight_layout()
plt.savefig("docs/figures/fig01_null_density_per_file.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# =============================================================== STEP 12
md("""\
## Step 12: Filling in the missing values (imputation)

For each column we need to decide what to do. We use four rules:

1. **Body-related numbers** (height, weight, body fat %, sprint time, shoe size) → fill with the median of the **same position**. Defenders and goalies have different body shapes, so it's better to fill with their group's median than the overall median.
2. **Other numeric columns** (age, draft year, salary, etc.) → fill with the column median. The amount missing is small, so this is fine.
3. **Categorical columns** (dominant_hand, fitness_level) → fill with the most common value (mode) for the same position.
4. **Free-text columns** (scout_notes, medical_information, personal_notes) → leave as NaN. We can deal with text in Phase 2 if we want.
""")

co("""\
# Rule 1: body-related numeric columns - median by position
body_columns = ["height", "weight", "body_fat_percentage", "sprint_time", "shoe_size"]
for column in body_columns:
    if column not in master.columns:
        continue
    # Use the median of the same position to fill in missing values
    fill_values = master.groupby("position")[column].transform("median")
    master[column] = master[column].fillna(fill_values)
    # If the whole position group is missing, use the overall median
    master[column] = master[column].fillna(master[column].median())

print("Body columns filled in.")
""")

co("""\
# Rule 2: other numeric columns - column median
other_numeric = [
    "age",
    "draft_year",
    "salary",
    "years_played",
    "years_pro",
    "school_grade",
    "years_in_usa",
    "contracts_signed",
    "number_of_previous_teams",
]
for column in other_numeric:
    if column not in master.columns:
        continue
    master[column] = master[column].fillna(master[column].median())

print("Other numeric columns filled in.")
""")

co("""\
# Rule 3: categorical columns - mode by position
categorical_columns = ["dominant_hand", "fitness_level"]
for column in categorical_columns:
    if column not in master.columns:
        continue
    # Calculate the most common value for each position
    def get_mode(series):
        modes = series.mode()
        if len(modes) > 0:
            return modes.iloc[0]
        else:
            return np.nan
    position_modes = master.groupby("position")[column].transform(get_mode)
    master[column] = master[column].fillna(position_modes)
    # If still missing, use the overall mode
    overall_modes = master[column].mode()
    if len(overall_modes) > 0:
        master[column] = master[column].fillna(overall_modes.iloc[0])

print("Categorical columns filled in.")
""")

co("""\
# Let's see what is still missing - these are the ones we left as NaN on purpose
still_missing = master.isnull().sum()
still_missing = still_missing[still_missing > 0].sort_values(ascending=False)
print("Columns still containing NaN (left for Phase 2):")
print(still_missing)
""")

md("""\
### Dropping some more columns that are not useful features

- `first_name`, `last_name` are just identifiers
- `medical_id` was only used for joining
- `jersey_number` is superstition, not skill
- `shoe_size` is mostly a function of height
- `age_in_years` is a duplicate of `age` (from a different file)
""")

co("""\
columns_to_drop = ["first_name", "last_name", "medical_id", "jersey_number", "shoe_size", "age_in_years"]
master = master.drop(columns=columns_to_drop, errors="ignore")
print("Master shape after dropping non-feature columns:", master.shape)
""")

# =============================================================== STEP 13
md("""\
## Step 13: Exploratory plots

Now that the data is clean, let's make some plots to understand it better. These plots will go on the slides.
""")

md("### 13a: Distributions panel (position, age, salary, height)")
co("""\
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

# Position counts
position_counts = master["position"].value_counts(dropna=False)
axes[0, 0].barh(position_counts.index.astype(str), position_counts.values, color="#0B2E4F")
axes[0, 0].invert_yaxis()
axes[0, 0].set_title("Number of players by position")

# Age histogram
axes[0, 1].hist(master["age"].dropna(), bins=20, color="#C8102E")
axes[0, 1].set_title("Age distribution")
axes[0, 1].set_xlabel("years")

# Salary on log scale (because most salaries are small but a few are huge)
salaries = master["salary"].dropna()
salaries_positive = salaries[salaries > 0]
axes[1, 0].hist(np.log10(salaries_positive), bins=30, color="#0B2E4F")
axes[1, 0].set_title("Salary (log10 scale)")
axes[1, 0].set_xlabel("log10(salary in USD)")

# Height in cm
axes[1, 1].hist(master["height"].dropna(), bins=30, color="#2E8B57")
axes[1, 1].set_title("Height (cm)")
axes[1, 1].set_xlabel("cm")

plt.tight_layout()
plt.savefig("docs/figures/fig02_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""\
### 13b: Correlation heatmap for the performance columns

A correlation tells us how two columns move together. +1 means they go up together, -1 means one goes up when the other goes down, and 0 means no relationship.
""")

co("""\
performance_columns = [
    "goals", "assists", "num_of_shots", "shot_speed",
    "shooting_percentage", "high_danger_shots",
    "power_play_goals", "time_on_ice", "puck_possession_time",
    "passes_completed", "pass_completion_rate",
    "penalty_minutes", "games_missed_due_to_injury",
]
# Only keep the columns that actually exist in master
performance_columns = [c for c in performance_columns if c in master.columns]

correlation_matrix = master[performance_columns].corr()

fig, ax = plt.subplots(figsize=(9, 7))
heatmap = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(performance_columns)))
ax.set_yticks(range(len(performance_columns)))
ax.set_xticklabels(performance_columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(performance_columns, fontsize=9)

# Write the number on each square
for i in range(len(performance_columns)):
    for j in range(len(performance_columns)):
        value = correlation_matrix.iloc[i, j]
        if abs(value) > 0.55:
            text_color = "white"
        else:
            text_color = "black"
        ax.text(j, i, str(round(value, 2)), ha="center", va="center", fontsize=7, color=text_color)

fig.colorbar(heatmap, ax=ax, shrink=0.8)
ax.set_title("Correlation between performance columns")
plt.tight_layout()
plt.savefig("docs/figures/fig03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# Print the strongest pairs
print("\\nStrongest correlations (positive or negative):")
top_pairs = []
for i in range(len(performance_columns)):
    for j in range(i + 1, len(performance_columns)):
        value = correlation_matrix.iloc[i, j]
        top_pairs.append((performance_columns[i], performance_columns[j], value))
top_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for a, b, v in top_pairs[:8]:
    print("  ", round(v, 2), "  ", a, "<->", b)
""")

md("""\
### 13c: Goals vs assists, colored by position

This plot shows that goalies form a separate cluster near (0, 0) — they don't score goals or assists. Forwards spread out more.
""")

co("""\
position_colors = {
    "defense": "#0B2E4F",
    "left wing": "#C8102E",
    "right wing": "#2E8B57",
    "center": "#DAA520",
    "goalie": "#6A0DAD",
}

fig, ax = plt.subplots(figsize=(8, 5.5))
for position_name, color in position_colors.items():
    subset = master[master["position"] == position_name]
    if len(subset) == 0:
        continue
    ax.scatter(subset["goals"], subset["assists"], s=8, alpha=0.5, label=position_name + " (n=" + str(len(subset)) + ")", color=color)
ax.set_xlabel("goals")
ax.set_ylabel("assists")
ax.set_title("Goals vs assists by position")
ax.legend(fontsize=9, markerscale=1.8, loc="upper left")
plt.tight_layout()
plt.savefig("docs/figures/fig04_goals_vs_assists_by_position.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""\
### 13d: Outlier check with boxplots

A boxplot shows the spread of a column. The dots outside the whiskers are outliers (values much higher or lower than most others).
""")

co("""\
columns_to_check = ["shooting_percentage", "salary", "time_on_ice", "sprint_time", "weight"]
columns_to_check = [c for c in columns_to_check if c in master.columns]

fig, axes = plt.subplots(1, len(columns_to_check), figsize=(3.2 * len(columns_to_check), 4.5))

for i, column in enumerate(columns_to_check):
    data = master[column].dropna()
    axes[i].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#0B2E4F", color="#0B2E4F"),
                    medianprops=dict(color="#C8102E", linewidth=2),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor="#C8102E", markeredgecolor="none"))
    # Calculate how many outliers using the 1.5*IQR rule
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((data < lower) | (data > upper)).sum()
    axes[i].set_title(column, fontsize=10)
    axes[i].set_xlabel("outliers: " + str(n_outliers))
    axes[i].set_xticks([])

plt.suptitle("Outlier check (1.5 x IQR rule)", fontweight="bold")
plt.tight_layout()
plt.savefig("docs/figures/fig06_outlier_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# =============================================================== STEP 14
md("""\
## Step 14: Checking the derived columns

Some columns are supposed to be calculated from others. We should check if they match!

- `pass_completion_rate` should equal `passes_completed / passes_attempted * 100`
- `shooting_percentage` should equal `goals / num_of_shots * 100`

If they don't match, the data has internal inconsistencies that we should flag.
""")

co("""\
# Check pass_completion_rate
passes_attempted_positive = master["passes_attempted"] > 0
recomputed_pass_rate = (master.loc[passes_attempted_positive, "passes_completed"] /
                        master.loc[passes_attempted_positive, "passes_attempted"] * 100)
recomputed_pass_rate = recomputed_pass_rate.round(2)
difference = (recomputed_pass_rate - master.loc[passes_attempted_positive, "pass_completion_rate"]).abs()

print("Pass completion rate check:")
print("  rows checked:", passes_attempted_positive.sum())
print("  rows where it disagrees by more than 1 percent:", (difference > 1).sum())
print("  worst difference:", round(difference.max(), 2))
""")

co("""\
# Check shooting_percentage
shots_positive = master["num_of_shots"] > 0
recomputed_shooting = (master.loc[shots_positive, "goals"] /
                       master.loc[shots_positive, "num_of_shots"] * 100)
recomputed_shooting = recomputed_shooting.round(2)
shooting_diff = (recomputed_shooting - master.loc[shots_positive, "shooting_percentage"]).abs()

print("Shooting percentage check:")
print("  rows checked:", shots_positive.sum())
print("  rows where it disagrees by more than 1 percent:", (shooting_diff > 1).sum())
print("  worst difference:", round(shooting_diff.max(), 2))
""")

md("""\
So **`pass_completion_rate` is correct** but **`shooting_percentage` is wrong on most rows**. We will mention this in the findings slide.
""")

# =============================================================== STEP 15
md("""\
## Step 15: Saving the final master table

We save everything to `data/newdata/master_merged.csv`. This is the file we will use in Phase 2.
""")

co("""\
master.to_csv("data/newdata/master_merged.csv", index=False)
print("Saved!")
print("Final shape:", master.shape)
print("Final columns:")
print(master.columns.tolist())
""")

# =============================================================== STEP 16
md("""\
## Step 16: Summary of what we found (Phase 1 findings)

These are the things we will put on the findings slide:

1. **The same 37 players are missing in every file.** This is a structural missing pattern, not random. We kept these rows in the master table (so total = 10,000) but did not invent data for them.

2. **identity_card_0 needed three separate fixes.** Latin column names, Roman numeral IDs, and Latin name endings — all in one file. We wrote three small functions and applied them.

3. **The medical Excel file had a broken header.** The real column names were in row 0, but `medical_id` was alone in row 1. We had to re-read the file with `header=None` and rebuild the columns by hand.

4. **`pass_completion_rate` is consistent but `shooting_percentage` is not.** When we recomputed shooting_percentage from goals/num_of_shots, 9,511 of 9,963 rows disagreed by more than 1 percent. Worst case was about 87 percent off. We will re-derive it in Phase 2 if needed.

5. **Goalies form a separate cluster** at (0, 0) on the goals-vs-assists scatter. This is why we used position-aware imputation: filling goalie heights with the overall median would have been wrong.

6. **Salary is very skewed to the right** but looks normal on a log scale. We will log-transform it before modeling in Phase 2.

7. **Distractor columns confirmed:** `eye_color`, `stuffed_animal_name`, `favourite_board_game`, `favourite_child_info` have no hockey relevance. `favourite_tv_show` and `favourite_food` are 100% empty. `physician_signature` has joke values like 'Dr. House'. We dropped them all.

8. **Final master table: 10,000 rows × about 55 columns** — all figures regenerated from this notebook.
""")

# =============================================================== WRITE OUT
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Phase1_DataCleaning.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("wrote", output_path, " (" + str(len(cells)) + " cells)")
