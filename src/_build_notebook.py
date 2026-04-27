"""Generate dataCleaning.ipynb from structured cells.

Run from project root:  python src/_build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataCleaning.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    }


CELLS: list[dict] = [
    md(
        "# Phase I — Data Cleaning\n"
        "**Project:** Ice Hockey Talent Scouting (KDDM)\n\n"
        "This notebook performs Step 2 of the Phase I roadmap and the cleaning that follows from it:\n\n"
        "1. Load each of the 7 raw files with the correct reader.\n"
        "2. Inspect each (`shape`, `dtypes`, `head`, `describe`, `isnull`).\n"
        "3. Document findings inline.\n"
        "4. Apply per-file cleaning.\n"
        "5. Combine identity records and persist clean CSVs to `data/clean/`."
    ),
    md("## 0. Setup"),
    code(
        "from pathlib import Path\n"
        "import re\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "ROOT = Path.cwd()\n"
        "DATA = ROOT / \"data\"\n"
        "OUT = DATA / \"clean\"\n"
        "OUT.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "pd.set_option(\"display.max_columns\", None)\n"
        "pd.set_option(\"display.width\", 200)\n"
        "pd.set_option(\"display.max_colwidth\", 60)\n"
    ),
    md(
        "## 1. Load each file with the correct reader\n\n"
        "| File | Reader | Notes |\n"
        "|---|---|---|\n"
        "| `contracts_competition.pkl` | `read_pickle` | binary pickle |\n"
        "| `hockey_scouting_notes.csv` | `read_csv` | comma-separated |\n"
        "| `identity_card_0.tsv` | `read_csv(sep=\"\\t\")` | Latin headers, Roman-numeral IDs |\n"
        "| `identity_card_1.csv` | `read_csv` | English headers |\n"
        "| `medical_information.xlsx` | `read_excel(engine=\"openpyxl\")` | header is split across two rows |\n"
        "| `moms_notes.json` | `read_json` | array of objects |\n"
        "| `performance.tsv` | `read_csv(sep=\"\\t\")` | tab-separated |"
    ),
    code(
        "contracts = pd.read_pickle(DATA / \"contracts_competition.pkl\")\n"
        "scouting  = pd.read_csv(DATA / \"hockey_scouting_notes.csv\")\n"
        "id0       = pd.read_csv(DATA / \"identity_card_0.tsv\", sep=\"\\t\")\n"
        "id1       = pd.read_csv(DATA / \"identity_card_1.csv\")\n"
        "medical   = pd.read_excel(DATA / \"medical_information.xlsx\", engine=\"openpyxl\")\n"
        "moms      = pd.read_json(DATA / \"moms_notes.json\")\n"
        "perf      = pd.read_csv(DATA / \"performance.tsv\", sep=\"\\t\")\n"
        "\n"
        "frames = {\n"
        "    \"contracts_competition\": contracts,\n"
        "    \"hockey_scouting_notes\": scouting,\n"
        "    \"identity_card_0\":       id0,\n"
        "    \"identity_card_1\":       id1,\n"
        "    \"medical_information\":   medical,\n"
        "    \"moms_notes\":            moms,\n"
        "    \"performance\":           perf,\n"
        "}\n"
        "{name: df.shape for name, df in frames.items()}"
    ),
    md("## 2. Inspect each file\n\nA single helper, then one cell per file."),
    code(
        "def inspect(name, df):\n"
        "    print(f\"=== {name} | shape: {df.shape} ===\")\n"
        "    print(\"\\n[dtypes]\")\n"
        "    print(df.dtypes)\n"
        "    print(\"\\n[head]\")\n"
        "    print(df.head())\n"
        "    num = df.select_dtypes(\"number\")\n"
        "    if not num.empty:\n"
        "        print(\"\\n[describe — numeric]\")\n"
        "        print(num.describe())\n"
        "    obj = df.select_dtypes(\"object\")\n"
        "    if not obj.empty:\n"
        "        print(\"\\n[describe — object]\")\n"
        "        print(obj.describe())\n"
        "    print(\"\\n[isnull %]\")\n"
        "    print((df.isnull().sum() / len(df) * 100).round(2))\n"
        "    print(f\"\\n[duplicate rows]: {df.duplicated().sum()}\")"
    ),
    code("inspect(\"contracts_competition\", contracts)"),
    code("inspect(\"hockey_scouting_notes\", scouting)"),
    code("inspect(\"identity_card_0\", id0)"),
    code("inspect(\"identity_card_1\", id1)"),
    code("inspect(\"medical_information\", medical)"),
    code("inspect(\"moms_notes\", moms)"),
    code("inspect(\"performance\", perf)"),
    md(
        "### Findings (feeds Phase I slides 2–3 and 6–7)\n\n"
        "**`contracts_competition.pkl`** — 10 000 × 8, keyed by `international_id`. `captain` and `won_championship` are stored as Python booleans inside object dtype. Low null rates (<2 %).\n\n"
        "**`hockey_scouting_notes.csv`** — 10 000 × 7, keyed by `international_id`. `dominant_hand` is ~6.5 % null and `scout_notes` is ~10 % null. Position/hand/level need lower-casing.\n\n"
        "**`identity_card_0.tsv`** — only **888 rows**. Column names are Latin (`praenomen`, `cognomen`, `sexus`, …). `international_id` is encoded as **Roman numerals**. Last names carry the Latinized `-ius` suffix. `gender` mixes `m` / `male` / `f` / `female`. `birth_city` mixes case (`vienna`, `Vienna`, `OSTRAVA`). `age` has impossible outliers (max **390**).\n\n"
        "**`identity_card_1.csv`** — 9 112 rows in English. Same gender / city / nationality inconsistencies (`U.S.A` vs `usa`). `age` max is **455** — same outlier pattern. Together id_0 + id_1 ≈ 10 000 unique players.\n\n"
        "**`medical_information.xlsx`** — header is **split across two rows**: `medical_id` lives in row 1 col 0 while the other column names live in row 0 cols 1–10. After re-reading: `height` is mixed `\"1.79m\"` / `\"180.1135cm\"`, `weight` includes the `kg` suffix and the literal token `\"nankg\"`, `body_fat_percentage` is sometimes `\"20.87%\"`. The `medical_information` column is ~85 % null.\n\n"
        "**`moms_notes.json`** — 10 000 × 13 with **36 duplicate rows** and **3 entirely constant columns** (`favourite_child_info`, `favourite_tv_show`, `favourite_food`). `personal_notes` is 25 % null. `age` carries the same impossible-value outliers as the identity files. **No id column** — must be joined on `(first_name, last_name, age, birth_city)`.\n\n"
        "**`performance.tsv`** — 10 000 × 29, the cleanest file. All numeric, very low nulls, keyed by `international_id`. Note typo `penality_minutes` (sic)."
    ),
    md(
        "## 3. Cleaning\n\n"
        "Each subsection produces one `*_clean` DataFrame. Final cell concatenates identities and writes everything to `data/clean/`."
    ),
    md("### 3.1 `moms_notes` — drop constant columns and duplicates"),
    code(
        "constant_cols = [c for c in moms.columns if moms[c].nunique(dropna=False) <= 1]\n"
        "print(\"constant columns dropped:\", constant_cols)\n"
        "\n"
        "moms_clean = (\n"
        "    moms\n"
        "    .drop(columns=constant_cols)\n"
        "    .drop_duplicates()\n"
        "    .reset_index(drop=True)\n"
        ")\n"
        "moms_clean.shape"
    ),
    md("### 3.2 `contracts_competition` — coerce booleans, sanity-check `draft_year`"),
    code(
        "contracts_clean = contracts.copy()\n"
        "\n"
        "BOOL_MAP = {True: True, False: False, \"True\": True, \"False\": False, \"true\": True, \"false\": False}\n"
        "for col in [\"captain\", \"won_championship\"]:\n"
        "    contracts_clean[col] = contracts_clean[col].map(BOOL_MAP).astype(\"boolean\")\n"
        "\n"
        "# draft_year minimum was 1850 — implausible. NHL drafts began in 1963.\n"
        "implausible = contracts_clean[\"draft_year\"] < 1960\n"
        "print(f\"implausible draft_year set to NaN: {implausible.sum()}\")\n"
        "contracts_clean.loc[implausible, \"draft_year\"] = np.nan\n"
        "\n"
        "contracts_clean.dtypes"
    ),
    md(
        "### 3.3 `identity_card_0` — Latin → English headers, Roman → integer IDs\n\n"
        "Decoder is the standard subtractive-rule Roman numeral parser (e.g. `MMDLX` = 2560)."
    ),
    code(
        "LATIN_TO_EN = {\n"
        "    \"numerus_internationalis_ad_identitatem\": \"international_id\",\n"
        "    \"numerus_identificationis_ad_medicinam\":  \"medical_id\",\n"
        "    \"praenomen\":      \"first_name\",\n"
        "    \"cognomen\":       \"last_name\",\n"
        "    \"sexus\":          \"gender\",\n"
        "    \"aetas_annorum\":  \"age\",\n"
        "    \"urbs_natalis\":   \"birth_city\",\n"
        "    \"natio\":          \"nationality\",\n"
        "}\n"
        "\n"
        "ROMAN = {\"M\": 1000, \"D\": 500, \"C\": 100, \"L\": 50, \"X\": 10, \"V\": 5, \"I\": 1}\n"
        "\n"
        "def roman_to_int(s):\n"
        "    if not isinstance(s, str):\n"
        "        return pd.NA\n"
        "    total, prev = 0, 0\n"
        "    for ch in reversed(s.upper().strip()):\n"
        "        v = ROMAN.get(ch)\n"
        "        if v is None:\n"
        "            return pd.NA\n"
        "        total += -v if v < prev else v\n"
        "        prev = v\n"
        "    return total\n"
        "\n"
        "id0_clean = id0.rename(columns=LATIN_TO_EN).copy()\n"
        "id0_clean[\"international_id\"] = id0_clean[\"international_id\"].map(roman_to_int).astype(\"Int64\")\n"
        "# strip the Latin -ius suffix that was bolted onto last names\n"
        "id0_clean[\"last_name\"] = id0_clean[\"last_name\"].str.replace(r\"ius$\", \"\", regex=True)\n"
        "id0_clean[[\"international_id\", \"last_name\"]].head()"
    ),
    md("### 3.4 Shared normalization for both identity files (gender, city, nationality, age outliers)"),
    code(
        "GENDER_MAP = {\n"
        "    \"m\": \"male\", \"male\": \"male\",\n"
        "    \"f\": \"female\", \"female\": \"female\",\n"
        "    \"other\": \"other\",\n"
        "}\n"
        "\n"
        "NATIONALITY_MAP = {\n"
        "    \"u.s.a\": \"usa\", \"u.s.a.\": \"usa\", \"usa\": \"usa\",\n"
        "    \"united states\": \"usa\", \"united states of america\": \"usa\",\n"
        "}\n"
        "\n"
        "def norm_gender(g):\n"
        "    if not isinstance(g, str):\n"
        "        return np.nan\n"
        "    return GENDER_MAP.get(g.strip().lower(), g.strip().lower())\n"
        "\n"
        "def norm_nationality(s):\n"
        "    if not isinstance(s, str):\n"
        "        return np.nan\n"
        "    key = s.strip().lower().rstrip(\".\")\n"
        "    return NATIONALITY_MAP.get(key, key)\n"
        "\n"
        "id1_clean = id1.copy()\n"
        "\n"
        "for df in (id0_clean, id1_clean):\n"
        "    df[\"gender\"]      = df[\"gender\"].map(norm_gender)\n"
        "    df[\"birth_city\"]  = df[\"birth_city\"].astype(\"string\").str.title()\n"
        "    df[\"nationality\"] = df[\"nationality\"].map(norm_nationality)\n"
        "    df.loc[df[\"age\"] > 60, \"age\"] = np.nan  # drop the 390 / 455 garbage\n"
        "\n"
        "moms_clean.loc[moms_clean[\"age\"] > 60, \"age\"] = np.nan\n"
        "\n"
        "print(\"id0 genders:\", id0_clean[\"gender\"].value_counts(dropna=False).to_dict())\n"
        "print(\"id1 genders:\", id1_clean[\"gender\"].value_counts(dropna=False).to_dict())\n"
        "print(\"id0 age max:\", id0_clean[\"age\"].max(), \"| id1 age max:\", id1_clean[\"age\"].max())"
    ),
    md("### 3.5 Combine the two identity files"),
    code(
        "identity = (\n"
        "    pd.concat([id0_clean, id1_clean], ignore_index=True)\n"
        "      .drop_duplicates(subset=[\"international_id\"])\n"
        "      .reset_index(drop=True)\n"
        ")\n"
        "identity[\"international_id\"] = identity[\"international_id\"].astype(\"Int64\")\n"
        "print(\"combined identity:\", identity.shape)\n"
        "print(\"unique medical_ids:\", identity[\"medical_id\"].nunique())"
    ),
    md(
        "### 3.6 `medical_information` — re-read with proper headers, parse units\n\n"
        "The file has a junk metadata row at the top, real column names on row 1, and `medical_id` parked alone in row 2 col 0. Real data starts at row 3."
    ),
    code(
        "med_raw = pd.read_excel(DATA / \"medical_information.xlsx\", engine=\"openpyxl\", header=None)\n"
        "cols = [\"medical_id\"] + med_raw.iloc[1, 1:].tolist()\n"
        "medical_clean = med_raw.iloc[3:].reset_index(drop=True).copy()\n"
        "medical_clean.columns = cols\n"
        "medical_clean.head()"
    ),
    code(
        "def parse_height(x):\n"
        "    \"\"\"Return height in metres. Handles '1.79m', '180.1135cm', bare numbers, NaN.\"\"\"\n"
        "    if pd.isna(x):\n"
        "        return np.nan\n"
        "    s = str(x).strip().lower()\n"
        "    if \"nan\" in s:\n"
        "        return np.nan\n"
        "    m = re.match(r\"([\\d.]+)\\s*(cm|m)?\", s)\n"
        "    if not m:\n"
        "        return np.nan\n"
        "    val = float(m.group(1))\n"
        "    return val / 100 if m.group(2) == \"cm\" else val\n"
        "\n"
        "def parse_weight(x):\n"
        "    \"\"\"Return weight in kg. 'nankg' and similar tokens become NaN.\"\"\"\n"
        "    if pd.isna(x):\n"
        "        return np.nan\n"
        "    s = str(x).strip().lower()\n"
        "    if \"nan\" in s:\n"
        "        return np.nan\n"
        "    m = re.match(r\"([\\d.]+)\", s)\n"
        "    return float(m.group(1)) if m else np.nan\n"
        "\n"
        "def parse_pct(x):\n"
        "    if pd.isna(x):\n"
        "        return np.nan\n"
        "    s = str(x).strip().rstrip(\"%\")\n"
        "    try:\n"
        "        return float(s)\n"
        "    except ValueError:\n"
        "        return np.nan\n"
        "\n"
        "medical_clean[\"height\"]              = medical_clean[\"height\"].map(parse_height)\n"
        "medical_clean[\"weight\"]              = medical_clean[\"weight\"].map(parse_weight)\n"
        "medical_clean[\"age_in_years\"]        = pd.to_numeric(medical_clean[\"age_in_years\"], errors=\"coerce\")\n"
        "medical_clean[\"shoe_size\"]           = pd.to_numeric(medical_clean[\"shoe_size\"], errors=\"coerce\")\n"
        "medical_clean[\"body_fat_percentage\"] = medical_clean[\"body_fat_percentage\"].map(parse_pct)\n"
        "medical_clean[\"sprint_time\"]         = pd.to_numeric(medical_clean[\"sprint_time\"], errors=\"coerce\")\n"
        "medical_clean[\"fitness_level\"]       = (\n"
        "    medical_clean[\"fitness_level\"].astype(\"string\").str.lower().str.strip()\n"
        ")\n"
        "medical_clean[\"return_date\"]         = pd.to_datetime(medical_clean[\"return_date\"], errors=\"coerce\")\n"
        "\n"
        "medical_clean.dtypes"
    ),
    md("### 3.7 `hockey_scouting_notes` — normalize categorical text"),
    code(
        "scouting_clean = scouting.copy()\n"
        "for col in [\"position\", \"dominant_hand\", \"experience_level\"]:\n"
        "    scouting_clean[col] = (\n"
        "        scouting_clean[col].astype(\"string\").str.lower().str.strip()\n"
        "    )\n"
        "scouting_clean[[\"position\", \"dominant_hand\", \"experience_level\"]].nunique()"
    ),
    md(
        "### 3.8 `performance` — sanity check `pass_completion_rate`\n\n"
        "Already numeric; just verify the rate column is consistent with `passes_completed / passes_attempted`."
    ),
    code(
        "perf_clean = perf.copy()\n"
        "perf_clean = perf_clean.rename(columns={\"penality_minutes\": \"penalty_minutes\"})\n"
        "\n"
        "mask = perf_clean[\"passes_attempted\"] > 0\n"
        "recomputed = (\n"
        "    perf_clean.loc[mask, \"passes_completed\"]\n"
        "    / perf_clean.loc[mask, \"passes_attempted\"] * 100\n"
        ").round(2)\n"
        "diff = (recomputed - perf_clean.loc[mask, \"pass_completion_rate\"]).abs()\n"
        "print(f\"rows where reported rate differs from recomputed by >1pp: {(diff > 1).sum()}\")"
    ),
    md("## 4. Persist clean tables"),
    code(
        "contracts_clean.to_csv(OUT / \"contracts.csv\",  index=False)\n"
        "scouting_clean.to_csv (OUT / \"scouting.csv\",   index=False)\n"
        "identity.to_csv       (OUT / \"identity.csv\",   index=False)\n"
        "medical_clean.to_csv  (OUT / \"medical.csv\",    index=False)\n"
        "moms_clean.to_csv     (OUT / \"moms.csv\",       index=False)\n"
        "perf_clean.to_csv     (OUT / \"performance.csv\",index=False)\n"
        "\n"
        "for f in sorted(OUT.iterdir()):\n"
        "    print(f.name, \"-\", f.stat().st_size, \"bytes\")"
    ),
    md(
        "## 5. Next steps\n\n"
        "- Join `identity` (international_id ↔ medical_id) with `medical_clean` on `medical_id`.\n"
        "- Join everything else on `international_id`.\n"
        "- Match `moms_clean` to identities on `(first_name, last_name, age, birth_city)` — there is no shared id.\n"
        "- Outlier handling on `performance` (e.g. `shooting_percentage`, `salary` log-distribution).\n"
        "- Imputation strategy for the remaining nulls (slide 6–7 material)."
    ),
]


notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
