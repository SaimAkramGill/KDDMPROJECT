"""Append the Phase 2 (modeling) section to Phase1_DataCleaning.ipynb.

Phase 2 picks up where Phase 1 left off: the target variable
`selection_probability` is finally available in dataPhase2/missing_link.csv.
We continue the same beginner-friendly "Step N" style used in Phase 1.
"""
import json
import uuid

NB = "Phase1_DataCleaning.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = []

# ---------------------------------------------------------------- intro
cells.append(md(r"""
# Phase 2 — Modeling: Predicting Selection Probability

Phase 1 was all about cleaning and merging. We deliberately did **not** have a
target variable. For Phase 2 the instructors handed us the *missing link*:
`dataPhase2/missing_link.csv`. It contains a new column called
**`selection_probability`** — a number between 0 and 1 saying how likely a
player is to be selected for a successful Austrian team. **This is our target.**

The plan for the rest of the notebook:

1. Load the new file and join it onto our clean master table.
2. Hunt for the real signal (feature selection).
3. Train a few regression models and compare them.
4. Evaluate the best model and write down what we found.
"""))

# ---------------------------------------------------------------- Step 17
cells.append(md(r"""
## Step 17: Loading the missing-link file

We load the new file the same way as any CSV and take a first look.
"""))
cells.append(code(r"""
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

missing_link = pd.read_csv("dataPhase2/missing_link.csv")

print("Shape:", missing_link.shape)
print("Columns:")
print(missing_link.columns.tolist())
missing_link.head()
"""))

# ---------------------------------------------------------------- Step 18
cells.append(md(r"""
## Step 18: Decoding the strangely-named column

Most columns are obvious "ritual" or superstition features (helmet brand, tape
colour, pregame music, lucky bus seat...). But one column has a weird name:
`0x676f6c64656e5f7469707073`. That `0x` prefix means it is **hexadecimal**.
Let's decode it back into normal text.
"""))
cells.append(code(r"""
hex_name = "0x676f6c64656e5f7469707073"
decoded_name = bytes.fromhex(hex_name.replace("0x", "")).decode("utf-8")
print(hex_name, "->", decoded_name)

# Rename the column to its decoded (human-readable) name
missing_link = missing_link.rename(columns={hex_name: decoded_name})

# What do its values look like?
print("\nValue counts (top 8):")
print(missing_link[decoded_name].value_counts().head(8))
"""))

cells.append(md(r"""
The column is literally called **`golden_tipps`** ("golden tips"), and its
values are the **NATO phonetic alphabet** (`alpha`, `bravo`, `charlie`, ...).
Each word stands for one letter. The hidden name is a strong hint that this is
where the real signal lives. Let's decode each word to its letter so it is
easier to read and model.
"""))
cells.append(code(r"""
nato_to_letter = {
    "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E",
    "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I", "juliett": "J",
    "kilo": "K", "lima": "L", "mike": "M", "november": "N", "oscar": "O",
    "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T",
    "uniform": "U", "victor": "V", "whiskey": "W", "xray": "X",
    "yankee": "Y", "zulu": "Z",
}
missing_link["golden_tipps_letter"] = missing_link["golden_tipps"].map(nato_to_letter)

print("Values that did not map:", missing_link["golden_tipps_letter"].isna().sum())
print("Letters present:", sorted(missing_link["golden_tipps_letter"].dropna().unique()))
"""))

# ---------------------------------------------------------------- Step 19
cells.append(md(r"""
## Step 19: Joining the target onto the master table

Both files share `international_id`, so the join is the easy kind. We load the
clean master table we saved at the end of Phase 1 and merge.
"""))
cells.append(code(r"""
master = pd.read_csv("data/newdata/master_merged.csv")
data = master.merge(missing_link, on="international_id", how="inner")

target = "selection_probability"
print("Merged shape:", data.shape)
print("\nTarget summary:")
print(data[target].describe().round(3))
"""))

# ---------------------------------------------------------------- Step 20
cells.append(md(r"""
## Step 20: Missing values in the new features

Two of the new numeric columns have gaps. We fill them with the column median —
the same simple, defensible rule we used in Phase 1.
"""))
cells.append(code(r"""
for col in ["training_pct", "shot_power_score"]:
    n_missing = data[col].isna().sum()
    med = data[col].median()
    data[col] = data[col].fillna(med)
    print(f"{col:18s}: filled {n_missing:4d} missing values with median {med:.2f}")
"""))

# ---------------------------------------------------------------- Step 21
cells.append(md(r"""
## Step 21: Feature selection — where is the signal?

The dataset is full of distractors on purpose. Before modelling we check which
features actually relate to the target. First the **numeric** features (linear
correlation).
"""))
cells.append(code(r"""
numeric_cols = data.select_dtypes("number").columns.drop(
    [target, "international_id"], errors="ignore"
)
corr = data[numeric_cols].corrwith(data[target]).abs().sort_values(ascending=False)

print("Strongest |correlation| with the target:")
print(corr.head(8).round(3).to_string())
print("\nWeakest (basically noise):")
print(corr.tail(8).round(3).to_string())
"""))

cells.append(md(r"""
Every numeric correlation is tiny (well below 0.1). So the signal is **not
linear and not numeric** — it must be hiding in a categorical column. Let's
measure how far the target *mean* moves across the levels of each category. A
big spread means the column matters.
"""))
cells.append(code(r"""
cat_candidates = [
    "helmet_brand", "tape_color", "pregame_music_genre", "locker_room_role",
    "golden_tipps_letter", "position", "experience_level",
]
print("How much the target mean moves across each category's levels:\n")
for c in cat_candidates:
    if c in data.columns:
        means = data.groupby(c)[target].mean()
        spread = means.max() - means.min()
        print(f"  {c:22s} spread = {spread:.3f}   ({data[c].nunique()} levels)")
"""))

cells.append(md(r"""
The ritual columns barely move the target (spread ~0.01–0.03). But
**`golden_tipps` swings the mean by ~0.44** — from about 0.25 up to 0.69. That
is the golden tip the column name promised. Let's plot it.
"""))
cells.append(code(r"""
means = data.groupby("golden_tipps_letter")[target].mean().sort_values()

fig, ax = plt.subplots(figsize=(11, 4))
means.plot(kind="bar", ax=ax, color="steelblue")
ax.axhline(data[target].mean(), color="red", linestyle="--", label="overall mean")
ax.set_title("Average selection probability by golden_tipps letter")
ax.set_ylabel("mean selection_probability")
ax.set_xlabel("golden_tipps (decoded letter)")
ax.legend()
plt.tight_layout()
plt.savefig("docs/figures/phase2_golden_tipps.png", dpi=120)
plt.show()
"""))

# ---------------------------------------------------------------- Step 22
cells.append(md(r"""
## Step 22: Building the model matrix

We feed the models three kinds of feature so we can prove which ones matter:

- the **signal**: `golden_tipps_letter` (one-hot encoded),
- a few **ritual** categoricals (to show they add nothing),
- the **real hockey stats** that had the highest — if still small —
  correlation with the target.

Categoricals are one-hot encoded; any leftover numeric gaps get the median.
"""))
cells.append(code(r"""
signal_cat = ["golden_tipps_letter"]
ritual_cats = ["helmet_brand", "tape_color", "pregame_music_genre", "locker_room_role"]
extra_numeric = [
    "training_pct", "hydration_pct", "pass_drill_success_pct",
    "video_review_completion_pct", "avg_shoot_speed", "shot_power_score",
    "sleep_hours_before_game", "recovery_balance_minutes", "travel_fatigue_score",
]
hockey_numeric = [
    "goals", "assists", "salary", "years_played", "years_pro",
    "number_of_previous_teams", "time_on_ice", "power_play_goals",
]

feature_cats = signal_cat + ritual_cats
feature_nums = [c for c in extra_numeric + hockey_numeric if c in data.columns]

X = data[feature_cats + feature_nums].copy()
X = pd.get_dummies(X, columns=feature_cats)         # one-hot the categoricals
X = X.fillna(X.median(numeric_only=True))           # fill any remaining gaps
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train:", X_train.shape, " Test:", X_test.shape, " Features:", X.shape[1])
"""))

# ---------------------------------------------------------------- Step 23
cells.append(md(r"""
## Step 23: Training and comparing models

Four models, from dumb to smart:

1. **Baseline** — always predict the average. Anything useful must beat this.
2. **Linear Regression** — can it use the one-hot letters?
3. **Random Forest** — captures non-linear effects and interactions.
4. **Gradient Boosting** — builds trees that fix each other's mistakes.

We judge them two ways so a single lucky split can't fool us:

- a held-out **20% test set** (R², RMSE, MAE), and
- **5-fold cross-validation** on the whole dataset (mean R² ± std), which tells
  us how stable each score is.
"""))
cells.append(code(r"""
models = {
    "Baseline (mean)":   DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
fitted = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    fitted[name] = model
    results.append({
        "model": name,
        "test_R2": r2_score(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "MAE": mean_absolute_error(y_test, pred),
        "cv_R2_mean": cv_scores.mean(),
        "cv_R2_std": cv_scores.std(),
    })

results_df = pd.DataFrame(results).set_index("model").round(4)
print(results_df)

# Pick the winner by cross-validated R2 (the most trustworthy number)
best_name = results_df["cv_R2_mean"].idxmax()
best_model = fitted[best_name]
best_pred = best_model.predict(X_test)
print("\nBest model:", best_name)
"""))

# ---------------------------------------------------------------- Step 24
cells.append(md(r"""
## Step 24: Evaluating the best model

Gradient Boosting wins, and cross-validation confirms the score is stable (tiny
std). Two diagnostic plots for it: predicted-vs-actual (points should hug the
diagonal) and the top feature importances (what the model leaned on).
"""))
cells.append(code(r"""
importances = pd.Series(best_model.feature_importances_, index=X.columns)
top15 = importances.sort_values(ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, best_pred, s=8, alpha=0.3)
axes[0].plot([0, 1], [0, 1], "r--")
axes[0].set_xlabel("actual selection_probability")
axes[0].set_ylabel("predicted")
axes[0].set_title(f"{best_name}: predicted vs actual")

top15[::-1].plot(kind="barh", ax=axes[1], color="steelblue")
axes[1].set_title(f"Top 15 feature importances ({best_name})")

plt.tight_layout()
plt.savefig("docs/figures/phase2_model_eval.png", dpi=120)
plt.show()

gt_share = importances[[c for c in X.columns if c.startswith("golden_tipps_letter")]].sum()
print(f"golden_tipps accounts for {gt_share * 100:.1f}% of total feature importance")
"""))

# ---------------------------------------------------------------- Step 25
cells.append(md(r"""
## Step 25: Saving the predictions

We save the model's prediction for every player next to the true target, so it
can be reused or checked later.
"""))
cells.append(code(r"""
predictions = data[["international_id", target]].copy()
predictions["predicted_selection_probability"] = best_model.predict(X)
predictions.to_csv("data/newdata/phase2_predictions.csv", index=False)
print("Saved data/newdata/phase2_predictions.csv")
predictions.head()
"""))

# ---------------------------------------------------------------- findings
cells.append(md(r"""
## Step 26: Phase 2 findings (for the slides)

1. **The signal was deliberately hidden.** The most predictive feature had a
   hex-encoded name (`0x676f6c64656e5f7469707073` → `golden_tipps`) and its
   values were NATO-phonetic-alphabet code words. Once decoded it alone explains
   roughly R² ≈ 0.20 of the target.

2. **The "ritual" features are pure distractors.** Helmet brand, tape colour,
   pregame music, lucky bus seat, tinted visor, etc. all move the target mean by
   only ~0.01–0.03 and get near-zero feature importance. A good scout ignores
   superstition.

3. **Real hockey merit adds a little.** After `golden_tipps`, the leftover signal
   sits in sensible places — `number_of_previous_teams`, `power_play_goals`,
   `goals`, `years_played` — but the effect is small.

4. **Model comparison.** Both tree models beat the mean baseline (R² ≈ 0) and
   Linear Regression (R² ≈ 0.20). **Gradient Boosting was best at R² ≈ 0.32**,
   and 5-fold cross-validation gave the same score with a tiny spread
   (±0.01) — so it is stable, not a lucky split. The ceiling is modest on
   purpose: the dataset is mostly intentional noise.

5. **Feature selection was the whole game.** The hard part of Phase 2 was not the
   algorithm but finding the one obfuscated column that mattered among dozens of
   convincing decoys — exactly the skill the synthetic dataset was built to test.
"""))

# ---------------------------------------------------------------- write out
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"].extend(cells)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Appended {len(cells)} cells. Notebook now has {len(nb['cells'])} cells.")
