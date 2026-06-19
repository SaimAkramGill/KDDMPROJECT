"""Build the Phase I presentation as a .pptx file.

Slide layout (9 slides = optional title + 8 required content slides):
  0  title (optional)
  1  approach
  2  preprocessing I
  3  preprocessing II
  4  relationships I  (correlation heatmap)
  5  relationships II (position-aware)
  6  cleaning I       (missing-value strategy)
  7  cleaning II      (outliers / impossible values)
  8  findings & discussion

Run from the project root:  python src/build_presentation.py
Output: docs/Phase1_Presentation.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "Phase1_Presentation.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

ACCENT = RGBColor(0x0F, 0x4C, 0x81)
INK = RGBColor(0x1F, 0x2A, 0x44)
MUTED = RGBColor(0x55, 0x65, 0x75)


def add_blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_title_band(slide, title: str, subtitle: str | None = None):
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(1.05))
    band.line.fill.background()
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    tf = band.text_frame
    tf.margin_left = Inches(0.5)
    tf.margin_right = Inches(0.5)
    tf.margin_top = Inches(0.15)
    tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    if subtitle:
        sp = tf.add_paragraph()
        sr = sp.add_run()
        sr.text = subtitle
        sr.font.size = Pt(14)
        sr.font.color.rgb = RGBColor(0xDC, 0xE6, 0xF1)


def add_bullets(slide, bullets, left, top, width, height, body_size=16):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p.level = level
        p.space_after = Pt(4)
        r = p.add_run()
        r.text = ("• " if level == 0 else "– ") + text
        r.font.size = Pt(body_size if level == 0 else body_size - 2)
        r.font.color.rgb = INK if level == 0 else MUTED


def add_image(slide, path: Path, left, top, width, height):
    if not path.exists():
        box = slide.shapes.add_textbox(left, top, width, height)
        box.text_frame.text = f"[missing figure: {path.name}]"
        return
    slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def add_footer(slide, text: str):
    box = slide.shapes.add_textbox(
        Inches(0.4), SLIDE_H - Inches(0.4), SLIDE_W - Inches(0.8), Inches(0.3)
    )
    tf = box.text_frame
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = text
    r.font.size = Pt(10)
    r.font.color.rgb = MUTED


def add_centered_text(slide, text, top, size=20, bold=False, color=INK):
    box = slide.shapes.add_textbox(Inches(0.5), top, SLIDE_W - Inches(1.0), Inches(0.6))
    tf = box.text_frame
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color


# ----------------------------- Build deck -----------------------------

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H


# Slide 0 — Title (optional, included to surface team + matriculation numbers)
s = add_blank_slide(prs)
add_title_band(
    s,
    "KDDM Ice Hockey Talent Scouting — Phase I",
    "Data preprocessing, cleaning and exploratory analysis",
)
add_centered_text(s, "Group 174", Inches(2.6), size=24, bold=True, color=ACCENT)
add_centered_text(s, "Muhammad Saim Akram   ·   Matriculation 12344542", Inches(3.4), size=18)
add_centered_text(s, "Zamarak Sufyan", Inches(3.95), size=18)
add_centered_text(s, "Knowledge Discovery and Data Mining   ·   Summer 2026", Inches(5.3), size=14, color=MUTED)
add_centered_text(s, "Files we submit:  Phase1_DataCleaning.ipynb   ·   data/newdata/master_merged.csv", Inches(5.9), size=12, color=MUTED)


# Slide 1 — Approach
s = add_blank_slide(prs)
add_title_band(
    s,
    "Approach — turning 7 messy files into 1 clean table",
    "Phase I · Group 174 · Muhammad Saim Akram (12344542), Zamarak Sufyan",
)
add_bullets(
    s,
    [
        "We have 10,000 ice hockey players spread across 7 files in 5 different formats. The goal of Phase 1 is just to clean the data — we don't have the answer column yet.",
        "Our plan:",
        ("1. Load each of the 7 files using the right pandas function for its format.", 1),
        ("2. Look at each file (shape, head, missing values) before changing anything.", 1),
        ("3. Clean each file on its own (rename, fix types, fix units).", 1),
        ("4. Combine all the files into one big master table using IDs and names.", 1),
        ("5. Fill in the missing values where it makes sense.", 1),
        ("6. Make some plots to understand the data.", 1),
        "Tools we used: pandas, numpy, matplotlib. Plus a small function we wrote to convert Roman numerals to normal numbers.",
        "Final result: data/newdata/master_merged.csv with 10,000 rows × 55 columns, and 6 plots saved in docs/figures/.",
    ],
    Inches(0.6), Inches(1.4), Inches(12.0), Inches(5.5), body_size=18,
)
add_footer(s, "1 / 8 — Approach")


# Slide 2 — Preprocessing I
s = add_blank_slide(prs)
add_title_band(s, "Preprocessing 1 — Fixing the weird files")
add_bullets(
    s,
    [
        "identity_card_0 had three problems we had to fix one by one:",
        ("The column names were in Latin (praenomen, cognomen, ...). We made a small dictionary to rename them.", 1),
        ("The international_id was in Roman numerals (like CCLXXV = 275). We wrote a small function that walks through each letter.", 1),
        ("Some names ended with -ius or -us (Keithus, Alleeius). We removed those endings so the names match the other files.", 1),
        "medical_information.xlsx had a broken header:",
        ("The first row was metadata, the real column names were in another row, and 'medical_id' was alone in a different cell.", 1),
        ("We re-read the file with header=None and built the column names by hand.", 1),
        "The height column was a mix of cm and m (\"1.79m\" vs \"180.1135cm\"). We wrote a small function that checks the ending and converts everything to cm.",
        "weight had a 'kg' suffix sometimes; body_fat_percentage had '%' sometimes. We stripped them off and converted to numbers.",
        "return_date came in 3 different formats. pd.to_datetime with errors='coerce' handles all of them.",
        "There was a typo in performance.tsv: 'penality_minutes' instead of 'penalty_minutes'. We renamed the column.",
    ],
    Inches(0.6), Inches(1.3), Inches(12.0), Inches(5.8), body_size=14,
)
add_footer(s, "2 / 8 — Preprocessing")


# Slide 3 — Preprocessing II
s = add_blank_slide(prs)
add_title_band(s, "Preprocessing 2 — Making things consistent and joining everything")
add_bullets(
    s,
    [
        "Making values consistent:",
        ("gender had values like 'm', 'male', 'M'. We mapped them all to 'male', 'female', or 'other'.", 1),
        ("nationality was written like 'austria', 'Austria', 'AUSTRIA'. We just lowercased everything.", 1),
        ("fitness_level had the literal string 'Nan' instead of a real missing value. We replaced it with np.nan, then lowercased.", 1),
        ("captain and won_championship were stored as object with mixed True/False/strings. We mapped them to real booleans.", 1),
        ("Some ages were over 400 years old. We set anything above 60 to NaN. Draft years before 1960 were also fixed (NHL drafts started in 1963).", 1),
        "Joining the tables (we used left joins so we keep all 10,000 players):",
        ("identity = identity_0 stacked on top of identity_1 = 10,000 rows", 1),
        ("Then we joined performance, contracts and scouting using international_id.", 1),
        ("Then we joined medical using medical_id.", 1),
        ("moms_notes has NO id column, so we built a name_key from lowercase first_name + last_name. 9,815 of our 9,818 keys matched.", 1),
        "Columns we dropped because they have no hockey meaning: eye_color, stuffed_animal_name, favourite_board_game, favourite_child_info, favourite_tv_show, favourite_food, physician_signature, first_name, last_name, medical_id, jersey_number, shoe_size.",
        "Final shape: 10,000 rows × 55 columns.",
    ],
    Inches(0.6), Inches(1.3), Inches(12.0), Inches(5.8), body_size=13,
)
add_footer(s, "3 / 8 — Preprocessing")


# Slide 4 — Relationships I: correlation heatmap
s = add_blank_slide(prs)
add_title_band(s, "Relationships 1 — Which columns move together?")
add_image(s, FIG / "fig03_correlation_heatmap.png", Inches(0.5), Inches(1.3), Inches(7.6), Inches(5.6))
add_bullets(
    s,
    [
        "A correlation tells us how two columns move together. +1 = they go up together, -1 = one goes up when the other goes down, 0 = no relation.",
        "The strongest pairs we found:",
        ("+0.88   power_play_goals and goals — makes sense, most goals happen on power plays.", 1),
        ("+0.75   pass_completion_rate and passes_completed — also makes sense, they measure related things.", 1),
        ("+0.69   games_missed_due_to_injury and assists — this is unexpected and worth looking at in Phase 2.", 1),
        ("+0.51   passes_completed and puck_possession_time — the more you have the puck, the more you can pass.", 1),
        "There is a small block of weak correlations between penalty time, injuries and time on ice — could be interesting.",
    ],
    Inches(8.3), Inches(1.4), Inches(4.8), Inches(5.6), body_size=13,
)
add_footer(s, "4 / 8 — Relationships")


# Slide 5 — Relationships II: position scatter
s = add_blank_slide(prs)
add_title_band(s, "Relationships 2 — Position separates the players cleanly")
add_image(s, FIG / "fig04_goals_vs_assists_by_position.png", Inches(0.5), Inches(1.3), Inches(7.6), Inches(5.6))
add_bullets(
    s,
    [
        "Each dot is one player. The colors show their position. Goalies are clearly different from everyone else — they sit near (0, 0) because they don't score goals.",
        "Forwards (centre, left wing, right wing) spread out to the upper-right (more goals and assists). Defenders sit between forwards and goalies.",
        "This is why we used position-aware imputation later — filling a goalie's missing height with the overall median would be wrong.",
        "We also checked the derived columns:",
        ("pass_completion_rate matches goals/passes_attempted on every row. Good.", 1),
        ("shooting_percentage disagrees with goals/num_of_shots in 9,511 of 9,963 rows. Worst case is about 87 percent off.", 1),
        ("So pass_completion_rate is trustworthy, but shooting_percentage is broken. We'll re-derive it in Phase 2 if we need it.", 1),
    ],
    Inches(8.3), Inches(1.4), Inches(4.8), Inches(5.6), body_size=13,
)
add_footer(s, "5 / 8 — Relationships")


# Slide 6 — Cleaning I: missing values
s = add_blank_slide(prs)
add_title_band(s, "Cleaning 1 — Dealing with missing values")
add_image(s, FIG / "fig05_missing_per_column.png", Inches(0.5), Inches(1.3), Inches(7.6), Inches(5.6))
add_bullets(
    s,
    [
        "Interesting finding: the SAME 37 players are missing in every file. This is not random — they just have no data anywhere. We kept those rows but did not invent data for them.",
        "Three groups of missing columns:",
        ("About 8,518 rows missing medical_information — only injured players have one. We leave it as NaN.", 1),
        ("About 1,000 - 2,500 rows missing free-text columns like scout_notes and personal_notes — we leave them for Phase 2.", 1),
        ("Exactly 37 rows missing across all feature columns — the structural cohort from above.", 1),
        "How we filled in the missing values:",
        ("Body-related columns (height, weight, body fat, sprint time): we used the MEDIAN of the same position. Defenders and goalies have different body shapes.", 1),
        ("Other numeric columns (age, draft_year, salary, etc.): we used the column median.", 1),
        ("Categorical columns (dominant_hand, fitness_level): we used the most common value (mode) for the same position.", 1),
        ("Free text (scout_notes, medical_information, personal_notes): we left as NaN — we can deal with text in Phase 2 if we want.", 1),
    ],
    Inches(8.3), Inches(1.4), Inches(4.8), Inches(5.6), body_size=12,
)
add_footer(s, "6 / 8 — Cleaning")


# Slide 7 — Cleaning II: outliers
s = add_blank_slide(prs)
add_title_band(s, "Cleaning 2 — Outliers and impossible values")
add_image(s, FIG / "fig06_outlier_boxplots.png", Inches(0.5), Inches(1.3), Inches(7.6), Inches(5.6))
add_bullets(
    s,
    [
        "Impossible values we fixed BEFORE filling in missing values (so the medians don't get poisoned):",
        ("age > 60 → NaN. The raw max was around 455 years old.", 1),
        ("draft_year < 1960 → NaN. NHL drafts started in 1963. We found 40 rows like this.", 1),
        ("age_in_years column dropped — it disagreed with age by up to 430 years in some rows.", 1),
        "Outlier check using the IQR rule (anything more than 1.5 × IQR from the box is an outlier):",
        ("We checked shooting_percentage, salary, time_on_ice, sprint_time, weight.", 1),
        ("We just flagged the outliers — we did NOT drop them, because the brief says some are intentional.", 1),
        "Height after cleaning goes from 131.6 to 309 cm. The 309 cm record is clearly a unit-confusion mistake — we should flag this for Phase 2.",
        "Joke / distractor columns we removed: 3 columns in moms_notes were 100% empty (favourite_child_info, favourite_tv_show, favourite_food). physician_signature had values like 'Dr. House' and 'Doctor Strange'.",
    ],
    Inches(8.3), Inches(1.4), Inches(4.8), Inches(5.6), body_size=13,
)
add_footer(s, "7 / 8 — Cleaning")


# Slide 8 — Findings
s = add_blank_slide(prs)
add_title_band(s, "What we found in Phase 1")
add_bullets(
    s,
    [
        "The same 37 players are missing in every file. This is not random — they have no data anywhere. We kept these rows in the master table instead of dropping them.",
        "identity_card_0 needed three independent fixes for one 888-row file: Latin column names, Roman numeral IDs, and Latin name endings. Each one needed its own small function.",
        "medical_information.xlsx had a broken header — the first row was junk, the real column names were in another row. We had to re-read the file and build the column names by hand.",
        "pass_completion_rate is consistent (matches passes_completed / passes_attempted), but shooting_percentage is NOT — 9,511 of 9,963 rows disagree by more than 1 percent.",
        "Goalies form a clear cluster near (0, 0) on goals vs assists — one scatter plot separates the positions. This is why we used position-aware imputation.",
        "Salary is very right-skewed (most are small, a few are huge). On a log scale it looks normal. We will log-transform it before modeling in Phase 2.",
        "Distractor / joke columns confirmed: 3 moms_notes columns are 100% empty, physician_signature has joke values. We dropped them and 9 other useless columns.",
        "Final master: 10,000 rows × 55 columns. All 6 figures were made by the same notebook, so the work is reproducible.",
    ],
    Inches(0.6), Inches(1.3), Inches(12.1), Inches(5.8), body_size=14,
)
add_footer(s, "8 / 8 — Findings")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes, {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
