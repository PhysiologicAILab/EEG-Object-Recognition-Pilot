from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ROOT = Path(
    os.environ.get(
        "BEHAVIOURAL_ANALYSIS_ROOT",
        PROJECT_ROOT / "processed_data" / "behavioural_speech_analysis",
    )
).resolve()
SOURCE = ROOT / "behavioural_final_manual_corrected"
REVIEW_ROOT = Path(
    os.environ.get("BEHAVIOURAL_MANUAL_REVIEW_ROOT", ROOT / "manual_review_sources")
).resolve()
INCONGRUENT_REVIEW = REVIEW_ROOT / "manual_incongruent_error_review.csv"
OUT = ROOT / "final_results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output folder: {OUT}")

    source_files = {
        "behavioural_final_trial_level.csv": "trial_level_results.csv",
        "behavioural_final_condition_accuracy.csv": "condition_accuracy.csv",
        "behavioural_final_overall_accuracy.csv": "overall_accuracy.csv",
        "behavioural_final_participant_condition_accuracy.csv": "participant_condition_accuracy.csv",
        "behavioural_final_rm_anova_2x3.csv": "rm_anova_2x3.csv",
        "behavioural_final_holm_posthoc.csv": "holm_pairwise_comparisons.csv",
        "behavioural_final_manual_review_summary.csv": "manual_review_summary.csv",
        "behavioural_final_parameters.json": "parameters.json",
    }
    required = [SOURCE / name for name in source_files] + [INCONGRUENT_REVIEW]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    hashes_before = {str(path): sha256(path) for path in required}

    trial = pd.read_csv(SOURCE / "behavioural_final_trial_level.csv", low_memory=False)
    conditions = pd.read_csv(SOURCE / "behavioural_final_condition_accuracy.csv")
    overall = pd.read_csv(SOURCE / "behavioural_final_overall_accuracy.csv")
    participant_cells = pd.read_csv(SOURCE / "behavioural_final_participant_condition_accuracy.csv")
    anova = pd.read_csv(SOURCE / "behavioural_final_rm_anova_2x3.csv")
    posthoc = pd.read_csv(SOURCE / "behavioural_final_holm_posthoc.csv")
    review = pd.read_csv(INCONGRUENT_REVIEW)

    checks = {
        "trial_rows": len(trial),
        "participants": trial.participant.nunique(),
        "conditions": trial.condition.nunique(),
        "final_correct_n": int(trial.final_correct.sum()),
        "final_incorrect_n": int((trial.final_correct == 0).sum()),
        "manual_review_rows": int(trial.automatic_error_candidate.sum()),
        "participant_condition_rows": len(participant_cells),
    }
    expected = {
        "trial_rows": 6462,
        "participants": 15,
        "conditions": 6,
        "final_correct_n": 5906,
        "final_incorrect_n": 556,
        "manual_review_rows": 1332,
        "participant_condition_rows": 90,
    }
    if checks != expected:
        raise ValueError(f"Final-label validation failed: {checks}")
    if int(conditions.total_n.sum()) != 6462 or int(conditions.correct_n.sum()) != 5906:
        raise ValueError("Condition totals do not reconcile with trial-level labels")
    if int(overall.correct_n.iloc[0]) != 5906 or int(overall.total_n.iloc[0]) != 6462:
        raise ValueError("Overall totals do not reconcile")

    errors = review.loc[review.manual_classification.ne("CORRECT_TARGET")].copy()
    if len(errors) != 224:
        raise ValueError(f"Expected 224 confirmed Incongruent errors, found {len(errors)}")
    rows = []
    for error_type in ["AUDITORY_DISTRACTOR", "OTHER_OBJECT", "NO_VALID_OR_UNCLEAR"]:
        n = int(errors.manual_classification.eq(error_type).sum())
        rows.append({
            "summary_type": "error_type_overall",
            "visibility": "all",
            "error_type": error_type,
            "n": n,
            "denominator_confirmed_errors": len(errors),
            "percent_of_confirmed_errors": 100 * n / len(errors),
        })
    for visibility in ["visible", "occluded"]:
        subset = errors.loc[errors.visibility.eq(visibility)]
        n = int(subset.manual_classification.eq("AUDITORY_DISTRACTOR").sum())
        rows.append({
            "summary_type": "auditory_distractor_by_visibility",
            "visibility": visibility,
            "error_type": "AUDITORY_DISTRACTOR",
            "n": n,
            "denominator_confirmed_errors": len(subset),
            "percent_of_confirmed_errors": 100 * n / len(subset),
        })
    error_summary = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=False)
    for old_name, new_name in source_files.items():
        shutil.copy2(SOURCE / old_name, OUT / new_name)
    error_summary.to_csv(
        OUT / "incongruent_error_types.csv",
        index=False,
        encoding="utf-8-sig",
    )

    condition_lines = "\n".join(
        f"- {row.condition}: {int(row.correct_n)}/{int(row.total_n)} = {row.accuracy_percent:.2f}%"
        for row in conditions.itertuples()
    )
    anova_lines = "\n".join(
        f"- {row.effect}: F({row.df1_reported:.2f}, {row.df2_reported:.2f}) = {row.F:.3f}, "
        f"p = {row.p_reported:.8g}, partial eta-squared = {row.partial_eta_squared:.3f}; {row.correction_used}"
        for row in anova.itertuples()
    )
    posthoc_lines = "\n".join(
        f"- {row.contrast}: t({int(row.df)}) = {row.t:.3f}, Holm p = {row.p_holm:.8g}, dz = {row.cohens_dz:.3f}"
        for row in posthoc.itertuples()
    )
    error_lines = "\n".join(
        f"- {row.error_type}: {int(row.n)}/{int(row.denominator_confirmed_errors)} = {row.percent_of_confirmed_errors:.2f}%"
        for row in error_summary.loc[error_summary.summary_type.eq("error_type_overall")].itertuples()
    )
    visibility_lines = "\n".join(
        f"- {row.visibility}: {int(row.n)}/{int(row.denominator_confirmed_errors)} = {row.percent_of_confirmed_errors:.2f}%"
        for row in error_summary.loc[error_summary.summary_type.eq("auditory_distractor_by_visibility")].itertuples()
    )
    report = f"""# FINAL MANUAL-CORRECTED BEHAVIOURAL RESULTS

All 1,332 automatic-error candidates were incorporated from the completed manual-review files. No WAV was replayed, no speech recognition was rerun, and no source or earlier result was overwritten.

## Overall accuracy

- {int(overall.correct_n.iloc[0])}/{int(overall.total_n.iloc[0])} = {overall.accuracy_percent.iloc[0]:.2f}%

## Six-condition accuracy

{condition_lines}

## 2 × 3 repeated-measures ANOVA

{anova_lines}

## Holm-corrected pairwise comparisons

{posthoc_lines}

## Incongruent confirmed error types

Confirmed errors: 224.

{error_lines}

Auditory-distractor errors by visibility (denominator: confirmed Incongruent errors within visibility):

{visibility_lines}
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    hashes_after = {str(path): sha256(path) for path in required}
    if hashes_after != hashes_before:
        raise RuntimeError("A source file changed during publication")

    print(json.dumps({
        "output": str(OUT),
        "checks": checks,
        "conditions": conditions.to_dict("records"),
        "overall": overall.to_dict("records"),
        "anova": anova[["effect", "F", "df1_reported", "df2_reported", "p_reported", "partial_eta_squared", "correction_used"]].to_dict("records"),
        "posthoc": posthoc[["contrast", "t", "df", "p_holm", "cohens_dz"]].to_dict("records"),
        "incongruent_errors": error_summary.to_dict("records"),
    }, indent=2))


if __name__ == "__main__":
    main()
