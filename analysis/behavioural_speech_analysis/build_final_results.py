from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ROOT = Path(
    os.environ.get(
        "BEHAVIOURAL_ANALYSIS_ROOT",
        PROJECT_ROOT / "processed_data" / "behavioural_speech_analysis",
    )
).resolve()
TRIAL_PATH = ROOT / "trial_level_speech_results.csv"
REVIEW_ROOT = Path(
    os.environ.get("BEHAVIOURAL_MANUAL_REVIEW_ROOT", ROOT / "manual_review_sources")
).resolve()
REMAINING_REVIEW_PATH = REVIEW_ROOT / "manual_remaining_error_review.csv"
INCONGRUENT_REVIEW_PATH = REVIEW_ROOT / "manual_incongruent_error_review.csv"
OUT = ROOT / "behavioural_final_manual_corrected"

PARTICIPANTS = [
    "P005", "P006", "P007", "P008", "P009", "P010", "P011", "P012",
    "P013", "P015", "P016", "P017", "P018", "P019", "P020",
]
VISIBILITIES = ["visible", "occluded"]
AUDIO = ["none", "congruent", "incongruent"]
CONDITIONS = [
    "visible_none", "visible_congruent", "visible_incongruent",
    "occluded_none", "occluded_congruent", "occluded_incongruent",
]
ALLOWED_MANUAL = {
    "CORRECT_TARGET", "OTHER_OBJECT", "NO_VALID_OR_UNCLEAR", "AUDITORY_DISTRACTOR"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    adjusted_sorted = np.maximum.accumulate((len(pvals) - np.arange(len(pvals))) * pvals[order])
    adjusted = np.empty_like(pvals)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def build_effect_contrasts():
    avg2 = np.ones(2) / np.sqrt(2)
    bin2 = np.array([-1.0, 1.0]) / np.sqrt(2)
    avg3 = np.ones(3) / np.sqrt(3)
    aud2 = np.array([[-1.0, 1.0, 0.0], [-1.0, -1.0, 2.0]])
    aud2[0] /= np.sqrt(2)
    aud2[1] /= np.sqrt(6)
    return {
        "Visibility": np.vstack([np.kron(v, a) for v in np.atleast_2d([bin2]) for a in np.atleast_2d([avg3])]),
        "Audio": np.vstack([np.kron(v, a) for v in np.atleast_2d([avg2]) for a in np.atleast_2d(aud2)]),
        "Visibility × Audio": np.vstack([np.kron(v, a) for v in np.atleast_2d([bin2]) for a in np.atleast_2d(aud2)]),
    }


EFFECT_CONTRASTS = build_effect_contrasts()


def rm_anova(cell_data: pd.DataFrame) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([VISIBILITIES, AUDIO], names=["visibility", "audio_condition"])
    pivot = cell_data.pivot(
        index="participant", columns=["visibility", "audio_condition"], values="accuracy"
    ).reindex(index=PARTICIPANTS, columns=columns)
    if pivot.isna().any().any():
        raise ValueError(f"Unbalanced repeated-measures table: {int(pivot.isna().sum().sum())} missing cells")
    y = pivot.to_numpy(float)
    n = len(pivot)
    rows = []
    for effect, contrast in EFFECT_CONTRASTS.items():
        z = y @ contrast.T
        if z.ndim == 1:
            z = z[:, None]
        r = z.shape[1]
        mean_z = z.mean(axis=0)
        ss_effect = n * float(np.sum(mean_z ** 2))
        ss_error = float(np.sum((z - mean_z) ** 2))
        df1 = float(r)
        df2 = float((n - 1) * r)
        fval = (ss_effect / df1) / (ss_error / df2) if ss_error > 0 else np.inf
        p_unc = float(stats.f.sf(fval, df1, df2))
        eta = ss_effect / (ss_effect + ss_error) if (ss_effect + ss_error) > 0 else np.nan
        epsilon = 1.0
        mauchly_w = mauchly_chi2 = mauchly_df = mauchly_p = np.nan
        violated = False
        if r > 1:
            covariance = np.cov(z, rowvar=False, ddof=1)
            trace = float(np.trace(covariance))
            trace2 = float(np.trace(covariance @ covariance))
            determinant = max(float(np.linalg.det(covariance)), np.finfo(float).tiny)
            mauchly_w = determinant / ((trace / r) ** r) if trace > 0 else np.nan
            correction = (n - 1) - (2 * r * r + r + 2) / (6 * r)
            mauchly_chi2 = -correction * np.log(max(mauchly_w, np.finfo(float).tiny))
            mauchly_df = r * (r + 1) / 2 - 1
            mauchly_p = float(stats.chi2.sf(mauchly_chi2, mauchly_df))
            epsilon = float(np.clip((trace * trace) / (r * trace2), 1 / r, 1.0)) if trace2 > 0 else 1.0
            violated = bool(mauchly_p < 0.05)
        df1_gg, df2_gg = df1 * epsilon, df2 * epsilon
        p_gg = float(stats.f.sf(fval, df1_gg, df2_gg))
        rows.append({
            "effect": effect,
            "n_participants": n,
            "F": fval,
            "df1_uncorrected": df1,
            "df2_uncorrected": df2,
            "p_uncorrected": p_unc,
            "partial_eta_squared": eta,
            "mauchly_W": mauchly_w,
            "mauchly_chi_square": mauchly_chi2,
            "mauchly_df": mauchly_df,
            "mauchly_p": mauchly_p,
            "sphericity_violated_p_lt_0_05": violated,
            "greenhouse_geisser_epsilon": epsilon,
            "df1_reported": df1_gg if violated else df1,
            "df2_reported": df2_gg if violated else df2,
            "p_greenhouse_geisser": p_gg,
            "p_reported": p_gg if violated else p_unc,
            "correction_used": "Greenhouse-Geisser" if violated else (
                "none; sphericity not applicable" if r == 1 else "none; Mauchly p >= .05"
            ),
            "significant_p_lt_0_05": bool((p_gg if violated else p_unc) < 0.05),
        })
    return pd.DataFrame(rows)


def paired_test(cell_data, effect, label, mask_a, mask_b, family):
    a = cell_data.loc[mask_a].groupby("participant")["accuracy"].mean()
    b = cell_data.loc[mask_b].groupby("participant")["accuracy"].mean()
    both = a.index.intersection(b.index)
    av, bv = a.loc[both].to_numpy(), b.loc[both].to_numpy()
    difference = av - bv
    tval, pval = stats.ttest_rel(av, bv)
    return {
        "triggering_effect": effect,
        "holm_family": family,
        "contrast": label,
        "n_participants": len(both),
        "mean_first": av.mean(),
        "mean_second": bv.mean(),
        "mean_difference_first_minus_second": difference.mean(),
        "t": float(tval),
        "df": len(both) - 1,
        "p_uncorrected": float(pval),
        "cohens_dz": float(difference.mean() / difference.std(ddof=1)) if difference.std(ddof=1) > 0 else np.nan,
    }


def limited_posthocs(cell_data: pd.DataFrame, anova: pd.DataFrame) -> pd.DataFrame:
    significant = set(anova.loc[anova.significant_p_lt_0_05, "effect"])
    rows = []
    if "Visibility" in significant:
        rows.append(paired_test(
            cell_data, "Visibility", "occluded - visible",
            cell_data.visibility.eq("occluded"), cell_data.visibility.eq("visible"), "Visibility",
        ))
    if "Audio" in significant:
        for first, second in [("congruent", "none"), ("incongruent", "none"), ("incongruent", "congruent")]:
            rows.append(paired_test(
                cell_data, "Audio", f"{first} - {second}",
                cell_data.audio_condition.eq(first), cell_data.audio_condition.eq(second), "Audio",
            ))
    if "Visibility × Audio" in significant:
        for audio in AUDIO:
            base = cell_data.audio_condition.eq(audio)
            rows.append(paired_test(
                cell_data, "Visibility × Audio", f"occluded - visible within {audio}",
                base & cell_data.visibility.eq("occluded"), base & cell_data.visibility.eq("visible"),
                "Visibility × Audio simple visibility",
            ))
    output = pd.DataFrame(rows, columns=[
        "triggering_effect", "holm_family", "contrast", "n_participants", "mean_first", "mean_second",
        "mean_difference_first_minus_second", "t", "df", "p_uncorrected", "cohens_dz"
    ])
    if not output.empty:
        output["p_holm"] = np.nan
        for _, indexes in output.groupby("holm_family").groups.items():
            output.loc[indexes, "p_holm"] = holm(output.loc[indexes, "p_uncorrected"])
        output["significant_holm_p_lt_0_05"] = output.p_holm < 0.05
    else:
        output["p_holm"] = pd.Series(dtype=float)
        output["significant_holm_p_lt_0_05"] = pd.Series(dtype=bool)
    return output


def fmt_p(value):
    return "< .001" if value < 0.001 else f"= {value:.3f}"


def main():
    if OUT.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {OUT}")

    source_hashes_before = {
        "trial_results": sha256(TRIAL_PATH),
        "remaining_review": sha256(REMAINING_REVIEW_PATH),
        "incongruent_review": sha256(INCONGRUENT_REVIEW_PATH),
    }
    trials = pd.read_csv(TRIAL_PATH, low_memory=False)
    remaining = pd.read_csv(REMAINING_REVIEW_PATH, low_memory=False)
    incongruent = pd.read_csv(INCONGRUENT_REVIEW_PATH, low_memory=False)

    if sorted(trials.participant.unique()) != sorted(PARTICIPANTS):
        raise ValueError(f"Unexpected participant set: {sorted(trials.participant.unique())}")
    if set(trials.condition.unique()) != set(CONDITIONS):
        raise ValueError("The six expected conditions are not all present")
    if len(remaining) != 850 or len(incongruent) != 482:
        raise ValueError("Expected 850 remaining and 482 Incongruent review rows")
    for table in (trials, remaining, incongruent):
        table["run"] = pd.to_numeric(table.run, errors="raise").astype(int)
        table["trial"] = pd.to_numeric(table.trial, errors="raise").astype(int)
        if table.duplicated(["participant", "run", "trial"]).any():
            raise ValueError("Duplicate participant/run/trial keys detected")

    expected_remaining = {
        ("none", "CORRECT_TARGET"): 265,
        ("none", "OTHER_OBJECT"): 182,
        ("none", "NO_VALID_OR_UNCLEAR"): 3,
        ("congruent", "CORRECT_TARGET"): 253,
        ("congruent", "OTHER_OBJECT"): 147,
    }
    actual_remaining = remaining.groupby(["audio_condition", "manual_classification"]).size().to_dict()
    if actual_remaining != expected_remaining:
        raise ValueError(f"Unexpected None/Congruent review counts: {actual_remaining}")
    expected_incongruent = {
        "AUDITORY_DISTRACTOR": 54, "OTHER_OBJECT": 167,
        "NO_VALID_OR_UNCLEAR": 3, "CORRECT_TARGET": 258,
    }
    if incongruent.manual_classification.value_counts().to_dict() != expected_incongruent:
        raise ValueError("Unexpected Incongruent review counts")
    if not set(remaining.manual_classification).issubset(ALLOWED_MANUAL):
        raise ValueError("Unexpected manual classification in remaining review")
    if not set(incongruent.manual_classification).issubset(ALLOWED_MANUAL):
        raise ValueError("Unexpected manual classification in Incongruent review")

    trials["target_label"] = trials.target_label.astype(str).str.strip().str.lower()
    trials["predicted_label_normalized"] = trials.predicted_label.fillna("").astype(str).str.strip().str.lower()
    predicted_correct = trials.predicted_label_normalized.eq(trials.target_label)
    candidate_mask = ~predicted_correct
    candidate_keys = set(map(tuple, trials.loc[candidate_mask, ["participant", "run", "trial"]].to_numpy()))

    remaining_review = remaining[[
        "participant", "run", "trial", "manual_classification", "reviewed_at_local"
    ]].copy()
    remaining_review["manual_review_source"] = "manual_remaining_error_review.csv"
    incongruent_review = incongruent[[
        "participant", "run", "trial", "manual_classification", "reviewed_at_local", "auditory_distractor"
    ]].copy()
    incongruent_review["manual_review_source"] = "manual_incongruent_error_review.csv"
    reviews = pd.concat([remaining_review, incongruent_review], ignore_index=True, sort=False)
    if reviews.duplicated(["participant", "run", "trial"]).any():
        raise ValueError("Review files overlap")
    review_keys = set(map(tuple, reviews[["participant", "run", "trial"]].to_numpy()))
    if candidate_keys != review_keys:
        raise ValueError(
            f"Review/candidate mismatch: candidates={len(candidate_keys)}, reviews={len(review_keys)}, "
            f"missing={len(candidate_keys-review_keys)}, extra={len(review_keys-candidate_keys)}"
        )
    if len(candidate_keys) != 1332:
        raise ValueError(f"Expected 1,332 automatic-error candidates, found {len(candidate_keys)}")

    updated = trials.merge(reviews, on=["participant", "run", "trial"], how="left", validate="one_to_one")
    updated["automatic_error_candidate"] = ~predicted_correct
    updated["final_correct"] = np.where(predicted_correct, 1, np.nan)
    reviewed = updated.manual_classification.notna()
    updated.loc[reviewed, "final_correct"] = updated.loc[reviewed, "manual_classification"].eq("CORRECT_TARGET").astype(int)
    if updated.final_correct.isna().any():
        raise ValueError("Missing final correctness labels remain")
    updated["final_correct"] = updated.final_correct.astype(int)
    updated["final_label_source"] = np.where(
        reviewed, updated.manual_review_source, "original_automatic_correct_retained"
    )
    updated["final_response_category"] = np.where(
        reviewed, updated.manual_classification, "CORRECT_TARGET"
    )

    manual_correct = int(updated.loc[reviewed, "final_correct"].sum())
    auto_correct = int((~reviewed).sum())
    if manual_correct != 776 or auto_correct != 5130:
        raise ValueError(f"Unexpected correct counts: manual={manual_correct}, original-auto={auto_correct}")
    if int(updated.final_correct.sum()) != 5906 or int((updated.final_correct == 0).sum()) != 556:
        raise ValueError("Final total correct/error counts differ from expected 5906/556")

    condition_summary = (
        updated.groupby(["condition", "visibility", "audio_condition"], sort=False)
        .agg(correct_n=("final_correct", "sum"), total_n=("final_correct", "size"))
        .reset_index()
    )
    condition_summary["accuracy"] = condition_summary.correct_n / condition_summary.total_n
    condition_summary["accuracy_percent"] = 100 * condition_summary.accuracy
    condition_summary["condition"] = pd.Categorical(condition_summary.condition, CONDITIONS, ordered=True)
    condition_summary = condition_summary.sort_values("condition").reset_index(drop=True)
    condition_summary["condition"] = condition_summary.condition.astype(str)

    participant_summary = (
        updated.groupby(["participant", "condition", "visibility", "audio_condition"], sort=False)
        .agg(correct_n=("final_correct", "sum"), total_n=("final_correct", "size"))
        .reset_index()
    )
    participant_summary["accuracy"] = participant_summary.correct_n / participant_summary.total_n
    participant_summary["accuracy_percent"] = 100 * participant_summary.accuracy
    if len(participant_summary) != len(PARTICIPANTS) * 6:
        raise ValueError("Participant × condition table is incomplete")

    overall = pd.DataFrame([{
        "correct_n": int(updated.final_correct.sum()),
        "total_n": len(updated),
        "incorrect_n": int((updated.final_correct == 0).sum()),
        "accuracy": float(updated.final_correct.mean()),
        "accuracy_percent": float(100 * updated.final_correct.mean()),
    }])
    anova = rm_anova(participant_summary)
    posthocs = limited_posthocs(participant_summary, anova)

    manual_summary = (
        updated.loc[reviewed]
        .groupby(["audio_condition", "manual_classification"], dropna=False)
        .size().rename("n").reset_index()
    )
    manual_summary["is_correct"] = manual_summary.manual_classification.eq("CORRECT_TARGET")

    OUT.mkdir(parents=True, exist_ok=False)
    updated.to_csv(OUT / "behavioural_final_trial_level.csv", index=False, encoding="utf-8-sig")
    condition_summary.to_csv(OUT / "behavioural_final_condition_accuracy.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT / "behavioural_final_overall_accuracy.csv", index=False, encoding="utf-8-sig")
    participant_summary.to_csv(
        OUT / "behavioural_final_participant_condition_accuracy.csv", index=False, encoding="utf-8-sig"
    )
    anova.to_csv(OUT / "behavioural_final_rm_anova_2x3.csv", index=False, encoding="utf-8-sig")
    posthocs.to_csv(OUT / "behavioural_final_holm_posthoc.csv", index=False, encoding="utf-8-sig")
    manual_summary.to_csv(OUT / "behavioural_final_manual_review_summary.csv", index=False, encoding="utf-8-sig")

    params = {
        "scope": "Formal cohort P005-P020, with P014 absent from the acquired dataset (15 participants)",
        "n_trials": len(updated),
        "n_automatic_error_candidates_manually_reviewed": int(reviewed.sum()),
        "manual_correct_n": manual_correct,
        "manual_true_error_n": int(reviewed.sum() - manual_correct),
        "original_automatic_correct_retained_n": auto_correct,
        "final_correct_n": int(updated.final_correct.sum()),
        "final_incorrect_n": int((updated.final_correct == 0).sum()),
        "manual_rules": {
            "CORRECT_TARGET": 1,
            "OTHER_OBJECT": 0,
            "AUDITORY_DISTRACTOR": 0,
            "NO_VALID_OR_UNCLEAR": 0,
        },
        "anova": "2 x 3 within-subject repeated-measures ANOVA using orthonormal contrasts, matching the final EEG statistics implementation",
        "sphericity": "Mauchly test for effects containing 3-level Audio; Greenhouse-Geisser p and dfs reported when Mauchly p < .05",
        "effect_size": "partial eta squared = SS_effect / (SS_effect + SS_error)",
        "posthocs": "paired t-tests only for significant omnibus effects; Holm correction within each comparison family",
        "source_sha256": source_hashes_before,
    }
    (OUT / "behavioural_final_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    condition_lines = "\n".join(
        f"- {r.condition}: {r.correct_n}/{r.total_n} = {r.accuracy_percent:.2f}%"
        for r in condition_summary.itertuples()
    )
    anova_lines = "\n".join(
        f"- {r.effect}: F({r.df1_reported:.2f}, {r.df2_reported:.2f}) = {r.F:.3f}, "
        f"p {fmt_p(r.p_reported)}, partial eta-squared = {r.partial_eta_squared:.3f}"
        + (" (Greenhouse-Geisser corrected)" if r.correction_used == "Greenhouse-Geisser" else "")
        for r in anova.itertuples()
    )
    posthoc_lines = "- No post-hoc tests were triggered."
    if not posthocs.empty:
        posthoc_lines = "\n".join(
            f"- {r.contrast}: t({r.df}) = {r.t:.3f}, Holm p {fmt_p(r.p_holm)}, dz = {r.cohens_dz:.3f}"
            for r in posthocs.itertuples()
        )
    report = f"""# Final manually corrected behavioural results

All 1,332 automatic-error candidates were assigned their completed manual classifications. The 5,130 original automatic-correct trials were retained as correct. No source file or earlier output was overwritten.

## Overall accuracy

- {int(overall.correct_n.iloc[0])}/{int(overall.total_n.iloc[0])} = {overall.accuracy_percent.iloc[0]:.2f}%

## Six-condition accuracy

{condition_lines}

## 2 × 3 repeated-measures ANOVA

{anova_lines}

## Holm-corrected paired comparisons

{posthoc_lines}
"""
    (OUT / "behavioural_final_summary.md").write_text(report, encoding="utf-8")

    source_hashes_after = {
        "trial_results": sha256(TRIAL_PATH),
        "remaining_review": sha256(REMAINING_REVIEW_PATH),
        "incongruent_review": sha256(INCONGRUENT_REVIEW_PATH),
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("A source file changed during processing")

    print(json.dumps({
        "output": str(OUT),
        "overall": overall.to_dict("records"),
        "conditions": condition_summary.to_dict("records"),
        "anova": anova[[
            "effect", "F", "df1_reported", "df2_reported", "p_reported",
            "partial_eta_squared", "greenhouse_geisser_epsilon", "correction_used"
        ]].to_dict("records"),
        "posthocs": posthocs.to_dict("records"),
    }, indent=2))


if __name__ == "__main__":
    main()
