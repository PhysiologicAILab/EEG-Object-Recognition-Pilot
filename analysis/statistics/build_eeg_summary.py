from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ROOT = Path(
    os.environ.get("EEG_PROCESSED_ROOT", PROJECT_ROOT / "processed_data" / "eeg")
).resolve()
PRE = ROOT / "final_preprocessing"
SPEC = ROOT / "final_spectral_features"
ROI = ROOT / "final_roi_statistics"
OUT = Path(os.environ.get("EEG_DISSERTATION_SUMMARY_OUTPUT", ROOT / "dissertation_eeg_summary")).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def p_text(p: float) -> str:
    return "p < .001" if p < .001 else f"p = {p:.3f}"


def result_interpretation(band: str, effect: str, p: float) -> str:
    if band == "alpha" and effect == "Visibility":
        return "Significant in N=14; visible conditions had higher alpha relative power than occluded conditions."
    if band == "alpha" and effect == "Audio":
        return "Not significant after Greenhouse–Geisser correction."
    if band == "theta" and effect == "Audio":
        return "Significant Audio main effect; Holm-corrected follow-up supported incongruent > none."
    if band == "theta" and effect == "ROI":
        return "Significant ROI main effect; Occipital theta relative power was higher than Parietal."
    if band == "theta" and effect == "Audio × ROI":
        return "Significant interaction; Occipital–Parietal differences were supported for congruent and incongruent, but not none."
    if band == "beta":
        return "Exploratory; not statistically significant." if p >= .05 else "Exploratory result."
    return "Not statistically significant." if p >= .05 else "Statistically significant omnibus effect."


def build_anova_table(band: str, include_theta_posthoc: bool = False) -> pd.DataFrame:
    source_name = f"{band}_exploratory_rm_anova.csv" if band == "beta" else f"{band}_rm_anova.csv"
    anova = pd.read_csv(ROI / source_name)
    sensitivity = pd.read_csv(ROI / "sensitivity_N13.csv") if band in {"alpha", "theta"} else pd.DataFrame()
    rows = []
    for _, r in anova.iterrows():
        sens = (sensitivity[(sensitivity.outcome == f"{band}_relative") & (sensitivity.effect == r.effect)]
                if not sensitivity.empty else pd.DataFrame())
        if sens.empty:
            n13 = {"F": None, "df1": None, "df2": None, "p": None, "eta": None, "sig": None, "note": "Not run for exploratory beta."}
        else:
            s = sens.iloc[0]
            change = "same significance classification" if bool(s.significance_pattern_same) else "crossed the .05 threshold"
            n13 = {"F": s.N13_F, "df1": s.N13_df1_reported, "df2": s.N13_df2_reported,
                   "p": s.N13_p_reported, "eta": s.N13_partial_eta_squared, "sig": bool(s.N13_significant),
                   "note": f"{change}; direction remained similar; |Δ partial η²|={s.absolute_eta_squared_change:.3f}."}
        rows.append({
            "row_type": "OMNIBUS_ANOVA", "band": band, "effect_or_contrast": r.effect,
            "N": int(r.n_participants), "F": r.F, "df1_reported": r.df1_reported,
            "df2_reported": r.df2_reported, "t": None, "paired_df": None,
            "p_uncorrected": r.p_uncorrected, "p_reported_or_holm": r.p_reported,
            "partial_eta_squared": r.partial_eta_squared, "cohens_dz": None,
            "correction": r.correction_used, "significant_p_lt_0_05": bool(r.significant_p_lt_0_05),
            "N13_F_excluding_P007": n13["F"], "N13_df1_reported": n13["df1"],
            "N13_df2_reported": n13["df2"], "N13_p_reported": n13["p"],
            "N13_partial_eta_squared": n13["eta"], "N13_significant": n13["sig"],
            "sensitivity_summary": n13["note"],
            "dissertation_interpretation": result_interpretation(band, r.effect, r.p_reported),
            "source_file": str(ROI / source_name),
        })
    if include_theta_posthoc:
        post = pd.read_csv(ROI / "posthoc.csv")
        post = post[post.outcome == "theta_relative"]
        for _, r in post.iterrows():
            rows.append({
                "row_type": "HOLM_POSTHOC", "band": band, "effect_or_contrast": r.contrast,
                "N": int(r.n), "F": None, "df1_reported": None, "df2_reported": None,
                "t": r.t, "paired_df": r.df, "p_uncorrected": r.p_uncorrected,
                "p_reported_or_holm": r.p_holm, "partial_eta_squared": None,
                "cohens_dz": r.cohens_dz, "correction": f"Holm within {r.holm_family}",
                "significant_p_lt_0_05": bool(r.significant_holm_p_lt_0_05),
                "N13_F_excluding_P007": None, "N13_df1_reported": None, "N13_df2_reported": None,
                "N13_p_reported": None, "N13_partial_eta_squared": None, "N13_significant": None,
                "sensitivity_summary": "Post-hoc sensitivity was not rerun; only the pre-existing omnibus N=13 analysis is summarized.",
                "dissertation_interpretation": ("Holm-corrected paired comparison was significant." if r.significant_holm_p_lt_0_05 else "Holm-corrected paired comparison was not significant."),
                "source_file": str(ROI / "posthoc.csv"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    source_files = [
        PRE / "report.md", PRE / "parameters.json",
        PRE / "participant_qc.csv", PRE / "channel_qc.csv",
        SPEC / "report.md", SPEC / "parameters.json",
        ROI / "alpha_anova.csv", ROI / "theta_anova.csv", ROI / "beta_anova.csv",
        ROI / "posthoc.csv", ROI / "sensitivity_N13.csv",
        ROI / "report.md", ROI / "parameters.json",
    ]
    hashes_before = {str(p): sha256(p) for p in source_files}
    pqc = pd.read_csv(PRE / "participant_qc.csv")
    if len(pqc) != 15 or pqc.original_formal_epochs.sum() != 6461 or pqc.final_retained_epochs.sum() != 3924:
        raise ValueError("Final preprocessing totals do not match the finalized report")
    if dict(pqc.provisional_qc.value_counts()) != {"YELLOW": 11, "GREEN": 3, "RED": 1}:
        raise ValueError("Participant QC counts do not match the finalized report")

    OUT.mkdir(parents=True, exist_ok=True)
    methods = pd.DataFrame([
        ["Cohort", "Final preprocessing cohort", "15 recorded participant/session labels: P005–P013 and P015–P020; P014 absent; pilots P001–P004 excluded", "report.md"],
        ["Cohort", "Primary EEG analysis cohort", "N=14; P007 retained; P011 excluded", "parameters.json"],
        ["Cohort", "Excluded participant", "P011: RED QC, 94/432 retained epochs (21.8%) and 10–19 epochs per condition", "participant_qc.csv"],
        ["Acquisition", "Sampling rate", "500 Hz nominal", "parameters.json"],
        ["Acquisition", "EEG channels", "16: Fp1, Fp2, F3, Fz, F4, C3, Cz, C4, P7, P3, Pz, P4, P8, O1, Oz, O2; stored Fpz renamed to Fp1 to match physical placement", "parameters.json"],
        ["Acquisition", "Montage", "MNE standard_1020; all 16 coordinates recognized", "report.md"],
        ["Acquisition", "Reference", "SAGA Average Reference retained; no additional software rereference; no zero-filled reference channel", "parameters.json"],
        ["Preprocessing", "Filter", "0.1–40 Hz zero-phase Hamming-window FIR (firwin), length 16,501 samples", "parameters.json"],
        ["Preprocessing", "Notch decision", "No 50-Hz notch; diagnostic showed the 40-Hz low-pass made an additional notch unnecessary", "report.md"],
        ["Epoching", "Epoch window", "Image onset, −0.2 to +0.8 s", "parameters.json"],
        ["Epoching", "Baseline", "−0.2 to 0 s", "parameters.json"],
        ["Artifact control", "Ocular-artifact strategy", "Synchronized Fp1/Fp2 detector: 8-Hz low-pass, 4 robust-sigma height, 1.5-sigma prominence, ≥0.40-s separation, same polarity; 20-s recording edges excluded", "parameters.json"],
        ["Artifact control", "Manual detector validation", "Finalized validation metrics carried into preprocessing: PPV 98.4% and recall 78.9%", "report.md; parameters.json"],
        ["Artifact control", "Detector interpretation", "High-confidence first rejection rule; detector-negative epochs were not assumed clean, so a residual amplitude rule followed", "report.md"],
        ["Artifact control", "Residual rejection", "After ocular exclusion, reject an epoch when maximum peak-to-peak amplitude across any EEG channel was >250 µV; fixed identically for all participants", "parameters.json"],
        ["Artifact control", "Other excluded methods", "No ICA, interpolation, participant-specific threshold tuning, software rereferencing, or additional notch", "report.md"],
        ["Spectral analysis", "PSD", "Welch per retained epoch: Hann, 500 samples, 250 overlap, nfft=500, constant detrend, density scaling; epoch PSDs averaged within participant×condition×channel", "parameters.json"],
        ["Spectral analysis", "Primary bands", "Relative theta 4–8 Hz and relative alpha 8–13 Hz; denominator total power 1–40 Hz", "parameters.json"],
        ["Spectral analysis", "Exploratory band", "Relative beta 13–30 Hz; denominator total power 1–40 Hz", "parameters.json"],
        ["ROI", "Occipital ROI", "O1, Oz, O2", "parameters.json"],
        ["ROI", "Parietal ROI", "P3, Pz, P4; P7/P8 excluded from primary ROIs", "parameters.json"],
        ["Statistics", "Primary model", "2 Visibility × 3 Audio × 2 ROI fully within-participant repeated-measures ANOVA; Mauchly test and Greenhouse–Geisser correction where required; partial eta-squared", "parameters.json"],
        ["Statistics", "Post-hoc rule", "Limited paired comparisons only after significant interpretable omnibus effects; Holm correction within comparison family", "parameters.json"],
        ["Statistics", "Sensitivity rule", "Primary N=14 includes P007; repeat alpha/theta omnibus model at N=13 excluding P007 only", "parameters.json"],
    ], columns=["section", "method_item", "finalized_specification", "source_output"])
    write_csv(methods, "dissertation_methods_summary.csv")

    technical = pd.DataFrame([
        ["Total FINAL-protocol participants", "15", "P005–P013 and P015–P020; P014 absent from the recorded session identifiers", "report.md"],
        ["Total raw formal image epochs", "6461", "Expected 6480 minus 19 missing P005 image onsets", "report.md"],
        ["Total retained epochs", "3924", "After ocular exclusion and fixed >250 µV peak-to-peak rejection", "report.md"],
        ["Median participant retention", "50.0%", "Across all 15 final participants", "report.md"],
        ["Participant retention range", "21.8%–92.1%", "P011 lowest; P013 highest", "participant_qc.csv"],
        ["Participant QC counts", "GREEN=3; YELLOW=11; RED=1", "Transparent screening labels; only RED participant excluded from primary EEG analysis", "report.md"],
        ["Final condition-level EEG N", "14", "P011 excluded; P007 retained with N=13 sensitivity analysis", "report.md"],
        ["Marker integrity", "14/15 participants had 432 image onsets; P005 had 413 and 19 documented missing image onsets", "All other final participants contributed the expected formal image epochs; one selected PsychoPyMarkers stream per XDF was used and duplicate streams were not merged", "participant_qc.csv; report.md"],
        ["EEG timestamp/sampling integrity", "Finalized import/epoch construction completed for all 15 participants", "The final three-stage outputs do not contain a separate cohort-wide timestamp-gap/non-monotonic count; therefore this package does not make that stronger validation claim", "report.md; parameters.json"],
        ["Posterior channel usability", "15/15 YES", "No globally bad channels; posterior WATCH channels remained documented and no interpolation was performed", "participant_qc.csv; channel_qc.csv"],
        ["Spectral feature completeness", "1344/1344 expected rows", "14 participants × 6 conditions × 16 channels; no NaN/Inf, negative powers, or missing combinations", "report.md"],
        ["ROI model completeness", "168/168 expected rows", "14 participants × 6 conditions × 2 preregistered-style ROIs", "report.md"],
    ], columns=["validation_item", "final_value", "interpretation_or_scope", "source_output"])
    write_csv(technical, "dissertation_technical_validation_summary.csv")

    alpha = build_anova_table("alpha", include_theta_posthoc=False)
    theta = build_anova_table("theta", include_theta_posthoc=True)
    beta = build_anova_table("beta", include_theta_posthoc=False)
    write_csv(alpha, "dissertation_alpha_results.csv")
    write_csv(theta, "dissertation_theta_results.csv")
    write_csv(beta, "dissertation_beta_results.csv")

    figures = pd.DataFrame([
        ["Figure 3.1", "participant_retention.png", str(PRE / "participant_retention.png"), "Methods / preprocessing QC", "Participant-level percentage of image epochs retained after the finalized two-stage artifact procedure.", "Retention varied substantially across participants; this figure documents data quantity rather than a condition effect.", "Main text"],
        ["Figure 3.2", "condition_counts.png", str(PRE / "condition_counts.png"), "Methods / preprocessing QC", "Retained epoch counts for each of the six experimental conditions by participant.", "All conditions remained represented, but several participants had low or imbalanced condition counts.", "Main text"],
        ["Figure 3.3", "posterior_psd.png", str(PRE / "posterior_psd.png"), "Methods / signal-quality validation", "Posterior-channel PSD overview after final preprocessing.", "The posterior spectra provide a technical plausibility check and should not be interpreted as evidence of condition effects.", "Main text or supplement"],
        ["Figure 4.1", "posterior_psd.png", str(SPEC / "posterior_psd.png"), "Results / descriptive spectral validation", "Grand-average posterior PSD curves for the six experimental conditions.", "The curves describe condition-level spectral shape only; no significance can be inferred from visual separation.", "Main text"],
        ["Figure 4.2", "participant_power.png", str(SPEC / "participant_power.png"), "Results / descriptive distributions", "Participant distributions of theta, alpha, and beta relative power.", "Between-participant variability is substantial and supports displaying individual observations alongside group summaries.", "Supplement or early Results"],
        ["Figure 4.3", "topographic_power.png", str(SPEC / "topographic_power.png"), "Results / channel-level descriptive QC", "Channel-wise topographic summaries of relative band power.", "The maps are descriptive channel summaries and were not used to define new data-driven ROIs.", "Supplement"],
        ["Figure 4.4", "alpha_power.png", str(ROI / "alpha_power.png"), "Results / primary alpha analysis", "Alpha relative power across six conditions in Occipital and Parietal ROIs, with participant traces and group mean ± SEM.", "Visible conditions had higher alpha relative power in N=14, but the effect was p=.053 after excluding P007.", "Main text"],
        ["Figure 4.5", "theta_power.png", str(ROI / "theta_power.png"), "Results / primary theta analysis", "Theta relative power across six conditions in Occipital and Parietal ROIs, with participant traces and group mean ± SEM.", "Theta showed Audio, ROI, and Audio×ROI effects; the ROI main effect was sensitivity-dependent at the .05 boundary.", "Main text"],
        ["Figure 4.6", "beta_power.png", str(ROI / "beta_power.png"), "Results / exploratory beta analysis", "Exploratory beta relative power across conditions and ROIs.", "No beta omnibus effect was significant; the plot should remain explicitly exploratory.", "Supplement"],
        ["Figure 4.7", "alpha_theta_summary.png", str(ROI / "alpha_theta_summary.png"), "Results / compact main-effect summary", "Participant-level marginal means for Visibility and Audio in alpha and theta bands.", "The plot summarizes participant variability and marginal patterns, but interaction conclusions must come from the ANOVA and corrected follow-ups.", "Main text or supplement"],
    ], columns=["suggested_figure_number", "figure_file", "absolute_path", "suggested_chapter_section", "caption_topic", "correct_one_sentence_interpretation", "recommended_placement"])
    write_csv(figures, "dissertation_figure_inventory.csv")

    alpha_sig = alpha[(alpha.row_type == "OMNIBUS_ANOVA") & alpha.significant_p_lt_0_05]
    theta_sig = theta[(theta.row_type == "OMNIBUS_ANOVA") & theta.significant_p_lt_0_05]
    theta_post_sig = theta[(theta.row_type == "HOLM_POSTHOC") & theta.significant_p_lt_0_05]
    report = f"""# Dissertation-ready EEG methods, QC, and results summary

## Scope

This package reorganizes existing finalized outputs only. No EEG data were reopened, no preprocessing was changed, and no new statistical test, ROI, band, ERP, or machine-learning analysis was performed.

## Methods summary

The final preprocessing cohort comprised 15 participants. P011 was classified RED because only 94/432 epochs (21.8%) remained and condition counts ranged from 10 to 19; it was excluded from the primary condition-level EEG analysis. The final analysis therefore used N=14, including P007, with an N=13 sensitivity analysis excluding P007.

EEG was sampled at 500 Hz from 16 scalp channels. Stored Fpz was renamed Fp1 to match its documented physical placement. SAGA Average Reference was retained without an additional MNE rereference. Data were filtered 0.1–40 Hz with a zero-phase Hamming FIR; no 50-Hz notch was applied. Image-locked epochs extended from −0.2 to +0.8 s and were baseline corrected from −0.2 to 0 s.

Artifact control used a manually validated synchronized Fp1/Fp2 ocular detector followed by a fixed all-channel >250 µV peak-to-peak rule. The finalized preprocessing records detector PPV of 98.4% and recall of 78.9%; therefore the detector was treated as a high-confidence first rule but not as proof that negative epochs were clean. No ICA or interpolation was used.

Relative theta (4–8 Hz) and alpha (8–13 Hz) were primary spectral outcomes. Relative beta (13–30 Hz) was exploratory. The fixed ROIs were Occipital (O1, Oz, O2) and Parietal (P3, Pz, P4). The primary model was a 2 Visibility × 3 Audio × 2 ROI repeated-measures ANOVA with Greenhouse–Geisser correction where required and partial eta-squared effect sizes.

## Technical validation summary

- Final participants: **15**; primary EEG analysis N: **14**.
- Raw formal image epochs: **6,461**; retained: **3,924**.
- Median retention: **50.0%**; range: **21.8%–92.1%**.
- QC: **3 GREEN, 11 YELLOW, 1 RED**.
- Marker structure: 14/15 participants contributed all 432 image onsets; P005 contributed 413 because 19 image onsets were missing and documented.
- Posterior coverage remained usable for **15/15** participants. No channel was globally marked bad or interpolated.
- Spectral features were complete for all 14 included participants, six conditions, and 16 channels.

## Final alpha results

The only significant alpha omnibus effect was Visibility: **F(1,13)=5.799, p=.032, partial η²=.308**. Visible conditions had higher mean alpha relative power than occluded conditions (paired difference occluded−visible = −0.016; p=.032). The Audio effect required Greenhouse–Geisser correction and was not significant (p=.083); all other alpha main effects and interactions were not significant.

Sensitivity caution: after excluding P007, the Visibility effect retained the same direction and a similar effect size but crossed the conventional threshold: **F(1,12)=4.588, p=.053, partial η²=.277**. The alpha result should therefore be described as significant in the primary N=14 analysis but partially sensitive to P007.

## Final theta results

Significant omnibus effects were:

- Audio: **F(2,26)=4.492, p=.021, partial η²=.257**.
- ROI: **F(1,13)=5.208, p=.040, partial η²=.286**.
- Audio × ROI: **F(2,26)=7.700, p=.002, partial η²=.372**.

Holm-corrected follow-ups supported incongruent > none (p=.042), Occipital > Parietal overall (p=.040), and Occipital > Parietal within congruent (p=.032) and incongruent (p=.042). The corresponding difference within none was not significant (p=.663). The other Audio pairwise comparisons were not significant after Holm correction.

Sensitivity caution: Audio remained significant without P007 (**p=.025, partial η²=.264**) and Audio × ROI remained significant (**p<.001, partial η²=.447**). The ROI main effect retained the same direction and similar size but changed from p=.040 to **p=.058** (partial η²=.268). Thus the core theta Audio and Audio×ROI pattern was stable, while the standalone ROI effect was boundary-sensitive.

## Exploratory beta results

No beta main effect or interaction was significant (all p≥.229). Beta must be reported as exploratory and non-significant.

## Figure inventory

Ten existing figures were judged potentially dissertation-suitable. The inventory CSV provides a suggested number, chapter placement, caption topic, one-sentence interpretation, and original absolute path. PSD and topographic figures are descriptive/technical validation figures; they do not establish condition effects.

## Supported statements

- The finalized preprocessing produced analysis-ready data for 15 participants, with P011 excluded from primary condition-level EEG analysis on transparent RED QC grounds.
- All 14 included participants contributed complete spectral features for all six conditions and 16 channels.
- Visible conditions showed higher alpha relative power in the primary N=14 analysis.
- Theta relative power showed Audio and Audio×ROI effects that remained significant in the N=13 sensitivity analysis.
- Occipital theta relative power exceeded Parietal theta in N=14, with the difference concentrated in congruent and incongruent conditions in Holm-corrected follow-ups.
- Beta analyses were exploratory and non-significant.
- Technical validation, transparent retention reporting, and sensitivity analyses are central strengths of the dataset.

## Writing cautions and unsupported overclaims

- Do not call the alpha Visibility effect robust without qualification: it changed from p=.032 to p=.053 when P007 was excluded.
- Do not call the theta ROI main effect robust without qualification: it changed from p=.040 to p=.058 without P007.
- Do not interpret non-significant pairwise comparisons as evidence that two conditions are equivalent.
- Do not infer condition significance from visual separation in PSD, topographic, or participant-distribution plots.
- Do not claim that the ocular detector captured every ocular artifact; finalized validation recall was 78.9%.
- Do not claim robust causal neural mechanisms, source localization, or region-specific generators from these scalp-level relative-power analyses.
- Do not generalize the beta pattern as a confirmed effect; all beta omnibus tests were non-significant.
- Do not imply that all participants had high retention: the range was 21.8%–92.1%, and multiple included participants were YELLOW.

## Output tables

- `dissertation_methods_summary.csv`: finalized acquisition, preprocessing, artifact, spectral, ROI, and statistical choices.
- `dissertation_technical_validation_summary.csv`: cohort, retention, QC, marker, posterior-channel, and feature-completeness summary.
- `dissertation_alpha_results.csv`: all alpha omnibus effects and N=13 sensitivity results.
- `dissertation_theta_results.csv`: all theta omnibus effects, corrected follow-ups, and N=13 sensitivity results.
- `dissertation_beta_results.csv`: all exploratory beta omnibus effects.
- `dissertation_figure_inventory.csv`: dissertation placement and interpretation guidance for existing figures.
"""
    (OUT / "dissertation_results_summary.md").write_text(report, encoding="utf-8")

    hashes_after = {str(p): sha256(p) for p in source_files}
    if hashes_before != hashes_after:
        raise RuntimeError("A finalized source output changed during summary construction")
    print({"output": str(OUT), "methods_rows": len(methods), "technical_rows": len(technical),
           "alpha_rows": len(alpha), "theta_rows": len(theta), "beta_rows": len(beta),
           "figure_rows": len(figures), "significant_alpha_effects": alpha_sig.effect_or_contrast.tolist(),
           "significant_theta_effects": theta_sig.effect_or_contrast.tolist(),
           "significant_theta_posthocs": theta_post_sig.effect_or_contrast.tolist()})


if __name__ == "__main__":
    main()
