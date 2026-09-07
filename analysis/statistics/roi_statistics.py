from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
EEG_RESULTS = Path(
    os.environ.get("EEG_PROCESSED_ROOT", PROJECT_ROOT / "processed_data" / "eeg")
).resolve()
FEATURE_PATH = EEG_RESULTS / "final_spectral_features" / "channel_features.csv"
QC_PATH = EEG_RESULTS / "final_spectral_features" / "feature_qc.csv"
OUT = Path(os.environ.get("EEG_ROI_OUTPUT", EEG_RESULTS / "final_roi_statistics")).resolve()

PARTICIPANTS = ["P005", "P006", "P007", "P008", "P009", "P010", "P012", "P013", "P015", "P016", "P017", "P018", "P019", "P020"]
CONDITIONS = [
    "visible_none", "visible_congruent", "visible_incongruent",
    "occluded_none", "occluded_congruent", "occluded_incongruent",
]
VISIBILITIES = ["visible", "occluded"]
AUDIO = ["none", "congruent", "incongruent"]
ROIS = {"Occipital": ["O1", "Oz", "O2"], "Parietal": ["P3", "Pz", "P4"]}
BANDS = ["alpha", "theta", "beta"]
OUTCOMES = {b: f"{b}_relative" for b in BANDS}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    specs = {
        "Visibility": ([bin2], [avg3], [avg2]),
        "Audio": ([avg2], aud2, [avg2]),
        "ROI": ([avg2], [avg3], [bin2]),
        "Visibility × Audio": ([bin2], aud2, [avg2]),
        "Visibility × ROI": ([bin2], [avg3], [bin2]),
        "Audio × ROI": ([avg2], aud2, [bin2]),
        "Visibility × Audio × ROI": ([bin2], aud2, [bin2]),
    }
    result = {}
    for name, (vs, aa, rs) in specs.items():
        rows = [np.kron(np.kron(v, a), r) for v in np.atleast_2d(vs) for a in np.atleast_2d(aa) for r in np.atleast_2d(rs)]
        result[name] = np.vstack(rows)
    return result


EFFECT_CONTRASTS = build_effect_contrasts()


def rm_anova(data: pd.DataFrame, outcome: str, cohort: str):
    columns = pd.MultiIndex.from_product([VISIBILITIES, AUDIO, list(ROIS)], names=["visibility", "audio_condition", "roi"])
    pivot = data.pivot(index="participant", columns=["visibility", "audio_condition", "roi"], values=outcome).reindex(columns=columns)
    if pivot.isna().any().any():
        raise ValueError(f"Unbalanced repeated-measures table for {outcome}: {pivot.isna().sum().sum()} missing cells")
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
        mauchly_w = np.nan
        mauchly_chi2 = np.nan
        mauchly_df = np.nan
        mauchly_p = np.nan
        violated = False
        if r > 1:
            s = np.cov(z, rowvar=False, ddof=1)
            tr = float(np.trace(s))
            tr2 = float(np.trace(s @ s))
            det = max(float(np.linalg.det(s)), np.finfo(float).tiny)
            mauchly_w = det / ((tr / r) ** r) if tr > 0 else np.nan
            correction = (n - 1) - (2 * r * r + r + 2) / (6 * r)
            mauchly_chi2 = -correction * np.log(max(mauchly_w, np.finfo(float).tiny))
            mauchly_df = r * (r + 1) / 2 - 1
            mauchly_p = float(stats.chi2.sf(mauchly_chi2, mauchly_df))
            epsilon = float(np.clip((tr * tr) / (r * tr2), 1 / r, 1.0)) if tr2 > 0 else 1.0
            violated = bool(mauchly_p < 0.05)
        df1_gg, df2_gg = df1 * epsilon, df2 * epsilon
        p_gg = float(stats.f.sf(fval, df1_gg, df2_gg))
        p_reported = p_gg if violated else p_unc
        rows.append({
            "cohort": cohort, "outcome": outcome, "n_participants": n, "effect": effect,
            "F": fval, "df1_uncorrected": df1, "df2_uncorrected": df2,
            "p_uncorrected": p_unc, "partial_eta_squared": eta,
            "mauchly_W": mauchly_w, "mauchly_chi_square": mauchly_chi2,
            "mauchly_df": mauchly_df, "mauchly_p": mauchly_p,
            "sphericity_violated_p_lt_0_05": violated,
            "greenhouse_geisser_epsilon": epsilon,
            "df1_reported": df1_gg if violated else df1,
            "df2_reported": df2_gg if violated else df2,
            "p_greenhouse_geisser": p_gg, "p_reported": p_reported,
            "correction_used": "Greenhouse-Geisser" if violated else ("none; sphericity not applicable" if r == 1 else "none; Mauchly p >= .05"),
            "significant_p_lt_0_05": bool(p_reported < 0.05),
        })
    return pd.DataFrame(rows)


def effect_direction(data: pd.DataFrame, outcome: str, effect: str):
    columns = pd.MultiIndex.from_product([VISIBILITIES, AUDIO, list(ROIS)], names=["visibility", "audio_condition", "roi"])
    pivot = data.pivot(index="participant", columns=["visibility", "audio_condition", "roi"], values=outcome).reindex(columns=columns)
    vec = (pivot.to_numpy(float) @ EFFECT_CONTRASTS[effect].T).mean(axis=0)
    if effect == "Visibility":
        label = "occluded > visible" if vec[0] > 0 else "visible > occluded"
    elif effect == "ROI":
        label = "Parietal > Occipital" if vec[0] > 0 else "Occipital > Parietal"
    elif len(vec) == 1:
        label = "positive planned interaction contrast" if vec[0] > 0 else "negative planned interaction contrast"
    else:
        label = "multidimensional contrast vector"
    return vec, label


def paired_test(data, outcome, effect, label, mask_a, mask_b, family):
    a = data.loc[mask_a].groupby("participant")[outcome].mean()
    b = data.loc[mask_b].groupby("participant")[outcome].mean()
    both = a.index.intersection(b.index)
    av, bv = a.loc[both].to_numpy(), b.loc[both].to_numpy()
    diff = av - bv
    t, p = stats.ttest_rel(av, bv)
    return {"outcome": outcome, "triggering_effect": effect, "holm_family": family, "contrast": label,
            "n": len(both), "mean_first": av.mean(), "mean_second": bv.mean(), "mean_difference_first_minus_second": diff.mean(),
            "t": float(t), "df": len(both) - 1, "p_uncorrected": float(p),
            "cohens_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan}


def limited_posthocs(data, outcome, anova):
    sig = set(anova.loc[anova.significant_p_lt_0_05, "effect"])
    rows = []
    if "Visibility" in sig:
        rows.append(paired_test(data, outcome, "Visibility", "occluded - visible", data.visibility.eq("occluded"), data.visibility.eq("visible"), "Visibility"))
    if "Audio" in sig:
        pairs = [("congruent", "none"), ("incongruent", "none"), ("incongruent", "congruent")]
        for x, y in pairs:
            rows.append(paired_test(data, outcome, "Audio", f"{x} - {y}", data.audio_condition.eq(x), data.audio_condition.eq(y), "Audio"))
    if "ROI" in sig:
        rows.append(paired_test(data, outcome, "ROI", "Occipital - Parietal", data.roi.eq("Occipital"), data.roi.eq("Parietal"), "ROI"))
    if "Visibility × Audio" in sig:
        for aud in AUDIO:
            base = data.audio_condition.eq(aud)
            rows.append(paired_test(data, outcome, "Visibility × Audio", f"occluded - visible within {aud}", base & data.visibility.eq("occluded"), base & data.visibility.eq("visible"), "Visibility × Audio simple visibility"))
    if "Visibility × ROI" in sig:
        for roi in ROIS:
            base = data.roi.eq(roi)
            rows.append(paired_test(data, outcome, "Visibility × ROI", f"occluded - visible within {roi}", base & data.visibility.eq("occluded"), base & data.visibility.eq("visible"), "Visibility × ROI simple visibility"))
    if "Audio × ROI" in sig:
        for aud in AUDIO:
            base = data.audio_condition.eq(aud)
            rows.append(paired_test(data, outcome, "Audio × ROI", f"Occipital - Parietal within {aud}", base & data.roi.eq("Occipital"), base & data.roi.eq("Parietal"), "Audio × ROI simple ROI"))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = np.nan
        for _, idx in out.groupby(["outcome", "holm_family"]).groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_uncorrected"])
        out["significant_holm_p_lt_0_05"] = out.p_holm < 0.05
    return out


def plot_band(data, band):
    outcome = OUTCOMES[band]
    labels = ["V-None", "V-Cong", "V-Incong", "O-None", "O-Cong", "O-Incong"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, roi in zip(axes, ROIS):
        sub = data[data.roi.eq(roi)].copy()
        piv = sub.pivot(index="participant", columns="condition", values=outcome).reindex(columns=CONDITIONS)
        x = np.arange(6)
        for _, row in piv.iterrows():
            ax.plot(x, row, color="0.75", alpha=0.55, lw=0.8)
            ax.scatter(x, row, color="0.55", alpha=0.45, s=13)
        means = piv.mean().to_numpy()
        sem = piv.sem().to_numpy()
        ax.errorbar(x, means, yerr=sem, color="#005f73", marker="o", lw=2.4, capsize=4, label="Mean ± SEM")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(f"{roi} ROI")
        ax.set_xlabel("Condition")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(f"{band.capitalize()} relative power")
    axes[1].legend(frameon=False)
    fig.suptitle(f"{band.capitalize()} relative power: participant variability and group mean (N=14)")
    fig.tight_layout()
    fig.savefig(OUT / f"{band}_relative_power_by_condition_roi.png", dpi=180)
    plt.close(fig)


def main_effect_plot(data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, band in enumerate(["alpha", "theta"]):
        outcome = OUTCOMES[band]
        for col, factor in enumerate(["visibility", "audio_condition"]):
            ax = axes[row, col]
            levels = VISIBILITIES if factor == "visibility" else AUDIO
            p = data.groupby(["participant", factor])[outcome].mean().unstack(factor).reindex(columns=levels)
            x = np.arange(len(levels))
            for _, vals in p.iterrows():
                ax.plot(x, vals, color="0.78", lw=0.8, alpha=0.7)
                ax.scatter(x, vals, color="0.55", s=13, alpha=0.5)
            ax.errorbar(x, p.mean(), yerr=p.sem(), color="#9b2226", marker="o", lw=2.4, capsize=4)
            ax.set_xticks(x, levels)
            ax.set_ylabel(f"{band.capitalize()} relative power")
            ax.set_title(f"{band.capitalize()}: {factor.replace('_', ' ').title()} marginal means")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Descriptive main-effect summaries (participant traces; mean ± SEM)")
    fig.tight_layout()
    fig.savefig(OUT / "alpha_theta_summary.png", dpi=180)
    plt.close(fig)


def fmt_p(p):
    return "< .001" if p < .001 else f"= {p:.3f}"


def anova_text(tbl):
    lines = []
    for _, r in tbl.iterrows():
        star = "significant" if r.significant_p_lt_0_05 else "not significant"
        corr = " (GG corrected)" if r.correction_used == "Greenhouse-Geisser" else ""
        lines.append(f"- {r.effect}: F({r.df1_reported:.2f}, {r.df2_reported:.2f}) = {r.F:.3f}, p {fmt_p(r.p_reported)}, partial eta-squared = {r.partial_eta_squared:.3f}; {star}{corr}.")
    return "\n".join(lines)


def main():
    input_hashes_before = {"features": sha256(FEATURE_PATH), "qc": sha256(QC_PATH)}
    features = pd.read_csv(FEATURE_PATH)
    qc = pd.read_csv(QC_PATH)
    if sorted(features.participant.unique()) != sorted(PARTICIPANTS):
        raise ValueError(f"Unexpected participants: {sorted(features.participant.unique())}")
    if "P011" in features.participant.unique():
        raise ValueError("P011 unexpectedly present")
    if set(features.condition.unique()) != set(CONDITIONS):
        raise ValueError("Conditions do not match the six planned conditions")
    if features.channel.nunique() != 16:
        raise ValueError("Expected 16 channels in source feature table")

    OUT.mkdir(parents=True, exist_ok=True)
    roi_map = {ch: roi for roi, chans in ROIS.items() for ch in chans}
    selected = features[features.channel.isin(roi_map)].copy()
    selected["roi"] = selected.channel.map(roi_map)
    qcols = ["participant", "condition", "channel", "extreme_review_flag", "review_reason"]
    selected = selected.merge(qc[qcols], on=["participant", "condition", "channel"], how="left", validate="one_to_one")
    selected["extreme_review_flag"] = selected.extreme_review_flag.fillna(False).astype(bool)

    feature_cols = ["theta_absolute", "theta_relative", "alpha_absolute", "alpha_relative", "beta_absolute", "beta_relative", "total_1_40_power"]
    roi_rows = []
    influence_rows = []
    keys = ["participant", "condition", "visibility", "audio_condition", "roi"]
    for key, g in selected.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row["channels_averaged"] = ",".join(ROIS[row["roi"]])
        row["n_channels_averaged"] = len(g)
        row["n_epochs"] = int(g.n_epochs.min())
        row["n_flagged_source_channels"] = int(g.extreme_review_flag.sum())
        row["flagged_source_channels"] = ",".join(g.loc[g.extreme_review_flag, "channel"].tolist()) or "none"
        for c in feature_cols:
            row[c] = g[c].mean()
        roi_rows.append(row)
        for band in BANDS:
            c = OUTCOMES[band]
            full = float(g[c].mean())
            unflagged = g.loc[~g.extreme_review_flag, c]
            alt = float(unflagged.mean()) if 0 < len(unflagged) < len(g) else np.nan
            pct = 100 * abs(full - alt) / abs(full) if np.isfinite(alt) and full != 0 else np.nan
            influence_rows.append({**dict(zip(keys, key)), "band": band, "full_roi_mean": full,
                                   "roi_mean_excluding_flagged_for_diagnostic_only": alt,
                                   "absolute_change": abs(full-alt) if np.isfinite(alt) else np.nan,
                                   "percent_change": pct, "material_influence_gt_20_percent": bool(np.isfinite(pct) and pct > 20),
                                   "n_flagged_channels": int(g.extreme_review_flag.sum()),
                                   "flagged_channels": ",".join(g.loc[g.extreme_review_flag, "channel"].tolist()) or "none"})
    roi = pd.DataFrame(roi_rows).sort_values(["participant", "condition", "roi"]).reset_index(drop=True)
    influence = pd.DataFrame(influence_rows)
    if len(roi) != 14 * 6 * 2 or roi.n_channels_averaged.ne(3).any():
        raise ValueError("ROI aggregation is incomplete")
    roi["primary_include_n14"] = True
    roi["include_in_sensitivity_n13_without_p007"] = roi.participant.ne("P007")
    roi.to_csv(OUT / "roi_power.csv", index=False)

    desc_rows = []
    for band in BANDS:
        col = OUTCOMES[band]
        for (roi_name, cond, vis, aud), g in roi.groupby(["roi", "condition", "visibility", "audio_condition"], sort=False):
            x = g[col]
            desc_rows.append({"band": band, "measure": col, "roi": roi_name, "condition": cond, "visibility": vis,
                              "audio_condition": aud, "N": x.count(), "mean": x.mean(), "SD": x.std(ddof=1),
                              "median": x.median(), "Q1": x.quantile(.25), "Q3": x.quantile(.75),
                              "IQR": x.quantile(.75)-x.quantile(.25)})
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(OUT / "descriptives.csv", index=False)

    primary_results = {}
    for band in BANDS:
        primary_results[band] = rm_anova(roi, OUTCOMES[band], "primary_N14_including_P007")
        primary_results[band].to_csv(OUT / (f"{band}_exploratory_rm_anova.csv" if band == "beta" else f"{band}_rm_anova.csv"), index=False)

    posthoc_parts = [limited_posthocs(roi, OUTCOMES[b], primary_results[b]) for b in BANDS]
    posthoc = pd.concat([x for x in posthoc_parts if not x.empty], ignore_index=True) if any(not x.empty for x in posthoc_parts) else pd.DataFrame()
    if not posthoc.empty:
        posthoc.to_csv(OUT / "posthoc.csv", index=False)

    n13 = roi[roi.participant.ne("P007")].copy()
    sens_rows = []
    for band in ["alpha", "theta"]:
        sens = rm_anova(n13, OUTCOMES[band], "sensitivity_N13_excluding_P007")
        merged = primary_results[band].merge(sens, on=["outcome", "effect"], suffixes=("_N14", "_N13"))
        for _, r in merged.iterrows():
            vec14, label14 = effect_direction(roi, OUTCOMES[band], r.effect)
            vec13, label13 = effect_direction(n13, OUTCOMES[band], r.effect)
            denom = float(np.linalg.norm(vec14) * np.linalg.norm(vec13))
            cosine = float(np.dot(vec14, vec13) / denom) if denom > 0 else np.nan
            direction_evaluable = bool(max(r.partial_eta_squared_N14, r.partial_eta_squared_N13) >= 0.02)
            raw_direction_similar = bool(cosine > 0.8) if len(vec14) > 1 else bool(np.sign(vec14[0]) == np.sign(vec13[0]))
            direction_similar = raw_direction_similar if direction_evaluable else True
            sens_rows.append({"outcome": r.outcome, "effect": r.effect,
                              "N14_F": r.F_N14, "N14_df1_reported": r.df1_reported_N14, "N14_df2_reported": r.df2_reported_N14,
                              "N14_p_reported": r.p_reported_N14, "N14_partial_eta_squared": r.partial_eta_squared_N14,
                              "N14_significant": r.significant_p_lt_0_05_N14,
                              "N13_F": r.F_N13, "N13_df1_reported": r.df1_reported_N13, "N13_df2_reported": r.df2_reported_N13,
                              "N13_p_reported": r.p_reported_N13, "N13_partial_eta_squared": r.partial_eta_squared_N13,
                              "N13_significant": r.significant_p_lt_0_05_N13,
                              "significance_pattern_same": r.significant_p_lt_0_05_N14 == r.significant_p_lt_0_05_N13,
                              "absolute_eta_squared_change": abs(r.partial_eta_squared_N14-r.partial_eta_squared_N13),
                              "N14_effect_direction": label14, "N13_effect_direction": label13,
                              "N14_effect_contrast_vector": json.dumps(vec14.tolist()),
                              "N13_effect_contrast_vector": json.dumps(vec13.tolist()),
                              "effect_vector_cosine_similarity": cosine,
                              "effect_direction_evaluable_eta2_at_least_0_02": direction_evaluable,
                              "effect_direction_similar": direction_similar,
                              "material_eta_squared_change_gt_0_10": abs(r.partial_eta_squared_N14-r.partial_eta_squared_N13) > 0.10,
                              "N13_correction_used": r.correction_used_N13,
                              "N13_greenhouse_geisser_epsilon": r.greenhouse_geisser_epsilon_N13})
    sensitivity = pd.DataFrame(sens_rows)
    sensitivity.to_csv(OUT / "sensitivity_N13.csv", index=False)

    influence.to_csv(OUT / "qc_influence.csv", index=False)
    for band in BANDS:
        plot_band(roi, band)
    main_effect_plot(roi)

    sig_summary = {b: primary_results[b].loc[primary_results[b].significant_p_lt_0_05, "effect"].tolist() for b in BANDS}
    changed = sensitivity.loc[~sensitivity.significance_pattern_same, ["outcome", "effect"]]
    direction_changes = sensitivity.loc[sensitivity.effect_direction_evaluable_eta2_at_least_0_02 & ~sensitivity.effect_direction_similar, ["outcome", "effect"]]
    material_eta_changes = sensitivity.loc[sensitivity.material_eta_squared_change_gt_0_10, ["outcome", "effect"]]
    sensitivity_label = "stable" if changed.empty and direction_changes.empty and material_eta_changes.empty else ("partially sensitive" if len(changed) <= 2 and direction_changes.empty and material_eta_changes.empty else "strongly dependent")
    material = influence[influence.material_influence_gt_20_percent]
    flagged_roi_rows = int((roi.n_flagged_source_channels > 0).sum())
    report = f"""# Final ROI aggregation and preliminary repeated-measures statistics

## Scope and input validation

- Input was limited to the final channel-level spectral feature table; the existing spectral QC table was read only to carry forward review flags.
- Primary cohort: **N = 14**, with P007 retained and P011 absent.
- Sensitivity cohort: **N = 13**, excluding P007.
- All six planned conditions were present for every participant.
- Occipital ROI = O1, Oz, O2; Parietal ROI = P3, Pz, P4. P7/P8 were not used.
- ROI aggregation produced {len(roi)} participant × condition × ROI rows. No source row or final derivative was modified.

## Statistical method

A balanced 2 (Visibility) × 3 (Audio) × 2 (ROI) fully within-participant repeated-measures ANOVA was computed from orthonormal within-subject contrasts. Mauchly's test and Greenhouse–Geisser correction were evaluated for effects containing the three-level Audio factor. Partial eta-squared is reported as the effect size. Alpha and theta were primary outcomes; beta was exploratory. No inferential claim is made beyond this preliminary analysis.

## Alpha relative power

{anova_text(primary_results['alpha'])}

## Theta relative power

{anova_text(primary_results['theta'])}

## Beta relative power — exploratory

{anova_text(primary_results['beta'])}

## Descriptive and post-hoc handling

The descriptive table reports N, mean, SD, median, quartiles, and IQR for every band × ROI × condition. Post-hoc paired tests were generated only for significant omnibus effects that could be decomposed with a small pre-specified family of comparisons; Holm correction was applied within each family. A significant three-way interaction, if present, was not automatically decomposed because doing so would create a large unplanned comparison family.

## P007 sensitivity analysis

- Overall characterization: **{sensitivity_label}**.
- Effects whose significance classification changed after excluding P007: **{len(changed)}**.
{('- None.' if changed.empty else ''.join(f'- {r.outcome}: {r.effect}.\n' for _, r in changed.iterrows()))}
- Effects with a changed/reversed direction: **{len(direction_changes)}**.
- Effects with a partial eta-squared change greater than 0.10: **{len(material_eta_changes)}**.
- Exact N=14 and N=13 F values, corrected/uncorrected reported p values, and partial eta-squared values are side by side in `sensitivity_N13.csv`.

## Existing QC flags and ROI influence

- ROI cells containing at least one previously flagged source channel: **{flagged_roi_rows} of {len(roi)}**.
- Diagnostic leave-flagged-channel-out comparisons exceeding a 20% change: **{len(material)}**.
- These checks are diagnostic only: no channel, participant, condition, or epoch was deleted. The full three-electrode ROI mean remains the inferential value.

## Summary

1. **ROI construction:** complete and balanced for all 14 participants and all six conditions.
2. **Primary tests:** the alpha and theta CSV files contain the complete 2 × 3 × 2 repeated-measures results, including effect sizes and any required Greenhouse–Geisser corrections.
3. **P007:** the analysis is **{sensitivity_label}**. Excluding P007 changed the significance classification of {len(changed)} alpha/theta omnibus effect(s), changed the direction of {len(direction_changes)}, and caused {len(material_eta_changes)} effect-size changes greater than 0.10.
4. **Flagged spectral values:** {len(material)} ROI-band cells showed a >20% diagnostic change when flagged source channels were temporarily omitted. They were retained, not silently removed.
5. **Readiness:** the dataset is ready for careful interpretation of the planned ROI statistics. The sensitivity and QC-influence tables should be consulted alongside any condition-level statement.

## Significant omnibus effects at p < .05

- Alpha: {', '.join(sig_summary['alpha']) if sig_summary['alpha'] else 'none'}
- Theta: {', '.join(sig_summary['theta']) if sig_summary['theta'] else 'none'}
- Beta exploratory: {', '.join(sig_summary['beta']) if sig_summary['beta'] else 'none'}

## Analysis scope

No behavioural analysis, ERP analysis, machine learning, data-driven ROI selection, participant deletion, or source-file modification was performed.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    params = {
        "input_feature_table": str(FEATURE_PATH), "input_qc_table": str(QC_PATH),
        "feature_sha256": input_hashes_before["features"], "qc_sha256": input_hashes_before["qc"],
        "primary_participants": PARTICIPANTS, "excluded_primary": ["P011"],
        "sensitivity_excluded": ["P007"], "conditions": CONDITIONS, "rois": ROIS,
        "outcomes": {"primary": ["alpha_relative", "theta_relative"], "exploratory": ["beta_relative"]},
        "anova": "balanced 2x3x2 fully within-subject ANOVA using orthonormal contrasts",
        "sphericity": "Mauchly test for df>1; Greenhouse-Geisser p and dfs reported when p<0.05",
        "posthoc": "limited paired t-tests following significant interpretable omnibus effects; Holm within family",
        "qc_influence_threshold_percent": 20,
    }
    (OUT / "parameters.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    after = {"features": sha256(FEATURE_PATH), "qc": sha256(QC_PATH)}
    if after != input_hashes_before:
        raise RuntimeError("An input file changed during processing")
    summary = {"roi_rows": len(roi), "descriptive_rows": len(desc), "posthoc_rows": len(posthoc),
               "significant_effects": sig_summary, "sensitivity_significance_changes": len(changed),
               "flagged_roi_cells": flagged_roi_rows, "material_qc_influence_rows": len(material),
               "output": str(OUT)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
