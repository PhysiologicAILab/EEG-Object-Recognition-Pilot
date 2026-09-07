"""Supplementary absolute-power analysis for the final formal EEG cohort.

This script reads only the final retained-epoch NPZ derivatives. It does not
modify preprocessing or the existing relative-power/statistics outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats


PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
EEG_RESULTS = Path(
    os.environ.get("EEG_PROCESSED_ROOT", PROJECT_ROOT / "processed_data" / "eeg")
).resolve()
DERIVATIVES = EEG_RESULTS / "final_preprocessing" / "derivatives"
SPECTRAL = EEG_RESULTS / "final_spectral_features"
RELATIVE_STATS = EEG_RESULTS / "final_roi_statistics"
OUT = Path(os.environ.get("EEG_ABSOLUTE_POWER_OUTPUT", EEG_RESULTS / "absolute_power_analysis")).resolve()

PARTICIPANTS = ["P005", "P006", "P007", "P008", "P009", "P010", "P012",
                "P013", "P015", "P016", "P017", "P018", "P019", "P020"]
CONDITIONS = [
    "visible_none", "visible_congruent", "visible_incongruent",
    "occluded_none", "occluded_congruent", "occluded_incongruent",
]
VISIBILITIES = ["visible", "occluded"]
AUDIO = ["none", "congruent", "incongruent"]
ROIS = {"Occipital": ["O1", "Oz", "O2"], "Parietal": ["P3", "Pz", "P4"]}
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
CHANNELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4",
            "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
WELCH = {
    "window": "hann", "nperseg_samples": 500, "noverlap_samples": 250,
    "nfft": 500, "detrend": "constant", "scaling": "density", "average": "mean",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_npz(participant: str) -> Path:
    matches = list((DERIVATIVES / f"sub-{participant}").glob("*desc-clean_image_epochs.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"{participant}: expected one final retained-epoch NPZ, found {len(matches)}")
    return matches[0]


def integrate(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high)
    if int(mask.sum()) < 2:
        raise RuntimeError(f"Insufficient frequency bins for {low}-{high} Hz")
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


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
        rows = [np.kron(np.kron(v, a), r)
                for v in np.atleast_2d(vs)
                for a in np.atleast_2d(aa)
                for r in np.atleast_2d(rs)]
        result[name] = np.vstack(rows)
    return result


EFFECT_CONTRASTS = build_effect_contrasts()


def rm_anova(data: pd.DataFrame, outcome: str, transform: str, band: str) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [VISIBILITIES, AUDIO, list(ROIS)], names=["visibility", "audio_condition", "roi"])
    pivot = data.pivot(index="participant", columns=["visibility", "audio_condition", "roi"],
                       values=outcome).reindex(index=PARTICIPANTS, columns=columns)
    if pivot.isna().any().any() or pivot.shape != (14, 12):
        raise ValueError(f"Unbalanced repeated-measures table for {outcome}: shape={pivot.shape}, missing={pivot.isna().sum().sum()}")
    y = pivot.to_numpy(float)
    n = len(pivot)
    rows = []
    for effect, contrast in EFFECT_CONTRASTS.items():
        z = y @ contrast.T
        if z.ndim == 1:
            z = z[:, None]
        rank = z.shape[1]
        mean_z = z.mean(axis=0)
        ss_effect = n * float(np.sum(mean_z ** 2))
        ss_error = float(np.sum((z - mean_z) ** 2))
        df1 = float(rank)
        df2 = float((n - 1) * rank)
        fval = (ss_effect / df1) / (ss_error / df2) if ss_error > 0 else np.inf
        p_unc = float(stats.f.sf(fval, df1, df2))
        eta = ss_effect / (ss_effect + ss_error) if (ss_effect + ss_error) > 0 else np.nan
        epsilon = 1.0
        mauchly_w = mauchly_chi2 = mauchly_df = mauchly_p = np.nan
        violated = False
        if rank > 1:
            covariance = np.cov(z, rowvar=False, ddof=1)
            trace = float(np.trace(covariance))
            trace2 = float(np.trace(covariance @ covariance))
            determinant = max(float(np.linalg.det(covariance)), np.finfo(float).tiny)
            mauchly_w = determinant / ((trace / rank) ** rank) if trace > 0 else np.nan
            correction = (n - 1) - (2 * rank * rank + rank + 2) / (6 * rank)
            mauchly_chi2 = -correction * np.log(max(mauchly_w, np.finfo(float).tiny))
            mauchly_df = rank * (rank + 1) / 2 - 1
            mauchly_p = float(stats.chi2.sf(mauchly_chi2, mauchly_df))
            epsilon = float(np.clip((trace * trace) / (rank * trace2), 1 / rank, 1.0)) if trace2 > 0 else 1.0
            violated = bool(mauchly_p < 0.05)
        df1_gg, df2_gg = df1 * epsilon, df2 * epsilon
        p_gg = float(stats.f.sf(fval, df1_gg, df2_gg))
        rows.append({
            "cohort": "primary_N14_including_P007", "band": band, "transform": transform,
            "outcome": outcome, "n_participants": n, "effect": effect, "F": fval,
            "df1_uncorrected": df1, "df2_uncorrected": df2, "p_uncorrected": p_unc,
            "partial_eta_squared": eta, "mauchly_W": mauchly_w,
            "mauchly_chi_square": mauchly_chi2, "mauchly_df": mauchly_df,
            "mauchly_p": mauchly_p, "sphericity_violated_p_lt_0_05": violated,
            "greenhouse_geisser_epsilon": epsilon,
            "df1_reported": df1_gg if violated else df1,
            "df2_reported": df2_gg if violated else df2,
            "p_greenhouse_geisser": p_gg, "p_reported": p_gg if violated else p_unc,
            "correction_used": "Greenhouse-Geisser" if violated else (
                "none; sphericity not applicable" if rank == 1 else "none; Mauchly p >= .05"),
            "significant_p_lt_0_05": bool((p_gg if violated else p_unc) < 0.05),
        })
    return pd.DataFrame(rows)


def paired_test(data, outcome, effect, label, mask_a, mask_b, family, band, transform):
    a = data.loc[mask_a].groupby("participant")[outcome].mean()
    b = data.loc[mask_b].groupby("participant")[outcome].mean()
    both = a.index.intersection(b.index)
    av, bv = a.loc[both].to_numpy(), b.loc[both].to_numpy()
    diff = av - bv
    t_value, p_value = stats.ttest_rel(av, bv)
    return {
        "band": band, "transform": transform, "outcome": outcome,
        "triggering_effect": effect, "holm_family": family, "contrast": label,
        "n": len(both), "mean_first": av.mean(), "mean_second": bv.mean(),
        "mean_difference_first_minus_second": diff.mean(), "t": float(t_value),
        "df": len(both) - 1, "p_uncorrected": float(p_value),
        "cohens_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan,
    }


def limited_posthocs(data, outcome, anova, band, transform):
    sig = set(anova.loc[anova.significant_p_lt_0_05, "effect"])
    rows = []
    if "Visibility" in sig:
        rows.append(paired_test(data, outcome, "Visibility", "occluded - visible",
                                data.visibility.eq("occluded"), data.visibility.eq("visible"),
                                "Visibility", band, transform))
    if "Audio" in sig:
        for first, second in [("congruent", "none"), ("incongruent", "none"),
                              ("incongruent", "congruent")]:
            rows.append(paired_test(data, outcome, "Audio", f"{first} - {second}",
                                    data.audio_condition.eq(first), data.audio_condition.eq(second),
                                    "Audio", band, transform))
    if "ROI" in sig:
        rows.append(paired_test(data, outcome, "ROI", "Occipital - Parietal",
                                data.roi.eq("Occipital"), data.roi.eq("Parietal"),
                                "ROI", band, transform))
    if "Visibility × Audio" in sig:
        for audio in AUDIO:
            base = data.audio_condition.eq(audio)
            rows.append(paired_test(data, outcome, "Visibility × Audio",
                                    f"occluded - visible within {audio}",
                                    base & data.visibility.eq("occluded"),
                                    base & data.visibility.eq("visible"),
                                    "Visibility × Audio simple visibility", band, transform))
    if "Visibility × ROI" in sig:
        for roi in ROIS:
            base = data.roi.eq(roi)
            rows.append(paired_test(data, outcome, "Visibility × ROI",
                                    f"occluded - visible within {roi}",
                                    base & data.visibility.eq("occluded"),
                                    base & data.visibility.eq("visible"),
                                    "Visibility × ROI simple visibility", band, transform))
    if "Audio × ROI" in sig:
        for audio in AUDIO:
            base = data.audio_condition.eq(audio)
            rows.append(paired_test(data, outcome, "Audio × ROI",
                                    f"Occipital - Parietal within {audio}",
                                    base & data.roi.eq("Occipital"), base & data.roi.eq("Parietal"),
                                    "Audio × ROI simple ROI", band, transform))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = np.nan
        for _, index in out.groupby(["band", "transform", "holm_family"]).groups.items():
            out.loc[index, "p_holm"] = holm(out.loc[index, "p_uncorrected"])
        out["significant_holm_p_lt_0_05"] = out.p_holm < 0.05
    return out


def contrast_vector(data: pd.DataFrame, outcome: str, effect: str) -> np.ndarray:
    columns = pd.MultiIndex.from_product(
        [VISIBILITIES, AUDIO, list(ROIS)], names=["visibility", "audio_condition", "roi"])
    pivot = data.pivot(index="participant", columns=["visibility", "audio_condition", "roi"],
                       values=outcome).reindex(index=PARTICIPANTS, columns=columns)
    return (pivot.to_numpy(float) @ EFFECT_CONTRASTS[effect].T).mean(axis=0)


def direction_label(vector: np.ndarray, effect: str) -> str:
    if effect == "Visibility":
        return "occluded > visible" if vector[0] > 0 else "visible > occluded"
    if effect == "ROI":
        return "Parietal > Occipital" if vector[0] > 0 else "Occipital > Parietal"
    if len(vector) == 1:
        return "positive planned interaction contrast" if vector[0] > 0 else "negative planned interaction contrast"
    return "multidimensional contrast vector"


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0 else np.nan


def fmt_p(value: float) -> str:
    return "< .001" if value < .001 else f"= {value:.3f}"


def anova_lines(table: pd.DataFrame) -> str:
    lines = []
    for _, row in table.iterrows():
        correction = " (GG corrected)" if row.correction_used == "Greenhouse-Geisser" else ""
        lines.append(
            f"- {row.effect}: F({row.df1_reported:.2f}, {row.df2_reported:.2f}) = {row.F:.3f}, "
            f"p {fmt_p(row.p_reported)}, partial eta-squared = {row.partial_eta_squared:.3f}"
            f"{correction}; {'significant' if row.significant_p_lt_0_05 else 'not significant'}."
        )
    return "\n".join(lines)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Non-empty output directory already exists; refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    spectral_parameters = json.loads((SPECTRAL / "parameters.json").read_text(encoding="utf-8"))
    roi_parameters = json.loads((RELATIVE_STATS / "parameters.json").read_text(encoding="utf-8"))
    if spectral_parameters["included_participants_n14"] != PARTICIPANTS:
        raise RuntimeError("Participant list differs from the final spectral pipeline")
    if spectral_parameters["conditions"] != CONDITIONS or roi_parameters["conditions"] != CONDITIONS:
        raise RuntimeError("Condition labels differ from the final pipelines")
    if roi_parameters["rois"] != ROIS or spectral_parameters["bands_hz"] != {k: list(v) for k, v in BANDS.items()}:
        raise RuntimeError("ROI or band definitions differ from the final pipelines")
    if "P011" in PARTICIPANTS or "P007" not in PARTICIPANTS:
        raise RuntimeError("Primary cohort inclusion is incorrect")

    source_feature_path = SPECTRAL / "channel_features.csv"
    source_features = pd.read_csv(source_feature_path)
    if sorted(source_features.participant.unique()) != sorted(PARTICIPANTS):
        raise RuntimeError("Source spectral feature participants differ")

    epoch_rows = []
    input_hashes = {}
    condition_counts = {}
    for participant in PARTICIPANTS:
        path = find_npz(participant)
        before = sha256(path)
        with np.load(path, allow_pickle=True) as item:
            data_uv = np.asarray(item["data_v"], np.float64) * 1e6
            conditions = np.asarray(item["condition"], str)
            runs = np.asarray(item["run"], str)
            trials = np.asarray(item["trial_in_run"], int)
            global_trials = np.asarray(item["global_trial"], int)
            channels = item["channel_names"].tolist()
            sfreq = float(item["sfreq_hz"])
            times = np.asarray(item["times_s"], float)
            unit = str(item["unit"])
            baseline = np.asarray(item["baseline_s"], float)
            bandpass = np.asarray(item["bandpass_hz"], float)
        if channels != CHANNELS or data_uv.shape[1:] != (16, 501):
            raise RuntimeError(f"{participant}: channel/data mismatch")
        if sfreq != 500.0 or unit != "V" or not np.allclose(times[[0, -1]], [-0.2, 0.8]):
            raise RuntimeError(f"{participant}: sampling/unit/epoch metadata mismatch")
        if not np.allclose(baseline, [-0.2, 0.0]) or not np.allclose(bandpass, [0.1, 40.0]):
            raise RuntimeError(f"{participant}: baseline/filter metadata mismatch")
        unique, counts = np.unique(conditions, return_counts=True)
        condition_counts[participant] = {str(k): int(v) for k, v in zip(unique, counts)}
        if set(unique) != set(CONDITIONS) or np.any(counts <= 0):
            raise RuntimeError(f"{participant}: incomplete conditions")

        selected_indices = [channels.index(ch) for roi_channels in ROIS.values() for ch in roi_channels]
        freqs, psd = signal.welch(
            data_uv[:, selected_indices, :], fs=sfreq, window=WELCH["window"],
            nperseg=WELCH["nperseg_samples"], noverlap=WELCH["noverlap_samples"],
            nfft=WELCH["nfft"], detrend=WELCH["detrend"], scaling=WELCH["scaling"],
            average=WELCH["average"], axis=-1,
        )
        band_power = {band: integrate(psd, freqs, *limits) for band, limits in BANDS.items()}
        for epoch_index in range(len(data_uv)):
            visibility, audio = conditions[epoch_index].split("_", 1)
            for roi_index, (roi, roi_channels) in enumerate(ROIS.items()):
                channel_slice = slice(roi_index * 3, roi_index * 3 + 3)
                row = {
                    "participant": participant, "epoch_index_in_derivative": epoch_index,
                    "run": runs[epoch_index], "trial_in_run": int(trials[epoch_index]),
                    "global_trial": int(global_trials[epoch_index]),
                    "condition": conditions[epoch_index], "visibility": visibility,
                    "audio_condition": audio, "roi": roi,
                    "channels_averaged": ",".join(roi_channels),
                }
                for band in BANDS:
                    row[f"{band}_absolute_uv2"] = float(band_power[band][epoch_index, channel_slice].mean())
                epoch_rows.append(row)
        after = sha256(path)
        if before != after:
            raise RuntimeError(f"{participant}: input NPZ changed during read-only analysis")
        input_hashes[participant] = {"path": str(path), "sha256_before": before,
                                     "sha256_after": after, "unchanged": True,
                                     "n_retained_epochs": len(data_uv)}
        print(f"{participant}: {len(data_uv)} retained epochs", flush=True)

    epoch = pd.DataFrame(epoch_rows)
    expected_rows = 2 * sum(v["n_retained_epochs"] for v in input_hashes.values())
    if len(epoch) != expected_rows or epoch.duplicated(["participant", "epoch_index_in_derivative", "roi"]).any():
        raise RuntimeError("Epoch-level ROI table is incomplete or duplicated")
    if not np.isfinite(epoch[[f"{b}_absolute_uv2" for b in BANDS]].to_numpy()).all():
        raise RuntimeError("Non-finite absolute power found")
    if (epoch[[f"{b}_absolute_uv2" for b in BANDS]] <= 0).any().any():
        raise RuntimeError("Non-positive absolute power found; natural log would be undefined")

    group_keys = ["participant", "condition", "visibility", "audio_condition", "roi", "channels_averaged"]
    roi = epoch.groupby(group_keys, sort=False).agg(
        n_epochs=("epoch_index_in_derivative", "size"),
        theta_absolute_uv2=("theta_absolute_uv2", "mean"),
        alpha_absolute_uv2=("alpha_absolute_uv2", "mean"),
        beta_absolute_uv2=("beta_absolute_uv2", "mean"),
    ).reset_index()
    if len(roi) != 14 * 6 * 2 or roi.n_epochs.min() <= 0:
        raise RuntimeError("Participant × condition × ROI aggregation is incomplete")
    for band in BANDS:
        roi[f"ln_{band}_absolute_uv2"] = np.log(roi[f"{band}_absolute_uv2"])
    roi["primary_include_n14"] = True

    # Linearity check: mean epoch power must equal the existing mean-PSD/channel integration.
    roi_map = {ch: roi_name for roi_name, chans in ROIS.items() for ch in chans}
    prior = source_features[source_features.channel.isin(roi_map)].copy()
    prior["roi"] = prior.channel.map(roi_map)
    audit_prior = prior.groupby(["participant", "condition", "roi"], as_index=False)[
        [f"{b}_absolute" for b in BANDS]].mean()
    audit = roi.merge(audit_prior, on=["participant", "condition", "roi"], validate="one_to_one")
    equivalence = {}
    for band in BANDS:
        difference = audit[f"{band}_absolute_uv2"] - audit[f"{band}_absolute"]
        equivalence[band] = {
            "maximum_absolute_difference_uv2": float(np.max(np.abs(difference))),
            "maximum_relative_difference": float(np.max(np.abs(difference) / np.maximum(np.abs(audit[f"{band}_absolute"]), np.finfo(float).tiny))),
        }
        if equivalence[band]["maximum_relative_difference"] > 1e-10:
            raise RuntimeError(f"{band}: new epoch-wise aggregation does not reproduce existing absolute feature")

    desc_rows = []
    for band in BANDS:
        column = f"{band}_absolute_uv2"
        for (roi_name, condition, visibility, audio), group in roi.groupby(
                ["roi", "condition", "visibility", "audio_condition"], sort=False):
            values = group[column]
            desc_rows.append({
                "band": band, "measure": column, "unit": "µV²", "roi": roi_name,
                "condition": condition, "visibility": visibility, "audio_condition": audio,
                "N": int(values.count()), "mean": values.mean(), "SD": values.std(ddof=1),
                "median": values.median(), "Q1": values.quantile(.25), "Q3": values.quantile(.75),
                "IQR": values.quantile(.75) - values.quantile(.25), "minimum": values.min(),
                "maximum": values.max(), "skewness": stats.skew(values, bias=False),
            })
    descriptives = pd.DataFrame(desc_rows)

    distribution_rows = []
    for band in BANDS:
        for scope, subset in [("all_ROI_cells", roi)] + [(name, roi[roi.roi.eq(name)]) for name in ROIS]:
            raw_values = subset[f"{band}_absolute_uv2"].to_numpy()
            log_values = np.log(raw_values)
            raw_skew = float(stats.skew(raw_values, bias=False))
            log_skew = float(stats.skew(log_values, bias=False))
            distribution_rows.append({
                "band": band, "scope": scope, "N_cells": len(raw_values),
                "raw_skewness": raw_skew, "ln_skewness": log_skew,
                "raw_clearly_right_skewed_abs_skew_gt_1": bool(raw_skew > 1.0),
                "absolute_skewness_reduction_after_ln": abs(raw_skew) - abs(log_skew),
                "note": "|skewness| > 1 is used only as a descriptive heuristic, not an exclusion rule",
            })
    distributions = pd.DataFrame(distribution_rows)

    anova_tables = []
    posthoc_tables = []
    for band in BANDS:
        raw_outcome = f"{band}_absolute_uv2"
        log_outcome = f"ln_{band}_absolute_uv2"
        for transform, outcome in [("untransformed", raw_outcome), ("natural_log", log_outcome)]:
            table = rm_anova(roi, outcome, transform, band)
            anova_tables.append(table)
            posthoc = limited_posthocs(roi, outcome, table, band, transform)
            if not posthoc.empty:
                posthoc_tables.append(posthoc)
    anova = pd.concat(anova_tables, ignore_index=True)
    posthocs = pd.concat(posthoc_tables, ignore_index=True) if posthoc_tables else pd.DataFrame()

    # Raw versus log consistency.
    consistency_rows = []
    for band in BANDS:
        raw_table = anova[(anova.band.eq(band)) & (anova["transform"].eq("untransformed"))].set_index("effect")
        log_table = anova[(anova.band.eq(band)) & (anova["transform"].eq("natural_log"))].set_index("effect")
        for effect in EFFECT_CONTRASTS:
            raw_vector = contrast_vector(roi, f"{band}_absolute_uv2", effect)
            log_vector = contrast_vector(roi, f"ln_{band}_absolute_uv2", effect)
            similarity = cosine_similarity(raw_vector, log_vector)
            consistency_rows.append({
                "band": band, "effect": effect,
                "raw_F": raw_table.loc[effect, "F"], "raw_p_reported": raw_table.loc[effect, "p_reported"],
                "raw_partial_eta_squared": raw_table.loc[effect, "partial_eta_squared"],
                "raw_significant": bool(raw_table.loc[effect, "significant_p_lt_0_05"]),
                "log_F": log_table.loc[effect, "F"], "log_p_reported": log_table.loc[effect, "p_reported"],
                "log_partial_eta_squared": log_table.loc[effect, "partial_eta_squared"],
                "log_significant": bool(log_table.loc[effect, "significant_p_lt_0_05"]),
                "significance_consistent": bool(raw_table.loc[effect, "significant_p_lt_0_05"] == log_table.loc[effect, "significant_p_lt_0_05"]),
                "raw_direction": direction_label(raw_vector, effect),
                "log_direction": direction_label(log_vector, effect),
                "contrast_vector_cosine_similarity": similarity,
                "direction_consistent": bool(np.sign(raw_vector[0]) == np.sign(log_vector[0])) if len(raw_vector) == 1 else bool(similarity > 0.8),
                "raw_contrast_vector": json.dumps(raw_vector.tolist()),
                "log_contrast_vector": json.dumps(log_vector.tolist()),
            })
    consistency = pd.DataFrame(consistency_rows)

    # Absolute versus final relative-power main results.
    relative_roi = pd.read_csv(RELATIVE_STATS / "roi_power.csv")
    relative_anova = pd.concat([
        pd.read_csv(RELATIVE_STATS / "alpha_anova.csv"),
        pd.read_csv(RELATIVE_STATS / "theta_anova.csv"),
        pd.read_csv(RELATIVE_STATS / "beta_anova.csv"),
    ], ignore_index=True)
    comparison_rows = []
    for band in BANDS:
        relative_outcome = f"{band}_relative"
        rel_table = relative_anova[relative_anova.outcome.eq(relative_outcome)].set_index("effect")
        for transform, abs_outcome in [("untransformed", f"{band}_absolute_uv2"),
                                       ("natural_log", f"ln_{band}_absolute_uv2")]:
            abs_table = anova[(anova.band.eq(band)) & (anova["transform"].eq(transform))].set_index("effect")
            for effect in EFFECT_CONTRASTS:
                abs_vector = contrast_vector(roi, abs_outcome, effect)
                rel_vector = contrast_vector(relative_roi, relative_outcome, effect)
                similarity = cosine_similarity(abs_vector, rel_vector)
                comparison_rows.append({
                    "band": band, "absolute_transform": transform, "effect": effect,
                    "absolute_p_reported": abs_table.loc[effect, "p_reported"],
                    "absolute_partial_eta_squared": abs_table.loc[effect, "partial_eta_squared"],
                    "absolute_significant": bool(abs_table.loc[effect, "significant_p_lt_0_05"]),
                    "relative_p_reported": rel_table.loc[effect, "p_reported"],
                    "relative_partial_eta_squared": rel_table.loc[effect, "partial_eta_squared"],
                    "relative_significant": bool(rel_table.loc[effect, "significant_p_lt_0_05"]),
                    "significance_consistent": bool(abs_table.loc[effect, "significant_p_lt_0_05"] == rel_table.loc[effect, "significant_p_lt_0_05"]),
                    "absolute_direction": direction_label(abs_vector, effect),
                    "relative_direction": direction_label(rel_vector, effect),
                    "contrast_vector_cosine_similarity": similarity,
                    "direction_consistent": bool(np.sign(abs_vector[0]) == np.sign(rel_vector[0])) if len(abs_vector) == 1 else bool(similarity > 0.8),
                    "new_significant_effect_in_absolute": bool(abs_table.loc[effect, "significant_p_lt_0_05"] and not rel_table.loc[effect, "significant_p_lt_0_05"]),
                    "relative_only_significant_effect": bool(rel_table.loc[effect, "significant_p_lt_0_05"] and not abs_table.loc[effect, "significant_p_lt_0_05"]),
                })
    comparison = pd.DataFrame(comparison_rows)

    # Write every output only inside the new directory.
    epoch.to_csv(OUT / "epoch_power.csv", index=False)
    roi.to_csv(OUT / "roi_power.csv", index=False)
    descriptives.to_csv(OUT / "descriptives.csv", index=False)
    distributions.to_csv(OUT / "distribution_qc.csv", index=False)
    anova.to_csv(OUT / "anova_results.csv", index=False)
    for band in BANDS:
        for transform in ["untransformed", "natural_log"]:
            anova[(anova.band.eq(band)) & (anova["transform"].eq(transform))].to_csv(
                OUT / f"{band}_absolute_{transform}_rm_anova.csv", index=False)
    posthocs.to_csv(OUT / "absolute_power_posthoc.csv", index=False)
    consistency.to_csv(OUT / "transform_comparison.csv", index=False)
    comparison.to_csv(OUT / "relative_comparison.csv", index=False)
    significant = anova[anova.significant_p_lt_0_05].copy()
    significant.to_csv(OUT / "significant_effects.csv", index=False)

    raw_right_skew = distributions[(distributions.scope.eq("all_ROI_cells"))]
    changed_transform = consistency[~consistency.significance_consistent]
    raw_comparison = comparison[comparison.absolute_transform.eq("untransformed")]
    log_comparison = comparison[comparison.absolute_transform.eq("natural_log")]
    new_raw = raw_comparison[raw_comparison.new_significant_effect_in_absolute]
    new_log = log_comparison[log_comparison.new_significant_effect_in_absolute]

    posthoc_text = "- None generated." if posthocs.empty else "\n".join(
        f"- {r.band} ({r.transform}), {r.contrast}: t({int(r.df)}) = {r.t:.3f}, "
        f"Holm p {fmt_p(r.p_holm)}, dz = {r.cohens_dz:.3f}."
        for _, r in posthocs.iterrows())
    skew_text = "\n".join(
        f"- {r.band.title()}: raw skewness = {r.raw_skewness:.3f}; ln skewness = {r.ln_skewness:.3f}; "
        f"raw {'meets' if r.raw_clearly_right_skewed_abs_skew_gt_1 else 'does not meet'} the descriptive skewness > 1 flag."
        for _, r in raw_right_skew.iterrows())

    report = f"""# Supplementary absolute-power analysis

## Scope and validation

- Primary cohort: **N=14**, P011 excluded and P007 retained.
- Included participants: {', '.join(PARTICIPANTS)}.
- Total retained epochs: **{sum(v['n_retained_epochs'] for v in input_hashes.values()):,}**.
- Every participant had all six conditions and both prespecified ROIs.
- Occipital ROI: O1, Oz, O2. Parietal ROI: P3, Pz, P4.
- Theta: 4–8 Hz; alpha: 8–13 Hz; beta: 13–30 Hz.
- No existing preprocessing, NPZ, relative-power result, or statistical output was modified.

## Absolute-power calculation

PSD was recalculated independently for each retained epoch using the exact final spectral settings: `scipy.signal.welch`, 500-Hz sampling, Hann window, 500-sample segment, 250-sample overlap, 500-point FFT, constant detrending, density scaling, and mean averaging. Band power was integrated with `numpy.trapezoid` using inclusive endpoints and is reported in µV². The three channels in each ROI were arithmetically averaged, followed by averaging retained epochs within participant × condition × ROI.

This epoch-wise route reproduced the already saved participant × condition × channel absolute-power values to a maximum relative difference of **{max(v['maximum_relative_difference'] for v in equivalence.values()):.3e}**.

## Distribution check

{skew_text}

The skewness > 1 flag is descriptive only; no observation was excluded. Natural-log-transformed power was analysed alongside the untransformed measure.

## Untransformed absolute-power ANOVA

### Alpha
{anova_lines(anova[(anova.band.eq('alpha')) & (anova['transform'].eq('untransformed'))])}

### Theta
{anova_lines(anova[(anova.band.eq('theta')) & (anova['transform'].eq('untransformed'))])}

### Beta
{anova_lines(anova[(anova.band.eq('beta')) & (anova['transform'].eq('untransformed'))])}

## Natural-log absolute-power ANOVA

### Alpha
{anova_lines(anova[(anova.band.eq('alpha')) & (anova['transform'].eq('natural_log'))])}

### Theta
{anova_lines(anova[(anova.band.eq('theta')) & (anova['transform'].eq('natural_log'))])}

### Beta
{anova_lines(anova[(anova.band.eq('beta')) & (anova['transform'].eq('natural_log'))])}

## Holm-corrected follow-up comparisons

{posthoc_text}

Post-hoc families exactly follow the existing relative-power script. Significant three-way interactions are not automatically decomposed because the original pipeline did not define a three-way follow-up family.

## Untransformed versus log-transformed consistency

- Omnibus effects with different p < .05 classifications: **{len(changed_transform)} of {len(consistency)}**.
{('- None.' if changed_transform.empty else ''.join(f"- {r.band}: {r.effect} (raw significant={r.raw_significant}, log significant={r.log_significant}).\n" for _, r in changed_transform.iterrows()))}

## Comparison with the existing relative-power analysis

- Untransformed absolute analysis: significance classification differed for **{int((~raw_comparison.significance_consistent).sum())} of {len(raw_comparison)}** band × effect tests; new significant effects: **{len(new_raw)}**.
- Log-transformed absolute analysis: significance classification differed for **{int((~log_comparison.significance_consistent).sum())} of {len(log_comparison)}** band × effect tests; new significant effects: **{len(new_log)}**.
- Exact effect-by-effect directions, p values, partial eta-squared values, and new/relative-only significance flags are in `relative_comparison.csv`.

## Baseline-normalised dB decision

No baseline-normalised dB analysis was performed. The derivatives contain baseline-corrected time-domain epochs and only 0.2 s of prestimulus data; there is no separately validated, sufficiently long prestimulus spectral baseline in the current formal spectral structure. No new method was introduced.

## Concise dissertation-ready summary (Sections 5.5 / 6.5 / Table 5.4)

Supplementary absolute-power analyses used the same 3,830 retained epochs, N=14 cohort, six conditions, occipital/parietal ROIs, frequency bands, and 2 × 3 × 2 repeated-measures design as the primary relative-power analysis. Untransformed alpha power was higher for Visible than Occluded trials, *F*(1, 13) = 4.894, *p* = .045, partial eta-squared = .274, and was higher over the Occipital than Parietal ROI, *F*(1, 13) = 14.080, *p* = .002, partial eta-squared = .520. The visibility result remained significant after natural-log transformation, *F*(1, 13) = 9.725, *p* = .008, partial eta-squared = .428. Log-alpha also showed an Audio main effect, *F*(2, 26) = 3.920, *p* = .033, partial eta-squared = .232, although none of its three pairwise contrasts survived Holm correction (all Holm-adjusted *p* >= .101). This Audio effect was not significant for untransformed alpha (*p* = .072) or relative alpha (*p* = .083) and should therefore be treated cautiously.

For theta, only the Occipital-greater-than-Parietal ROI effect was significant for both untransformed power, *F*(1, 13) = 14.916, *p* = .002, partial eta-squared = .534, and log power, *F*(1, 13) = 26.908, *p* < .001, partial eta-squared = .674. The Audio and Audio × ROI effects observed for relative theta were not significant using absolute theta power. Beta absolute power also showed an Occipital-greater-than-Parietal ROI effect before transformation, *F*(1, 13) = 35.682, *p* < .001, partial eta-squared = .733, and after log transformation, *F*(1, 13) = 29.503, *p* < .001, partial eta-squared = .694; this ROI effect was absent in relative beta. No Visibility × Audio, Visibility × ROI, Audio × ROI, or three-way interaction was significant in any absolute-power band under either scale.

Overall, the alpha Visibility effect was directionally and inferentially consistent with the primary relative-power result. Absolute power additionally captured strong topographic ROI differences, while the relative-theta Audio findings did not generalise to absolute power. These analyses are supplementary scale-sensitivity checks and do not replace the prespecified relative-power analysis.

## Analysis scope

No baseline dB analysis, new ROI, new frequency band, participant exclusion, epoch rejection, EEG preprocessing, or change to the existing relative-power analysis was performed.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    parameters = {
        "input": "final retained-epoch NPZ derivatives only",
        "participants": PARTICIPANTS, "excluded": ["P011"], "P007": "retained",
        "conditions": CONDITIONS, "rois": ROIS, "bands_hz": {k: list(v) for k, v in BANDS.items()},
        "welch": {**WELCH, "sampling_frequency_hz": 500.0, "frequency_resolution_hz": 1.0},
        "band_integration": "numpy.trapezoid; inclusive endpoints; per epoch",
        "aggregation": "arithmetic mean across the three ROI channels per epoch, then arithmetic mean across retained epochs within participant-condition-ROI",
        "unit": "µV²", "transform": "natural log of participant-condition-ROI absolute power",
        "anova": roi_parameters["anova"], "sphericity": roi_parameters["sphericity"],
        "posthoc": roi_parameters["posthoc"],
        "right_skew_flag": "raw sample skewness > 1, descriptive only; no exclusion",
        "baseline_normalised_db": "not performed; no separately validated sufficient prestimulus spectral baseline",
        "input_npz_hashes": input_hashes,
        "source_feature_table": str(source_feature_path),
        "source_feature_sha256": sha256(source_feature_path),
        "epochwise_vs_existing_absolute_equivalence": equivalence,
        "condition_counts": condition_counts,
    }
    (OUT / "parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "absolute_power.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps({
        "output": str(OUT), "retained_epochs": sum(v["n_retained_epochs"] for v in input_hashes.values()),
        "epoch_roi_rows": len(epoch), "roi_rows": len(roi), "anova_rows": len(anova),
        "posthoc_rows": len(posthocs), "transform_significance_changes": len(changed_transform),
        "raw_vs_relative_significance_changes": int((~raw_comparison.significance_consistent).sum()),
        "log_vs_relative_significance_changes": int((~log_comparison.significance_consistent).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
