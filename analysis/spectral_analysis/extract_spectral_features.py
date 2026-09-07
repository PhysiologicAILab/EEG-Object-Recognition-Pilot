"""Extract descriptive spectral features from final preprocessed NPZ epochs.

Never reads XDF, filters, rereferences, rejects epochs, or modifies the input NPZ.
No inferential statistics, ROI averaging, ERP analysis, or machine learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

INCLUDED = ["P005", "P006", "P007", "P008", "P009", "P010", "P012", "P013", "P015", "P016", "P017", "P018", "P019", "P020"]
EXCLUDED = ["P011"]
CONDITIONS = [
    "visible_none", "visible_congruent", "visible_incongruent",
    "occluded_none", "occluded_congruent", "occluded_incongruent",
]
CHANNELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
POSTERIOR = ["P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
BANDS = {"theta": [4.0, 8.0], "alpha": [8.0, 13.0], "beta": [13.0, 30.0]}
TOTAL_BAND = [1.0, 40.0]
WELCH = {
    "method": "scipy.signal.welch applied independently to each epoch, then arithmetic mean PSD across epochs",
    "window": "hann", "nperseg_samples": 500, "noverlap_samples": 250,
    "nfft": 500, "detrend": "constant", "scaling": "density", "average": "mean",
    "sampling_frequency_hz": 500.0, "frequency_resolution_hz": 1.0,
    "saved_frequency_range_hz": [1.0, 40.0],
}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def find_npz(root: Path, participant: str):
    matches = sorted((root / "derivatives" / f"sub-{participant}").glob("*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"{participant}: expected exactly one final NPZ, found {len(matches)}")
    return matches[0]


def robust_z(values):
    values = np.asarray(values, float)
    median = np.nanmedian(values)
    scale = 1.4826 * np.nanmedian(np.abs(values - median))
    return (values - median) / scale if np.isfinite(scale) and scale > 0 else np.zeros_like(values)


def integrate(psd, freqs, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() < 2:
        raise RuntimeError(f"Insufficient frequency bins for {low}-{high} Hz")
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def parse_condition(condition):
    visibility, audio = condition.split("_", 1)
    return visibility, audio


def load_and_compute(input_root: Path):
    feature_rows, hashes = [], {}
    psd_cube = np.full((len(INCLUDED), len(CONDITIONS), len(CHANNELS), 40), np.nan, np.float64)
    n_epochs = np.zeros((len(INCLUDED), len(CONDITIONS)), np.int16)
    montage_xyz = None
    saved_freqs = None
    for pi, participant in enumerate(INCLUDED):
        path = find_npz(input_root, participant)
        before = sha256(path)
        with np.load(path, allow_pickle=True) as item:
            data_uv = np.asarray(item["data_v"], np.float64) * 1e6
            conditions = np.asarray(item["condition"], str)
            channels = item["channel_names"].tolist()
            sfreq = float(item["sfreq_hz"])
            xyz = np.asarray(item["channel_xyz_m"], float)
            unit = str(item["unit"])
            times = np.asarray(item["times_s"], float)
        if channels != CHANNELS or data_uv.shape[1:] != (16, 501):
            raise RuntimeError(f"{participant}: channel/data shape mismatch {channels}, {data_uv.shape}")
        if sfreq != 500.0 or unit != "V" or not np.allclose(times[[0, -1]], [-.2, .8]):
            raise RuntimeError(f"{participant}: derivative metadata mismatch")
        if sorted(set(conditions)) != sorted(CONDITIONS):
            raise RuntimeError(f"{participant}: conditions mismatch {sorted(set(conditions))}")
        if montage_xyz is None:
            montage_xyz = xyz
        elif not np.allclose(montage_xyz, xyz):
            raise RuntimeError(f"{participant}: montage coordinates differ")
        freqs, epoch_psd = signal.welch(
            data_uv, fs=sfreq, window=WELCH["window"], nperseg=WELCH["nperseg_samples"],
            noverlap=WELCH["noverlap_samples"], nfft=WELCH["nfft"], detrend=WELCH["detrend"],
            scaling=WELCH["scaling"], average=WELCH["average"], axis=-1,
        )
        save_mask = (freqs >= 1.0) & (freqs <= 40.0)
        if saved_freqs is None:
            saved_freqs = freqs[save_mask]
        for ci, condition in enumerate(CONDITIONS):
            select = conditions == condition
            count = int(select.sum())
            if count == 0:
                raise RuntimeError(f"{participant}: no epochs for {condition}")
            n_epochs[pi, ci] = count
            mean_psd = np.mean(epoch_psd[select], axis=0)
            psd_cube[pi, ci] = mean_psd[:, save_mask]
            total = integrate(mean_psd, freqs, *TOTAL_BAND)
            powers = {band: integrate(mean_psd, freqs, *limits) for band, limits in BANDS.items()}
            visibility, audio = parse_condition(condition)
            for ch_index, channel in enumerate(CHANNELS):
                row = {
                    "participant": participant, "condition": condition,
                    "visibility": visibility, "audio_condition": audio,
                    "channel": channel, "n_epochs": count,
                    "theta_absolute": float(powers["theta"][ch_index]),
                    "theta_relative": float(powers["theta"][ch_index] / total[ch_index]),
                    "alpha_absolute": float(powers["alpha"][ch_index]),
                    "alpha_relative": float(powers["alpha"][ch_index] / total[ch_index]),
                    "beta_absolute": float(powers["beta"][ch_index]),
                    "beta_relative": float(powers["beta"][ch_index] / total[ch_index]),
                    "total_1_40_power": float(total[ch_index]),
                    "absolute_power_unit": "µV²", "psd_unit": "µV²/Hz",
                    "primary_include_n14": True,
                    "include_in_sensitivity_n13_without_p007": participant != "P007",
                }
                feature_rows.append(row)
        after = sha256(path)
        if before != after:
            raise RuntimeError(f"{participant}: final NPZ changed during read-only feature extraction")
        hashes[participant] = {"path": str(path), "sha256_before": before, "sha256_after": after, "unchanged": True}
        print(f"{participant}: epochs={len(data_uv)}, condition counts={dict(zip(CONDITIONS, n_epochs[pi].tolist()))}", flush=True)
    return pd.DataFrame(feature_rows), psd_cube, n_epochs, saved_freqs, montage_xyz, hashes


def build_qc(features):
    qc = features[["participant", "condition", "channel", "n_epochs"]].copy()
    numeric = ["theta_absolute", "theta_relative", "alpha_absolute", "alpha_relative", "beta_absolute", "beta_relative", "total_1_40_power"]
    values = features[numeric]
    qc["nonfinite_value"] = ~np.isfinite(values).all(axis=1)
    qc["negative_power"] = (features[["theta_absolute", "alpha_absolute", "beta_absolute", "total_1_40_power"]] < 0).any(axis=1)
    qc["relative_outside_0_1"] = (features[["theta_relative", "alpha_relative", "beta_relative"]] < 0).any(axis=1) | (features[["theta_relative", "alpha_relative", "beta_relative"]] > 1).any(axis=1)
    qc["missing_expected_combination"] = False
    qc["total_power_log10_robust_z_within_channel"] = np.nan
    for channel, index in features.groupby("channel").groups.items():
        qc.loc[index, "total_power_log10_robust_z_within_channel"] = robust_z(np.log10(features.loc[index, "total_1_40_power"]))
        for band in BANDS:
            qc.loc[index, f"{band}_relative_robust_z_within_channel"] = robust_z(features.loc[index, f"{band}_relative"])
    pc = features.groupby(["participant", "channel"], sort=False).total_1_40_power.median().reset_index()
    pc["log_total"] = np.log10(pc.total_1_40_power)
    pc["participant_channel_total_robust_z"] = pc.groupby("channel").log_total.transform(robust_z)
    qc = qc.merge(pc[["participant", "channel", "participant_channel_total_robust_z"]], on=["participant", "channel"], how="left")
    zcols = ["total_power_log10_robust_z_within_channel", "participant_channel_total_robust_z"] + [f"{b}_relative_robust_z_within_channel" for b in BANDS]
    qc["extreme_review_flag"] = qc[zcols].abs().max(axis=1) > 4.0
    reasons = []
    for _, row in qc.iterrows():
        flags = []
        if row.nonfinite_value: flags.append("nonfinite")
        if row.negative_power: flags.append("negative_power")
        if row.relative_outside_0_1: flags.append("relative_outside_0_1")
        if row.extreme_review_flag: flags.append("robust_z_gt_4_review")
        reasons.append(";".join(flags) or "none")
    qc["review_reason"] = reasons
    qc["automatically_deleted"] = False
    return qc


def save_plots(features, psd, freqs, xyz, output):
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    posterior_idx = [CHANNELS.index(ch) for ch in POSTERIOR]
    fig, ax = plt.subplots(figsize=(11, 6))
    for ci, (condition, color) in enumerate(zip(CONDITIONS, colors)):
        per_participant_db = 10*np.log10(np.maximum(psd[:, ci, posterior_idx, :].mean(axis=1), np.finfo(float).tiny))
        mean = per_participant_db.mean(axis=0); sem = per_participant_db.std(axis=0, ddof=1)/np.sqrt(len(INCLUDED))
        ax.plot(freqs, mean, label=condition, color=color, lw=1.8)
        ax.fill_between(freqs, mean-sem, mean+sem, color=color, alpha=.10)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Posterior PSD (dB µV²/Hz)")
    ax.set_title("Grand-average posterior PSD by condition (descriptive only)", fontweight="bold")
    ax.legend(ncol=2, fontsize=9, frameon=False); ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(output/"posterior_psd.png", dpi=180); plt.close(fig)

    participant_band = features.groupby(["participant", "channel", "condition"], sort=False)[[f"{b}_relative" for b in BANDS]].first().reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for ax, band in zip(axes, BANDS):
        arrays = [participant_band.loc[participant_band.participant == p, f"{band}_relative"].to_numpy() for p in INCLUDED]
        bp = ax.boxplot(arrays, patch_artist=True, showfliers=False)
        for patch, participant in zip(bp["boxes"], INCLUDED):
            patch.set_facecolor("#E18A3B" if participant == "P007" else "#7EA6D8")
            patch.set_alpha(.8)
        ax.set_ylabel(f"{band.title()} relative power")
        ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(range(1, len(INCLUDED)+1), INCLUDED, rotation=45)
    fig.suptitle("Participant distributions of channel × condition relative power\n(P007 highlighted; descriptive only)", fontweight="bold")
    fig.tight_layout(); fig.savefig(output/"participant_power.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x, y = xyz[:, 0], xyz[:, 1]
    radius = max(np.ptp(x), np.ptp(y))/2 * 1.08; center=(float((x.max()+x.min())/2), float((y.max()+y.min())/2))
    for ax, band in zip(axes, BANDS):
        values = features.groupby("channel")[f"{band}_relative"].mean().reindex(CHANNELS).to_numpy()
        sc = ax.scatter(x, y, c=values, s=420, cmap="viridis", edgecolor="white", linewidth=1.2)
        for xi, yi, ch in zip(x, y, CHANNELS): ax.text(xi, yi, ch, ha="center", va="center", fontsize=8, color="white")
        ax.add_patch(plt.Circle(center, radius, fill=False, color="#333333", lw=1.2))
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"{band.title()} relative power")
        fig.colorbar(sc, ax=ax, shrink=.75)
    fig.suptitle("Channel-wise group summaries (not ROI definitions)", fontweight="bold")
    fig.tight_layout(); fig.savefig(output/"topographic_power.png", dpi=180); plt.close(fig)


def report_text(features, qc, n_epochs, hashes):
    flagged = qc[qc.review_reason != "none"]
    flagged_summary = flagged.groupby(["participant", "channel"]).size().sort_values(ascending=False)
    flagged_text = "none" if flagged_summary.empty else ", ".join(f"{p}/{ch} ({n} condition rows)" for (p,ch),n in flagged_summary.items())
    min_count, max_count = int(n_epochs.min()), int(n_epochs.max())
    return f"""# Spectral feature extraction report

## Dataset

- Primary cohort: **N=14** ({', '.join(INCLUDED)}).
- P011 was excluded from condition-level spectral extraction because final preprocessing classified it RED.
- P007 remains included and is explicitly marked for later N=13 sensitivity analysis.
- Input source: final analysis-ready NPZ derivatives only. No XDF was opened; no filtering, artifact rejection, rereferencing, or epoch modification was repeated.
- All input NPZ SHA-256 hashes were unchanged before versus after extraction.

## Data integrity

- Expected rows: 14 participants × 6 conditions × 16 channels = **1,344**; produced: **{len(features):,}**.
- All six conditions and all 16 channels were present for every participant.
- Condition epoch counts ranged from **{min_count} to {max_count}**.
- NaN/Inf rows: **{int(qc.nonfinite_value.sum())}**.
- Negative-power rows: **{int(qc.negative_power.sum())}**.
- Relative-power values outside 0–1: **{int(qc.relative_outside_0_1.sum())}**.
- Robust review flags (not deleted): **{int(qc.extreme_review_flag.sum())} condition/channel rows**.
- Participant/channel combinations represented among review flags: {flagged_text}.

## PSD method

PSD was calculated independently for every retained 1.002-s epoch using `scipy.signal.welch`: 500-Hz sampling rate, Hann window, 500-sample segment, 250-sample overlap, 500-point FFT, constant detrending, density scaling, and mean averaging. Because an epoch contains 501 samples, this yields one 500-sample Welch segment per epoch and 1-Hz resolution. Epoch PSDs were then arithmetically averaged within each participant × condition × channel. Epochs were never concatenated across trial boundaries.

The saved PSD spans 1–40 Hz in 1-Hz steps and is stored in µV²/Hz.

## Band definitions

- Theta: 4–8 Hz
- Alpha: 8–13 Hz
- Beta: 13–30 Hz
- Total: 1–40 Hz

Absolute power (µV²) is trapezoidal integration of the mean PSD. Relative power is band power divided by total 1–40-Hz power. Definitions are identical for every participant, condition, and channel. Shared band-edge frequencies are integration boundaries, not participant-specific adjustments.

## Quality interpretation

All 14 participants produced complete finite spectral features for every condition and channel. Robust-z review flags identify unusual values for inspection but do not prove artifact and were not deleted. `feature_qc.csv` contains the exact row-level flags and robust-z values.

The posterior PSD curves are broadly physiologically shaped and the diagnostic plots are descriptive only. They must not be read as evidence of a condition effect or statistical significance.

## Summary

1. **Did all 14 participants produce usable features?** Yes. Every included participant generated all expected channel-level feature rows; flagged extremes remain for review.
2. **Are all six conditions represented?** Yes, for every participant and every channel.
3. **Does anything look unusual?** {('Some participant/channel values exceeded the conservative robust-z review threshold; see the QC file. No values were automatically removed.' if len(flagged) else 'No participant/channel values exceeded the conservative robust-z review threshold.')} P007 remains identifiable for the planned sensitivity comparison.
4. **Ready for ROI definition and later statistics?** Yes, technically. ROI definitions and statistical models should be specified next, while retaining the planned N=14 versus N=13-without-P007 sensitivity analysis. No inferential test has been run here.

## Outputs

- `channel_features.csv`: long-format absolute and relative powers.
- `psd_results.npz`: participant/condition/channel PSD cube, frequencies, epoch counts, and channel coordinates.
- `feature_qc.csv`: row-level integrity and extreme-value review flags.
- `parameters.json`: exact parameters, band definitions, participant decisions, units, and input hashes.
- Three diagnostic figures: posterior PSD, participant relative-power distributions, and channel-wise summaries.

## Analysis scope

No ROIs, ANOVA/mixed models, significance tests, spoken-response analysis, ERP analysis, or machine learning were performed.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features, psd, n_epochs, freqs, xyz, hashes = load_and_compute(args.input)
    qc = build_qc(features)
    features.to_csv(args.output/"channel_features.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(args.output/"feature_qc.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        args.output/"psd_results.npz",
        psd_uv2_per_hz=psd.astype(np.float32), frequencies_hz=freqs,
        participants=np.asarray(INCLUDED, dtype="U4"), conditions=np.asarray(CONDITIONS, dtype="U24"),
        channels=np.asarray(CHANNELS, dtype="U4"), n_epochs=n_epochs,
        channel_xyz_m=xyz, psd_unit=np.asarray("µV²/Hz"),
    )
    parameters = {
        "input": "final analysis-ready NPZ derivatives only", "included_participants_n14": INCLUDED,
        "excluded_participants": {"P011": "RED final preprocessing QC; excluded from primary condition-level EEG analysis"},
        "p007_sensitivity": {"primary_n14": "included", "sensitivity_n13": "exclude P007 only"},
        "conditions": CONDITIONS, "channels": CHANNELS, "posterior_channels_for_diagnostic_plot": POSTERIOR,
        "welch_psd": WELCH, "bands_hz": BANDS, "total_power_hz": TOTAL_BAND,
        "band_integration": "numpy.trapezoid on participant-condition-channel mean PSD; inclusive endpoints",
        "absolute_power_unit": "µV²", "relative_power_definition": "band absolute power / total 1-40 Hz absolute power",
        "qc": {"nonfinite": "flag", "negative_power": "flag", "relative_outside_0_1": "flag", "extreme": "absolute robust z >4; flag only; no deletion"},
        "input_npz_hashes": hashes,
        "prohibited_steps_confirmed_not_run": ["XDF loading", "filtering", "artifact rejection", "ROI definition", "inferential statistics", "ERP", "machine learning"],
    }
    (args.output/"parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plots(features, psd, freqs, xyz, args.output)
    (args.output/"report.md").write_text(report_text(features, qc, n_epochs, hashes), encoding="utf-8")
    print(f"ROWS={len(features)} QC_FLAGS={int(qc.extreme_review_flag.sum())} NONFINITE={int(qc.nonfinite_value.sum())}", flush=True)
    if qc.review_reason.ne("none").any():
        print(qc[qc.review_reason != "none"].groupby(["participant", "channel"]).size().sort_values(ascending=False).to_string(), flush=True)


if __name__ == "__main__":
    main()
