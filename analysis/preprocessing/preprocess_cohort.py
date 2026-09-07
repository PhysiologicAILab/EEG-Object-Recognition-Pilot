"""Final, fixed-protocol preprocessing for the 15-participant EEG cohort.

The original XDF files are opened read-only and hashed before/after processing.
This script intentionally does not apply notch filtering, software rereferencing,
ICA, interpolation, band-power analysis, ROI analysis, or condition statistics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
from scipy import signal

# Allow this script to be run directly from the repository root while importing
# the shared detector implementation from analysis/qc.
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from qc.ocular_validation import (
    CHANNELS, MICROVOLT_UNITS, POSTERIOR, channel_metadata,
    fast_mne_zero_phase_fir, one, parse_markers, trial_table,
)

FINAL_PARTICIPANTS = [
    "P005", "P006", "P007", "P008", "P009", "P010", "P011", "P012",
    "P013", "P015", "P016", "P017", "P018", "P019", "P020",
]
PILOTS = ["P001", "P002", "P003", "P004"]
CONDITIONS = [
    "visible_none", "visible_congruent", "visible_incongruent",
    "occluded_none", "occluded_congruent", "occluded_incongruent",
]
CONDITION_LABELS = {
    "visible_none": "Visible + None",
    "visible_congruent": "Visible + Congruent",
    "visible_incongruent": "Visible + Incongruent",
    "occluded_none": "Occluded + None",
    "occluded_congruent": "Occluded + Congruent",
    "occluded_incongruent": "Occluded + Incongruent",
}
EVENT_ID = {condition: index + 1 for index, condition in enumerate(CONDITIONS)}
SFREQ = 500.0
TMIN, TMAX = -0.2, 0.8
BASELINE = (-0.2, 0.0)
RESIDUAL_THRESHOLD_UV = 250.0

NEIGHBOURS = {
    "Fp1": ["Fp2", "F3", "Fz"], "Fp2": ["Fp1", "F4", "Fz"],
    "F3": ["Fp1", "Fz", "C3"], "Fz": ["Fp1", "Fp2", "F3", "F4", "Cz"], "F4": ["Fp2", "Fz", "C4"],
    "C3": ["F3", "Cz", "P3", "P7"], "Cz": ["Fz", "C3", "C4", "Pz"], "C4": ["F4", "Cz", "P4", "P8"],
    "P7": ["C3", "P3", "O1"], "P3": ["C3", "P7", "Pz", "O1"], "Pz": ["Cz", "P3", "P4", "Oz"],
    "P4": ["C4", "Pz", "P8", "O2"], "P8": ["C4", "P4", "O2"], "O1": ["P7", "P3", "Oz"],
    "Oz": ["Pz", "O1", "O2"], "O2": ["P4", "P8", "Oz"],
}


def robust_sigma(values):
    values = np.asarray(values, float)
    center = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - center)))


def robust_z(values):
    values = np.asarray(values, float)
    scale = robust_sigma(values)
    return (values - np.nanmedian(values)) / scale if scale > 0 else np.zeros_like(values)


def detect_ocular_events(recordings, sfreq):
    """Exact synchronized Fp1/Fp2 detector used in cohort validation."""
    sos = signal.butter(4, 8.0, btype="lowpass", fs=sfreq, output="sos")
    smooth_pairs, scale_values = [], []
    for rec in recordings:
        fp1 = signal.sosfiltfilt(sos, rec["data_uv"][CHANNELS.index("Fp1")])
        fp2 = signal.sosfiltfilt(sos, rec["data_uv"][CHANNELS.index("Fp2")])
        common = (fp1 + fp2) / 2.0
        smooth_pairs.append((fp1, fp2, common)); scale_values.append(common)
    combined = np.concatenate(scale_values)
    center, scale = float(np.median(combined)), robust_sigma(combined)
    events = []
    for rec_index, (fp1, fp2, common) in enumerate(smooth_pairs):
        peaks, _ = signal.find_peaks(np.abs(common - center), height=4*scale, prominence=1.5*scale, distance=int(round(.4*sfreq)))
        edge = int(round(20*sfreq)); peaks = peaks[(peaks >= edge) & (peaks < common.size-edge)]
        keep = (fp1[peaks]-np.median(fp1)) * (fp2[peaks]-np.median(fp2)) > 0
        events.extend({"recording_index": rec_index, "sample": int(p), "time_seconds": float(p/sfreq)} for p in peaks[keep])
    return events, {"robust_sigma_uv": scale, "threshold_uv": 4*scale, "lowpass_hz": 8.0, "minimum_separation_s": .4}


def channel_metrics(participant, recordings, sfreq):
    decimated = np.concatenate([rec["data_uv"][:, ::2] for rec in recordings], axis=1)
    fs = sfreq / 2.0
    median_other = np.stack([np.median(np.delete(decimated, i, axis=0), axis=0) for i in range(len(CHANNELS))])
    freq, psd = signal.welch(decimated, fs=fs, nperseg=min(int(20*fs), decimated.shape[1]), noverlap=min(int(10*fs), max(0, decimated.shape[1]//2-1)), axis=1)
    def power(row, low, high):
        mask = (freq >= low) & (freq <= high)
        return float(np.trapezoid(row[mask], freq[mask]))
    duration_min = decimated.shape[1] / fs / 60
    rows = []
    for i, channel in enumerate(CHANNELS):
        x = decimated[i]; med = float(np.median(x)); mad = robust_sigma(x); q5, q95 = np.percentile(x, [5, 95]); total = power(psd[i], .1, 40)
        extreme = np.abs(x-med) > 8*mad if mad > 0 else np.zeros(x.size, bool)
        starts = np.flatnonzero(extreme & ~np.r_[False, extreme[:-1]])
        neighbour_corrs = [abs(float(np.corrcoef(x, decimated[CHANNELS.index(n)])[0, 1])) for n in NEIGHBOURS[channel]]
        rows.append({
            "participant_id": participant, "channel": channel,
            "region": "posterior" if channel in POSTERIOR else ("frontal" if channel in {"Fp1", "Fp2"} else "other"),
            "std_uv": float(np.std(x)), "mad_sigma_uv": mad, "p5_p95_range_uv": float(q95-q5), "peak_to_peak_uv": float(np.ptp(x)),
            "near_zero_variance": bool(np.std(x)<.5 or q95-q5<1), "near_identical_step_percent": float(100*np.mean(np.abs(np.diff(x))<1e-6)),
            "extreme_transient_events_per_min": float(len(starts)/duration_min),
            "slow_0p1_1_fraction": power(psd[i], .1, 1)/total if total>0 else np.nan,
            "high_30_40_fraction": power(psd[i], 30, 40)/total if total>0 else np.nan,
            "correlation_with_median_other": float(np.corrcoef(x, median_other[i])[0,1]),
            "median_absolute_neighbor_correlation": float(np.median(neighbour_corrs)),
        })
    table = pd.DataFrame(rows)
    metric_cols = ["std_uv", "mad_sigma_uv", "p5_p95_range_uv", "extreme_transient_events_per_min", "slow_0p1_1_fraction", "high_30_40_fraction"]
    for col in metric_cols: table[col+"_robust_z"] = robust_z(table[col])
    statuses, reasons = [], []
    for _, row in table.iterrows():
        evidence=[]; flat=bool(row.near_zero_variance or row.near_identical_step_percent>5)
        high_var=row.std_uv_robust_z>5 or row.mad_sigma_uv_robust_z>5; high_hf=row.high_30_40_fraction_robust_z>5
        high_extreme=row.extreme_transient_events_per_min_robust_z>5
        low_spatial=abs(row.correlation_with_median_other)<.10 and row.median_absolute_neighbor_correlation<.15
        if flat: evidence.append("flat/near-zero variability")
        if high_var: evidence.append("extreme variability outlier")
        if high_hf: evidence.append("high 30-40 Hz contamination outlier")
        if high_extreme: evidence.append("extreme-transient-rate outlier")
        if low_spatial: evidence.append("weak spatial consistency")
        frontal = row.channel in {"Fp1", "Fp2"}
        if flat or (not frontal and low_spatial and sum([high_var, high_hf, high_extreme])>=2) or (frontal and low_spatial and high_hf and high_extreme): status="CANDIDATE BAD"
        elif evidence or row.slow_0p1_1_fraction_robust_z>5:
            status="WATCH"
            if row.slow_0p1_1_fraction_robust_z>5: evidence.append("slow-power outlier")
        else: status="GOOD"
        if frontal and status=="WATCH" and not flat and not high_hf: evidence.append("frontal activity may be ocular rather than electrode failure")
        statuses.append(status); reasons.append("; ".join(evidence) or "No convincing multi-metric abnormality")
    table["status"] = statuses; table["reason"] = reasons
    return table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def discover(root: Path):
    mapping = {}
    for participant in FINAL_PARTICIPANTS + PILOTS:
        paths = sorted((root / f"sub-{participant}").rglob("*.xdf"))
        mapping[participant] = paths
    # P006 has a nested sub-006 filename but lives under sub-P006.
    missing = [p for p in FINAL_PARTICIPANTS if not mapping[p]]
    if missing:
        raise RuntimeError(f"Missing FINAL participant XDFs: {missing}")
    if len([p for p in FINAL_PARTICIPANTS if mapping[p]]) != 15:
        raise RuntimeError("Exactly 15 FINAL participants were not discovered")
    return mapping


def choose_streams(streams, participant, path):
    eegs = [s for s in streams if str(one(s["info"].get("type", [""]))).lower() == "eeg" and "saga" in str(one(s["info"].get("name", [""]))).lower()]
    markers = [s for s in streams if str(one(s["info"].get("type", [""]))).lower() in {"marker", "markers"} and "psychopy" in str(one(s["info"].get("name", [""]))).lower()]
    if len(eegs) != 1 or not markers:
        raise RuntimeError(f"{participant} {path}: SAGA={len(eegs)}, PsychoPyMarkers={len(markers)}")
    # Never merge duplicate marker streams. Choose the stream with the greatest
    # number of parseable formal image-onset markers; ties retain XDF order.
    marker_scores = []
    for index, marker in enumerate(markers):
        score = 0
        for sample in marker.get("time_series", []):
            try:
                item = json.loads(sample[0])
                event = str(item.get("event_name", ""))
                if str(item.get("run_id", "")) in {f"{x:02d}" for x in range(1, 7)} and event.endswith("image_onset"):
                    score += 1
            except Exception:
                pass
        marker_scores.append((score, -index, marker))
    marker_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return eegs[0], marker_scores[0][2], len(markers), marker_scores[0][0]


def load_participant(root: Path, participant: str, paths: list[Path]):
    recordings, all_markers, source_rows = [], [], []
    marker_failures = 0
    for recording_index, path in enumerate(paths):
        before = sha256(path)
        streams, _ = pyxdf.load_xdf(str(path), verbose=False)
        eeg, marker, marker_count, marker_score = choose_streams(streams, participant, path)
        metadata = channel_metadata(eeg)
        selected = [r for r in metadata if r["type"].strip().lower() == "eeg" and r["name"].upper() not in {"TRIGGERS", "STATUS", "COUNTER"}]
        original_names = [r["name"] for r in selected]
        final_names = ["Fp1" if name == "Fpz" else name for name in original_names]
        if final_names != CHANNELS:
            raise RuntimeError(f"{participant} {path}: unexpected channels {final_names}")
        units = {r["unit"].strip().lower() for r in selected}
        if not units or not units.issubset(MICROVOLT_UNITS):
            raise RuntimeError(f"{participant} {path}: EEG units not explicitly µV: {units}")
        sfreq = float(one(eeg["info"].get("nominal_srate", [0])))
        if sfreq != SFREQ:
            raise RuntimeError(f"{participant} {path}: expected {SFREQ} Hz, got {sfreq}")
        timestamps = np.asarray(eeg["time_stamps"], float)
        data_uv = np.asarray(eeg["time_series"][:, [r["index"] for r in selected]].T, np.float64)
        filtered_uv, filter_details = fast_mne_zero_phase_fir(data_uv, sfreq)
        file_run_match = re.search(r"_run-(\d{3})_eeg\.xdf$", path.name, flags=re.I)
        file_run = f"{int(file_run_match.group(1)):02d}" if file_run_match else None
        formal_override = file_run if participant == "P005" and file_run in {"05", "06"} else None
        parsed, failures = parse_markers(marker, recording_index, float(timestamps[0]))
        if formal_override:
            for item in parsed:
                if str(item.get("run_id", "")) in {f"{x:02d}" for x in range(1, 7)}:
                    item["run_id"] = formal_override
        all_markers.extend(parsed)
        marker_failures += failures
        recordings.append({"path": path, "data_uv": filtered_uv, "timestamps": timestamps, "nominal": sfreq})
        after = sha256(path)
        if before != after:
            raise RuntimeError(f"{participant}: source XDF hash changed: {path}")
        source_rows.append({
            "participant": participant, "xdf_path": str(path), "xdf_sha256_before": before,
            "xdf_sha256_after": after, "hash_unchanged": before == after,
            "saga_stream_name": str(one(eeg["info"].get("name", [""]))),
            "selected_marker_stream_name": str(one(marker["info"].get("name", [""]))),
            "psychopy_marker_streams_found": marker_count,
            "selected_stream_formal_image_marker_score": marker_score,
            "xdf_eeg_unit": ";".join(sorted(units)), "unit_conversion_for_derivative": "µV x 1e-6 = V once",
            "nominal_sfreq_hz": sfreq, "original_channel_names": ";".join(original_names),
            "final_channel_names": ";".join(final_names),
        })
        del streams, eeg, marker, data_uv
        gc.collect()
    if marker_failures:
        raise RuntimeError(f"{participant}: {marker_failures} marker JSON failures")
    trials, malformed = trial_table(all_markers)
    if malformed:
        raise RuntimeError(f"{participant}: malformed markers: {malformed[:3]}")
    return recordings, trials.sort_values(["run", "trial_in_run"]).reset_index(drop=True), source_rows, filter_details


def classify_qc(final_n, min_condition, imbalance, posterior_usable, posterior_watch, missing_trials):
    if not posterior_usable or final_n < 120 or min_condition < 15:
        return "RED"
    if final_n < 240 or min_condition < 30 or imbalance > 25 or posterior_watch or missing_trials:
        return "YELLOW"
    return "GREEN"


def standard_1020_positions():
    """Read MNE's bundled standard_1020 ELC file without importing MNE."""
    spec = importlib.util.find_spec("mne")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("MNE package data unavailable; cannot attach standard_1020 coordinates")
    path = Path(list(spec.submodule_search_locations)[0]) / "channels" / "data" / "montages" / "standard_1020.elc"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    n = int(next(line.split("=")[1] for line in lines if line.startswith("NumberPositions")))
    pos_start = lines.index("Positions") + 1
    label_start = lines.index("Labels") + 1
    positions = np.asarray([[float(x) for x in line.split()] for line in lines[pos_start:pos_start+n]], float) / 1000.0
    labels = lines[label_start:label_start+n]
    lookup = dict(zip(labels, positions))
    missing = [ch for ch in CHANNELS if ch not in lookup]
    if missing:
        raise RuntimeError(f"standard_1020 coordinates missing: {missing}")
    return np.stack([lookup[ch] for ch in CHANNELS]), str(path)


def make_epochs_and_logs(participant, recordings, trials, channel_table, output: Path, montage_xyz_m):
    ocular_events, detector = detect_ocular_events(recordings, SFREQ)
    by_recording = defaultdict(list)
    for event in ocular_events:
        by_recording[event["recording_index"]].append(event["sample"])
    for key in by_recording:
        by_recording[key] = np.asarray(sorted(by_recording[key]), int)

    pre, post = int(round(abs(TMIN) * SFREQ)), int(round(TMAX * SFREQ))
    kept_data, kept_metadata, logs = [], [], []
    for _, trial in trials.iterrows():
        base = {
            "participant": participant, "run": str(trial.run), "trial_in_run": int(trial.trial_in_run),
            "global_trial": trial.global_trial, "condition": str(trial.condition),
            "image_onset_count": int(trial.image_onset_count),
        }
        if int(trial.image_onset_count) == 0:
            logs.append({**base, "epoch_created": False, "ocular_rejected": False,
                         "residual_rejected": False, "final_retained": False,
                         "rejection_stage": "MISSING_IMAGE_ONSET", "ocular_events_in_epoch": 0,
                         "max_peak_to_peak_uv": np.nan, "max_peak_to_peak_channel": "",
                         "channels_over_250uv": ""})
            continue
        rec_index = int(trial.recording_index)
        rec = recordings[rec_index]
        center = int(round(float(trial.image_onset_seconds) * SFREQ))
        lo, hi = center - pre, center + post + 1
        if lo < 0 or hi > rec["data_uv"].shape[1]:
            logs.append({**base, "epoch_created": False, "ocular_rejected": False,
                         "residual_rejected": False, "final_retained": False,
                         "rejection_stage": "OUT_OF_BOUNDS", "ocular_events_in_epoch": 0,
                         "max_peak_to_peak_uv": np.nan, "max_peak_to_peak_channel": "",
                         "channels_over_250uv": ""})
            continue
        samples = by_recording.get(rec_index, np.array([], int))
        left, right = np.searchsorted(samples, [lo, hi], side="left")
        ocular_count = int(right - left)
        epoch_uv = rec["data_uv"][:, lo:hi].copy()
        # Baseline correction is used for the same peak-to-peak diagnostic and
        # then applied by MNE to the saved derivative. PTP itself is offset-invariant.
        diagnostic_uv = epoch_uv - epoch_uv[:, : pre + 1].mean(axis=1, keepdims=True)
        channel_p2p = np.ptp(diagnostic_uv, axis=1)
        max_index = int(np.argmax(channel_p2p))
        over = [CHANNELS[i] for i, value in enumerate(channel_p2p) if value > RESIDUAL_THRESHOLD_UV]
        ocular_rejected = ocular_count > 0
        residual_rejected = (not ocular_rejected) and bool(over)
        retained = not ocular_rejected and not residual_rejected
        stage = "OCULAR" if ocular_rejected else ("RESIDUAL_250UV" if residual_rejected else "RETAINED")
        logs.append({**base, "epoch_created": True, "ocular_rejected": ocular_rejected,
                     "residual_rejected": residual_rejected, "final_retained": retained,
                     "rejection_stage": stage, "ocular_events_in_epoch": ocular_count,
                     "max_peak_to_peak_uv": float(channel_p2p[max_index]),
                     "max_peak_to_peak_channel": CHANNELS[max_index],
                     "channels_over_250uv": ";".join(over)})
        if retained:
            kept_data.append(diagnostic_uv * 1e-6)  # single µV -> V conversion; saved data are baseline corrected
            kept_metadata.append(base)

    log = pd.DataFrame(logs)
    if not kept_data:
        raise RuntimeError(f"{participant}: no retained epochs")
    data_v = np.stack(kept_data).astype(np.float32)
    metadata = pd.DataFrame(kept_metadata)
    unknown = sorted(set(metadata.condition) - set(CONDITIONS))
    if unknown:
        raise RuntimeError(f"{participant}: unknown conditions {unknown}")
    candidate_bads = channel_table.loc[channel_table.status == "CANDIDATE BAD", "channel"].tolist()
    participant_dir = output / "derivatives" / f"sub-{participant}"
    participant_dir.mkdir(parents=True, exist_ok=True)
    derivative_path = participant_dir / f"sub-{participant}_task-Default_desc-clean_image_epochs.npz"
    times = np.arange(data_v.shape[-1], dtype=float) / SFREQ + TMIN
    np.savez_compressed(
        derivative_path,
        data_v=data_v,
        times_s=times,
        channel_names=np.asarray(CHANNELS),
        channel_types=np.asarray(["eeg"] * len(CHANNELS)),
        channel_xyz_m=montage_xyz_m,
        sfreq_hz=np.asarray(SFREQ),
        condition=np.asarray(metadata.condition),
        event_code=metadata.condition.map(EVENT_ID).to_numpy(np.int16),
        run=np.asarray(metadata.run),
        trial_in_run=metadata.trial_in_run.to_numpy(np.int16),
        global_trial=pd.to_numeric(metadata.global_trial, errors="coerce").fillna(-1).to_numpy(np.int32),
        unit=np.asarray("V"),
        reference=np.asarray("SAGA Average Reference retained"),
        montage=np.asarray("standard_1020"),
        baseline_s=np.asarray(BASELINE),
        bandpass_hz=np.asarray([0.1, 40.0]),
        bad_channels=np.asarray(candidate_bads),
    )
    metadata_path = participant_dir / f"sub-{participant}_task-Default_desc-clean_image_metadata.csv"
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    with np.load(derivative_path) as check:
        if check["data_v"].shape != data_v.shape or check["channel_names"].tolist() != CHANNELS or float(check["sfreq_hz"]) != SFREQ:
            raise RuntimeError(f"{participant}: saved NPZ verification failed")
        if check["channel_xyz_m"].shape != (16, 3) or str(check["unit"]) != "V":
            raise RuntimeError(f"{participant}: derivative montage/unit verification failed")

    posterior = data_v[:, [CHANNELS.index(ch) for ch in POSTERIOR], :] * 1e6
    freqs, power = signal.welch(posterior, fs=SFREQ, nperseg=500, noverlap=250, axis=-1)
    psd = np.median(power, axis=(0, 1))
    return log, detector, derivative_path, freqs, psd


def condition_rows(log: pd.DataFrame):
    rows = []
    observed = log[log.epoch_created]
    for participant in FINAL_PARTICIPANTS:
        part = observed[observed.participant == participant]
        for condition in CONDITIONS:
            subset = part[part.condition == condition]
            rows.append({
                "participant": participant, "condition_code": condition,
                "condition_label": CONDITION_LABELS[condition],
                "original_epochs": len(subset),
                "ocular_rejected": int(subset.ocular_rejected.sum()),
                "residual_250uv_rejected": int(subset.residual_rejected.sum()),
                "final_retained": int(subset.final_retained.sum()),
            })
    return pd.DataFrame(rows)


def participant_rows(log, channel_qc, derivative_paths):
    rows = []
    for participant in FINAL_PARTICIPANTS:
        all_rows = log[log.participant == participant]
        observed = all_rows[all_rows.epoch_created]
        counts = observed[observed.final_retained].condition.value_counts()
        values = [int(counts.get(c, 0)) for c in CONDITIONS]
        ch = channel_qc[channel_qc.participant_id == participant]
        bad = ch.loc[ch.status == "CANDIDATE BAD", "channel"].tolist()
        watch = ch.loc[ch.status == "WATCH", "channel"].tolist()
        posterior_bad = [c for c in bad if c in POSTERIOR]
        posterior_watch = [c for c in watch if c in POSTERIOR]
        posterior_usable = len(posterior_bad) <= 1
        retained = int(observed.final_retained.sum())
        minimum, maximum = min(values), max(values)
        imbalance = maximum - minimum
        missing = int((all_rows.rejection_stage == "MISSING_IMAGE_ONSET").sum())
        qc = classify_qc(retained, minimum, imbalance, posterior_usable, bool(posterior_watch), missing > 0)
        reasons = []
        if missing: reasons.append(f"{missing} formal trials missing image onset")
        if retained < 120: reasons.append("fewer than 120 retained epochs")
        elif retained < 240: reasons.append("fewer than 240 retained epochs")
        if minimum < 15: reasons.append("minimum condition count below 15")
        elif minimum < 30: reasons.append("minimum condition count below 30")
        if imbalance > 25: reasons.append("condition imbalance greater than 25 trials")
        if posterior_bad: reasons.append("posterior candidate-bad channel(s): " + ",".join(posterior_bad))
        if posterior_watch: reasons.append("posterior watch channel(s): " + ",".join(posterior_watch))
        if not reasons: reasons.append("sufficient retained data, balance, and posterior coverage")
        row = {
            "participant": participant, "original_formal_epochs": len(observed),
            "missing_image_onset_trials": missing,
            "ocular_rejected_epochs": int(observed.ocular_rejected.sum()),
            "residual_250uv_rejected_epochs": int(observed.residual_rejected.sum()),
            "final_retained_epochs": retained,
            "retention_pct": 100 * retained / len(observed) if len(observed) else 0,
            "minimum_condition_count": minimum, "maximum_condition_count": maximum,
            "condition_imbalance": imbalance,
            "bad_channels": ";".join(bad) or "none", "watch_channels": ";".join(watch) or "none",
            "posterior_channels_usable": "YES" if posterior_usable else "NO",
            "posterior_watch_channels": ";".join(posterior_watch) or "none",
            "provisional_qc": qc, "qc_reasons": "; ".join(reasons),
            "epochs_derivative_npz": str(derivative_paths[participant]),
        }
        for condition, value in zip(CONDITIONS, values):
            row[f"retained_{condition}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(participant_qc, condition_qc, psd_curves, output):
    status_colors = {"GREEN": "#2E8B57", "YELLOW": "#D9A300", "RED": "#C44E52"}
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(participant_qc.participant, participant_qc.final_retained_epochs,
                  color=[status_colors[x] for x in participant_qc.provisional_qc])
    for bar, value in zip(bars, participant_qc.final_retained_epochs):
        ax.text(bar.get_x() + bar.get_width()/2, value + 5, str(value), ha="center", fontsize=8)
    ax.set_ylabel("Final retained image epochs")
    ax.set_title("Final cohort retention after ocular + 250 µV residual rejection", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output / "participant_retention.png", dpi=180); plt.close(fig)

    matrix = condition_qc.pivot(index="participant", columns="condition_code", values="final_retained").loc[FINAL_PARTICIPANTS, CONDITIONS]
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(matrix.to_numpy(), cmap="YlGn", aspect="auto", vmin=0, vmax=max(72, int(matrix.to_numpy().max())))
    ax.set_xticks(range(6), [CONDITION_LABELS[c].replace(" + ", "\n") for c in CONDITIONS])
    ax.set_yticks(range(15), FINAL_PARTICIPANTS)
    for i in range(15):
        for j in range(6):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if matrix.iloc[i, j] > 52 else "#1F2937")
    fig.colorbar(image, ax=ax, label="Retained epochs")
    ax.set_title("Final retained trials per condition", fontweight="bold")
    fig.tight_layout(); fig.savefig(output / "condition_counts.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharex=True, sharey=True)
    for ax, participant in zip(axes.flat, FINAL_PARTICIPANTS):
        freqs, psd = psd_curves[participant]
        mask = (freqs >= 1) & (freqs <= 40)
        ax.plot(freqs[mask], 10*np.log10(np.maximum(psd[mask], np.finfo(float).tiny)), color="#3B6EA8", lw=1.5)
        ax.set_title(participant)
        ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    for ax in axes[-1, :-1]: ax.set_xlabel("Hz")
    for ax in axes[:, 0]: ax.set_ylabel("dB µV²/Hz")
    fig.suptitle("Posterior PSD overview of final retained epochs", fontweight="bold")
    fig.tight_layout(); fig.savefig(output / "posterior_psd.png", dpi=180); plt.close(fig)


def report_text(participant_qc, condition_qc, source_qc, channel_qc):
    total_raw = int(participant_qc.original_formal_epochs.sum())
    total_retained = int(participant_qc.final_retained_epochs.sum())
    status = participant_qc.provisional_qc.value_counts()
    condition_totals = condition_qc.groupby("condition_label").final_retained.sum().to_dict()
    p007 = participant_qc.set_index("participant").loc["P007"]
    p011 = participant_qc.set_index("participant").loc["P011"]
    p020 = participant_qc.set_index("participant").loc["P020"]
    proposed_n = int((participant_qc.provisional_qc != "RED").sum())
    participant_lines = []
    for _, row in participant_qc.iterrows():
        participant_lines.append(
            f"| {row.participant} | {row.original_formal_epochs} | {row.ocular_rejected_epochs} | "
            f"{row.residual_250uv_rejected_epochs} | {row.final_retained_epochs} | {row.retention_pct:.1f}% | "
            f"{row.minimum_condition_count}–{row.maximum_condition_count} | {row.provisional_qc} | {row.qc_reasons} |"
        )
    paths = "\n".join(f"- {p}: " + " | ".join(source_qc[source_qc.participant == p].xdf_path.tolist()) for p in FINAL_PARTICIPANTS)
    conditions = "\n".join(f"- {label}: {int(condition_totals.get(label, 0))}" for label in CONDITION_LABELS.values())
    bads = channel_qc[channel_qc.status == "CANDIDATE BAD"].groupby("participant_id").channel.apply(lambda x: ", ".join(x)).to_dict()
    return f"""# Final preprocessing and cohort QC report

## 1. Cohort rediscovery

Exactly **15 FINAL-protocol participants** were processed: {', '.join(FINAL_PARTICIPANTS)}.

P001–P004 were treated as pilot/protocol-development participants and excluded. P014 is absent from the recorded session identifiers, and all original identifiers were retained. P020 is the fifteenth recorded participant/session label.

### XDF paths

{paths}

## 2. Fixed preprocessing protocol

- Imported SAGA EEG and one selected PsychoPyMarkers stream per XDF; duplicate streams were never merged.
- EEG channel list: {', '.join(CHANNELS)}. Stored Fpz was renamed to Fp1 because of its established physical placement.
- XDF EEG units were explicitly microvolts. Retained baseline-corrected epochs were multiplied by 1e-6 exactly once for derivatives stored in volts.
- Standard montage: `standard_1020`; all 16 channel locations were recognised and their XYZ coordinates are embedded in every derivative.
- Filter: 0.1–40 Hz, zero-phase Hamming FIR, 16,501 samples at 500 Hz, MNE-equivalent coefficients/reflect-limited padding.
- No 50-Hz notch, software rereference, zero-filled reference channel, ICA, or interpolation. SAGA Average Reference retained.
- Epochs: image onset -0.2 to +0.8 s; baseline -0.2 to 0 s.
- First rejection: the fixed, manually blind-validated synchronized Fp1/Fp2 ocular detector (PPV 98.4%, recall 78.9%). Detector-negative epochs were not assumed clean.
- Second rejection: identical all-channel 250 µV peak-to-peak threshold for every participant. No participant-specific tuning.

## 3. Channel QC

Confirmed candidate-bad channels: **{json.dumps(bads, ensure_ascii=False) if bads else 'none'}**.

Watch channels remain documented in `channel_qc.csv`; frontal ocular activity alone did not cause Fp1/Fp2 to be labelled bad. No channel was interpolated. Posterior usability was assessed from P7, P3, Pz, P4, P8, O1, Oz, O2 and is listed participant-by-participant.

## 4. Participant results

| Participant | Original | Ocular rejected | 250 µV rejected | Retained | Retention | Condition range | QC | Main reason |
|---|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(participant_lines)}

The QC labels are provisional and combine retained total, minimum condition count, imbalance, and posterior quality. Screening guide: RED for unusable posterior coverage, <120 retained epochs, or <15 in any condition; YELLOW for <240 retained, <30 in any condition, imbalance >25, posterior WATCH channels, or documented missing image onsets. These rules are transparent review aids, not silent automatic exclusions.

## 5. Requested participant decisions

- **P007:** {p007.provisional_qc}; retained {p007.final_retained_epochs}/{p007.original_formal_epochs} epochs, minimum condition count {p007.minimum_condition_count}. It remains {'provisionally usable with caution and sensitivity analysis' if p007.provisional_qc != 'RED' else 'not sufficiently supported for condition-level analysis'}.
- **P011:** {p011.provisional_qc}; retained {p011.final_retained_epochs}/{p011.original_formal_epochs} epochs, minimum condition count {p011.minimum_condition_count}. {'Participant-level exclusion from condition EEG analysis is recommended.' if p011.provisional_qc == 'RED' else 'It is not automatically excluded, but requires review.'}
- **P020 (new participant):** {p020.provisional_qc}; retained {p020.final_retained_epochs}/{p020.original_formal_epochs}, condition range {p020.minimum_condition_count}–{p020.maximum_condition_count}. It {'passes provisional QC' if p020.provisional_qc != 'RED' else 'does not pass provisional condition-level QC'}.

## 6. Cohort summary

- FINAL participants: **15**
- Raw formal image epochs available: **{total_raw}**
- Final retained epochs: **{total_retained}**
- Cohort median retention: **{participant_qc.retention_pct.median():.1f}%**
- Retention range: **{participant_qc.retention_pct.min():.1f}%–{participant_qc.retention_pct.max():.1f}%**
- QC counts: GREEN={int(status.get('GREEN', 0))}, YELLOW={int(status.get('YELLOW', 0))}, RED={int(status.get('RED', 0))}
- Proposed N for condition-level EEG analysis: **{proposed_n}**, excluding only provisional RED participants pending investigator confirmation.

### Retained trials by condition across all 15 participants

{conditions}

## 7. Derivatives and provenance

Each participant has a compressed analysis derivative (`.npz`) plus trial metadata in `derivatives/sub-Pxxx/`. The derivative contains filtered, baseline-corrected retained epochs in volts, exact time points, condition/event codes, all 16 standard_1020 coordinates, channel types, reference, and fixed preprocessing metadata. The complete trial-level reasons are in `rejection_log.csv`; triggering channels for 250 µV rejection are recorded.

The system MNE import was pathologically slow in this environment, so FIF export was not allowed to block cohort processing. The NPZ derivatives are fully numerical and analysis-ready; a later format-conversion step can create FIF without repeating preprocessing. This limitation does not change any epoch, rejection, condition count, or QC result.

Every source XDF SHA-256 was identical before and after read-only processing. `parameters.json` records the fixed parameters and software versions. No final band power, ROIs, condition statistics, machine learning, or neuroscience interpretation was performed.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = discover(args.root)

    all_logs, all_channels, all_sources = [], [], []
    derivative_paths, psd_curves, detector_parameters = {}, {}, {}
    filter_details_final = None
    montage_xyz_m, montage_source = standard_1020_positions()
    pd.DataFrame({"channel": CHANNELS, "x_m": montage_xyz_m[:, 0], "y_m": montage_xyz_m[:, 1], "z_m": montage_xyz_m[:, 2]}).to_csv(
        args.output / "channel_positions.csv", index=False, encoding="utf-8-sig"
    )
    for participant in FINAL_PARTICIPANTS:
        print(f"PROCESSING {participant} ({len(mapping[participant])} XDF)", flush=True)
        recordings, trials, source_rows, filter_details = load_participant(args.root, participant, mapping[participant])
        filter_details_final = filter_details
        channels = channel_metrics(participant, recordings, SFREQ)
        log, detector, derivative_path, freqs, psd = make_epochs_and_logs(participant, recordings, trials, channels, args.output, montage_xyz_m)
        all_logs.append(log); all_channels.append(channels); all_sources.extend(source_rows)
        derivative_paths[participant] = derivative_path; psd_curves[participant] = (freqs, psd); detector_parameters[participant] = detector
        print(f"  observed={int(log.epoch_created.sum())}, ocular={int(log.ocular_rejected.sum())}, residual={int(log.residual_rejected.sum())}, retained={int(log.final_retained.sum())}", flush=True)
        del recordings, trials
        gc.collect()

    rejection_log = pd.concat(all_logs, ignore_index=True)
    channel_qc = pd.concat(all_channels, ignore_index=True)
    source_qc = pd.DataFrame(all_sources)
    condition_qc = condition_rows(rejection_log)
    participant_qc = participant_rows(rejection_log, channel_qc, derivative_paths)

    participant_qc.to_csv(args.output / "participant_qc.csv", index=False, encoding="utf-8-sig")
    rejection_log.to_csv(args.output / "rejection_log.csv", index=False, encoding="utf-8-sig")
    condition_qc.to_csv(args.output / "condition_counts.csv", index=False, encoding="utf-8-sig")
    channel_qc.to_csv(args.output / "channel_qc.csv", index=False, encoding="utf-8-sig")
    source_qc.to_csv(args.output / "xdf_sources.csv", index=False, encoding="utf-8-sig")

    parameters = {
        "protocol": "FINAL", "participants": FINAL_PARTICIPANTS, "excluded_pilots": PILOTS,
        "channel_names": CHANNELS, "rename": {"Fpz": "Fp1"}, "montage": "standard_1020", "montage_source_file": montage_source,
        "source_unit": "microvolts (explicit XDF metadata)", "mne_unit": "volts",
        "conversion": "multiply retained baseline-corrected epochs by 1e-6 exactly once before derivative storage",
        "sfreq_hz": SFREQ, "bandpass_hz": [0.1, 40.0], "notch": None,
        "reference": "SAGA Average Reference retained", "software_rereference": False,
        "zero_filled_reference": False, "ica": False, "interpolation": False,
        "epoch_tmin_s": TMIN, "epoch_tmax_s": TMAX, "baseline_s": list(BASELINE),
        "ocular_detector": {
            "definition": "synchronized Fp1/Fp2, 8-Hz low-pass, 4 robust-sigma height, 1.5-sigma prominence, 0.40-s separation, same polarity, 20-s recording edges excluded",
            "manual_validation_precision_pct": 98.4, "manual_validation_recall_pct": 78.9,
            "participant_parameters": detector_parameters,
        },
        "residual_rejection": {"measure": "max peak-to-peak over every EEG channel", "threshold_uv": 250.0, "operator": ">"},
        "qc_screening": {
            "red": "posterior unusable OR final epochs <120 OR minimum condition <15",
            "yellow": "not RED and final <240 OR minimum condition <30 OR imbalance >25 OR posterior watch OR missing image onset",
            "green": "otherwise",
        },
        "filter_implementation": filter_details_final,
        "derivative_format": "compressed NPZ, float32 volts, with standard_1020 XYZ coordinates and per-participant CSV metadata",
        "mne_fif_export": "not performed because system MNE import was pathologically slow; conversion can be done later without repeating preprocessing",
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "pyxdf": getattr(pyxdf, "__version__", "unknown"), "scipy": signal.__package__},
    }
    (args.output / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plots(participant_qc, condition_qc, psd_curves, args.output)
    (args.output / "report.md").write_text(report_text(participant_qc, condition_qc, source_qc, channel_qc), encoding="utf-8")
    print("\nFINAL PARTICIPANT QC\n" + participant_qc.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
