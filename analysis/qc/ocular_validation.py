"""Build a blinded manual-review package for the cohort ocular detector.

This script reads XDF files only. It creates review PDFs and CSV/Markdown
summaries. It does not reject epochs, change the XDF files, run ICA, or save
processed EEG data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pyxdf
from scipy import signal

sys.stdout.reconfigure(encoding="utf-8")

CHANNELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
FRONTAL = ["Fp1", "Fp2"]
POSTERIOR = ["P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
MICROVOLT_UNITS = {"µv", "uv", "µvolt", "microvolt", "microvolts"}
FORMAL_RUNS = {f"{number:02d}" for number in range(1, 7)}
EXPECTED_KEYS = {(f"{run:02d}", trial) for run in range(1, 7) for trial in range(1, 73)}
BASE_ORDER = ["trial_start", "fixation_onset", "image_onset", "image_offset", "silent_delay_onset", "microphone_recording_start", "response_cue_onset", "microphone_recording_stop", "trial_end"]
PARTICIPANTS = ["P008", "P012", "P007", "P011"]
SEED = 20260824
PRE_SECONDS = 0.2
POST_SECONDS = 0.8
PLOT_CHANNELS = FRONTAL + POSTERIOR
PLOT_LAYOUT = ["Fp1", "P7", "P3", "Pz", "P4", "Fp2", "P8", "O1", "Oz", "O2"]


def one(value, default=""):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return default if value is None else value


def channel_metadata(eeg):
    count = int(float(one(eeg["info"].get("channel_count", [0]))))
    try:
        channels = eeg["info"]["desc"][0]["channels"][0]["channel"]
    except Exception:
        channels = []
    rows = []
    for index in range(count):
        item = channels[index] if index < len(channels) else {}
        rows.append(
            {
                "index": index,
                "name": str(one(item.get("label", [f"Ch{index + 1}"]))),
                "type": str(one(item.get("type", [""]))),
                "unit": str(one(item.get("unit", [""]))),
            }
        )
    return rows


def parse_markers(marker, recording_index, eeg_start):
    parsed, failures = [], 0
    for sample, timestamp in zip(marker.get("time_series", []), marker.get("time_stamps", [])):
        try:
            item = json.loads(sample[0])
            item["_original_run_id"] = str(item.get("run_id", ""))
            item["_timestamp"] = float(timestamp)
            item["_recording_index"] = recording_index
            item["_eeg_start"] = eeg_start
            parsed.append(item)
        except Exception:
            failures += 1
    return parsed, failures


def trial_table(markers):
    formal, malformed = [], []
    for item in markers:
        run = str(item.get("run_id", ""))
        if run not in FORMAL_RUNS:
            continue
        try:
            trial = int(item.get("trial_index_run"))
        except Exception:
            malformed.append(f"invalid run/trial on {item.get('event_name')}")
            continue
        formal.append((run, trial, item))
    grouped = defaultdict(list)
    for run, trial, item in formal:
        grouped[(run, trial)].append(item)
    rows = []
    for run, trial in sorted(set(grouped) | EXPECTED_KEYS):
        items = sorted(grouped.get((run, trial), []), key=lambda x: x["_timestamp"])
        events = defaultdict(list)
        for item in items:
            event = str(item.get("event_name", ""))
            event = "image_onset" if event.endswith("_image_onset") else event
            events[event].append(item)
        image_items = events.get("image_onset", [])
        first = items[0] if items else {}
        image = image_items[0] if image_items else None
        rows.append(
            {
                "run": run,
                "trial_in_run": trial,
                "global_trial": first.get("trial_index_global", first.get("global_trial_index", np.nan)),
                "condition": first.get("condition", ""),
                "image_onset_count": len(image_items),
                "image_timestamp": image["_timestamp"] if image else np.nan,
                "recording_index": int(image["_recording_index"]) if image else -1,
                "image_onset_seconds": float(image["_timestamp"] - image["_eeg_start"]) if image else np.nan,
            }
        )
    return pd.DataFrame(rows), malformed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def locate_xdf(root: Path, participant: str) -> Path:
    matches = sorted((root / f"sub-{participant}").rglob("*.xdf"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one XDF for {participant}; found {len(matches)}")
    return matches[0]


def find_impedance_metadata(value, path="info"):
    """Return metadata entries whose key names explicitly mention impedance."""
    hits = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if "imped" in str(key).lower() or "resistan" in str(key).lower():
                rendered = repr(child)
                hits.append({"metadata_path": child_path, "value_preview": rendered[:500]})
            hits.extend(find_impedance_metadata(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(find_impedance_metadata(child, f"{path}[{index}]"))
    return hits


def robust_metrics(x_uv: np.ndarray, sfreq: float):
    x = np.asarray(x_uv, float)
    median = float(np.median(x))
    mad_raw = float(np.median(np.abs(x - median)))
    q5, q95 = np.percentile(x, [5, 95])
    freq, psd = signal.welch(
        x,
        fs=sfreq,
        nperseg=min(int(round(20 * sfreq)), x.size),
        noverlap=min(int(round(10 * sfreq)), max(0, x.size // 2 - 1)),
    )
    slow = (freq >= 0.1) & (freq <= 1.0)
    full = (freq >= 0.1) & (freq <= 40.0)
    slow_power = float(np.trapezoid(psd[slow], freq[slow])) if slow.any() else np.nan
    full_power = float(np.trapezoid(psd[full], freq[full])) if full.any() else np.nan
    return {
        "median_uv": median,
        "std_uv": float(np.std(x)),
        "mad_uv": mad_raw,
        "mad_robust_sigma_uv": 1.4826 * mad_raw,
        "p5_uv": float(q5),
        "p95_uv": float(q95),
        "p5_p95_range_uv": float(q95 - q5),
        "slow_0p1_1_rms_uv": float(math.sqrt(max(slow_power, 0.0))),
        "slow_0p1_1_power_fraction": slow_power / full_power if full_power > 0 else np.nan,
    }


def stable_sample(candidate_indices: np.ndarray, participant: str, label: str, n: int = 20):
    candidate_indices = np.asarray(candidate_indices, int)
    if candidate_indices.size <= n:
        return candidate_indices.copy()
    participant_number = int(participant[1:])
    label_code = 1 if label == "POSITIVE" else 0
    rng = np.random.default_rng(np.random.SeedSequence([SEED, participant_number, label_code]))
    return np.sort(rng.choice(candidate_indices, size=n, replace=False))


def detect_ocular_events_from_front(data_uv: np.ndarray, sfreq: float):
    """Exact cohort detector, specialized to the Fp1/Fp2-only array."""
    sos = signal.butter(4, 8.0, btype="lowpass", fs=sfreq, output="sos")
    fp1 = signal.sosfiltfilt(sos, data_uv[PLOT_CHANNELS.index("Fp1")])
    fp2 = signal.sosfiltfilt(sos, data_uv[PLOT_CHANNELS.index("Fp2")])
    common = (fp1 + fp2) / 2.0
    center = float(np.median(common))
    robust_sigma = float(1.4826 * np.median(np.abs(common - center)))
    detection = np.abs(common - center)
    peaks, _ = signal.find_peaks(
        detection,
        height=4.0 * robust_sigma,
        prominence=1.5 * robust_sigma,
        distance=int(round(0.40 * sfreq)),
    )
    edge = int(round(20 * sfreq))
    peaks = peaks[(peaks >= edge) & (peaks < detection.size - edge)]
    keep = (fp1[peaks] - np.median(fp1)) * (fp2[peaks] - np.median(fp2)) > 0
    peaks = peaks[keep]
    events = [{"recording_index": 0, "sample": int(peak), "time_seconds": float(peak / sfreq)} for peak in peaks]
    return events, {
        "robust_sigma_uv": robust_sigma,
        "threshold_uv": 4.0 * robust_sigma,
        "lowpass_hz": 8.0,
        "minimum_separation_s": 0.40,
    }


def fast_mne_zero_phase_fir(data_uv: np.ndarray, sfreq: float):
    """Apply MNE's standard FIR coefficients with equivalent FFT convolution.

    MNE's public filter_data is unusually slow in this Windows environment.
    This uses the exact filter designed by MNE, MNE's reflect_limited padding,
    and the same group-delay removal as MNE's zero-phase overlap-add routine.
    """
    if sfreq != 500.0:
        raise RuntimeError(f"This exact verified filter construction expects 500 Hz, got {sfreq}")
    # Exact expansion of MNE's firwin design for this 500-Hz 0.1-40 Hz case:
    # auto transitions are 0.1 Hz (lower) and 10 Hz (upper). MNE constructs
    # a centered 165-tap 45-Hz low-pass and subtracts a 16501-tap 0.05-Hz
    # low-pass, both Hamming-window firwin filters.
    h = -signal.firwin(16501, 0.05, window="hamming", pass_zero=True, fs=sfreq)
    upper = signal.firwin(165, 45.0, window="hamming", pass_zero=True, fs=sfreq)
    offset = (len(h) - len(upper)) // 2
    h[offset : offset + len(upper)] += upper
    n_edge = max(min(len(h), data_uv.shape[1]) - 1, 0)
    shift = (len(h) - 1) // 2 + n_edge
    filtered = np.empty_like(data_uv)
    for channel_index, values in enumerate(data_uv):
        padded = np.concatenate(
            [
                2 * values[0] - values[n_edge:0:-1],
                values,
                2 * values[-1] - values[-2 : -n_edge - 2 : -1],
            ]
        )
        convolved = signal.oaconvolve(padded, h, mode="full")
        filtered[channel_index] = convolved[shift : shift + values.size]
    return filtered, {
        "filter_length_samples": int(len(h)),
        "phase": "zero",
        "fir_design": "firwin",
        "fir_window": "hamming",
        "execution": "SciPy FFT overlap-add using MNE coefficients and reflect_limited padding",
    }


def load_participant(root: Path, participant: str):
    path = locate_xdf(root, participant)
    source_hash_before = sha256(path)
    streams, _ = pyxdf.load_xdf(str(path), verbose=False)
    eegs = [
        stream
        for stream in streams
        if str(one(stream["info"].get("type", [""]))).lower() == "eeg"
        and "saga" in str(one(stream["info"].get("name", [""]))).lower()
    ]
    markers = [
        stream
        for stream in streams
        if str(one(stream["info"].get("type", [""]))).lower() in {"marker", "markers"}
        and "psychopy" in str(one(stream["info"].get("name", [""]))).lower()
    ]
    if len(eegs) != 1 or not markers:
        raise RuntimeError(f"{participant}: SAGA={len(eegs)}, PsychoPyMarkers={len(markers)}")
    eeg = eegs[0]
    marker = markers[0]  # Do not merge duplicate marker streams.
    metadata = channel_metadata(eeg)
    selected = [
        row
        for row in metadata
        if row["type"].strip().lower() == "eeg"
        and row["name"].upper() not in {"TRIGGERS", "STATUS", "COUNTER"}
    ]
    original_names = [row["name"] for row in selected]
    final_names = ["Fp1" if name == "Fpz" else name for name in original_names]
    if final_names != CHANNELS:
        raise RuntimeError(f"{participant}: unexpected EEG channels after Fpz->Fp1: {final_names}")
    units = {row["unit"].strip().lower() for row in selected}
    if not units or not units.issubset(MICROVOLT_UNITS):
        raise RuntimeError(f"{participant}: EEG units are not explicitly microvolts: {units}")
    sfreq = float(one(eeg["info"].get("nominal_srate", [0])))
    timestamps = np.asarray(eeg["time_stamps"], float)
    metadata_by_final_name = {final: row for final, row in zip(final_names, selected)}
    plot_indices = [metadata_by_final_name[channel]["index"] for channel in PLOT_CHANNELS]
    data_uv = np.asarray(eeg["time_series"][:, plot_indices].T, dtype=np.float64)

    # Use the same in-memory 0.1-40 Hz, zero-phase FIR detector input as the
    # cohort QC; it is never saved.
    data_uv, filter_details = fast_mne_zero_phase_fir(data_uv, sfreq)
    parsed, failures = parse_markers(marker, 0, float(timestamps[0]))
    trials, malformed = trial_table(parsed)
    trials = trials[trials.image_onset_count > 0].sort_values(["run", "trial_in_run"]).reset_index(drop=True)
    if len(trials) != 432:
        raise RuntimeError(f"{participant}: expected 432 image epochs; found {len(trials)}")

    ocular_events, detector = detect_ocular_events_from_front(data_uv, sfreq)
    event_samples = np.asarray(sorted(event["sample"] for event in ocular_events), int)
    pre = int(round(PRE_SECONDS * sfreq))
    post = int(round(POST_SECONDS * sfreq))
    epoch_records = []
    epoch_arrays = []
    for trial_index, trial in trials.iterrows():
        center = int(round(float(trial.image_onset_seconds) * sfreq))
        lo, hi = center - pre, center + post + 1
        if lo < 0 or hi > data_uv.shape[1]:
            raise RuntimeError(f"{participant}: out-of-bounds image epoch at trial row {trial_index}")
        left, right = np.searchsorted(event_samples, [lo, hi], side="left")
        positive = bool(right > left)
        epoch_records.append(
            {
                "trial_row": int(trial_index),
                "run": str(trial.run),
                "trial_in_run": int(trial.trial_in_run),
                "global_trial": int(float(trial.global_trial)),
                "condition": str(trial.condition),
                "algorithm_label": "POSITIVE" if positive else "NEGATIVE",
            }
        )
        # Epochs are baseline-corrected only for display, matching the earlier
        # diagnostic epoch sanity check. No source or continuous data is changed.
        epoch = data_uv[:, lo:hi].copy()
        epoch -= np.mean(epoch[:, : pre + 1], axis=1, keepdims=True)
        epoch_arrays.append(epoch)
    epochs = np.stack(epoch_arrays)
    epoch_table = pd.DataFrame(epoch_records)

    positive_indices = np.flatnonzero(epoch_table.algorithm_label.to_numpy() == "POSITIVE")
    negative_indices = np.flatnonzero(epoch_table.algorithm_label.to_numpy() == "NEGATIVE")
    selected_indices = np.concatenate(
        [
            stable_sample(positive_indices, participant, "POSITIVE"),
            stable_sample(negative_indices, participant, "NEGATIVE"),
        ]
    )
    sampled_epochs = epochs[selected_indices]
    sampled_table = epoch_table.iloc[selected_indices].copy().reset_index(drop=True)

    decimation = 2
    fp = data_uv[[PLOT_CHANNELS.index("Fp1"), PLOT_CHANNELS.index("Fp2")], ::decimation]
    metric_sfreq = sfreq / decimation
    fp_corr = float(np.corrcoef(fp[0], fp[1])[0, 1])
    impedance_hits = find_impedance_metadata(eeg["info"])
    characteristic_rows = []
    for channel_index, channel in enumerate(FRONTAL):
        characteristic_rows.append(
            {
                "participant": participant,
                "channel": channel,
                "analysis_data": "0.1-40 Hz filtered diagnostic copy; no notch; SAGA reference retained",
                **robust_metrics(fp[channel_index], metric_sfreq),
                "fp1_fp2_correlation": fp_corr,
                "detector_robust_sigma_uv": detector["robust_sigma_uv"],
                "detector_threshold_uv": detector["threshold_uv"],
                "detector_session_event_count": len(ocular_events),
                "detector_positive_image_epochs": int((epoch_table.algorithm_label == "POSITIVE").sum()),
                "detector_negative_image_epochs": int((epoch_table.algorithm_label == "NEGATIVE").sum()),
                "impedance_metadata_present": "YES" if impedance_hits else "NO",
                "impedance_metadata_note": json.dumps(impedance_hits, ensure_ascii=False) if impedance_hits else "No explicit impedance/resistance field found in SAGA XDF stream metadata",
            }
        )

    source_hash_after = sha256(path)
    if source_hash_before != source_hash_after:
        raise RuntimeError(f"{participant}: source XDF hash changed during read-only processing")
    metadata_summary = {
        "participant": participant,
        "xdf_path": str(path),
        "source_sha256": source_hash_before,
        "sfreq_hz": sfreq,
        "original_channels": original_names,
        "final_channels": final_names,
        "xdf_units": sorted(units),
        "marker_streams_found": len(markers),
        "selected_marker_stream": str(one(marker["info"].get("name", [""]))),
        "marker_json_failures": failures,
        "malformed_formal_markers": len(malformed),
        "detector_positive_epochs": int((epoch_table.algorithm_label == "POSITIVE").sum()),
        "detector_negative_epochs": int((epoch_table.algorithm_label == "NEGATIVE").sum()),
        "sampled_positive": int((sampled_table.algorithm_label == "POSITIVE").sum()),
        "sampled_negative": int((sampled_table.algorithm_label == "NEGATIVE").sum()),
        "detector_threshold_uv": detector["threshold_uv"],
        "filter_details": filter_details,
        "impedance_metadata_present": bool(impedance_hits),
    }
    del streams, eeg, marker, data_uv, epochs, epoch_arrays, fp
    gc.collect()
    return sampled_epochs, sampled_table, characteristic_rows, metadata_summary


def rounded_scale(values: np.ndarray):
    absolute = np.abs(values).ravel()
    q999 = float(np.percentile(absolute, 99.9))
    return max(100.0, 25.0 * math.ceil(1.08 * q999 / 25.0))


def plot_epoch(pdf, row, epoch, times, y_limit, blinded, page_number, total_pages):
    fig, axes = plt.subplots(2, 5, figsize=(11.69, 8.27), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.08, top=0.865, wspace=0.20, hspace=0.26)
    title = f"Manual ocular review - {row['epoch_id']}"
    if not blinded:
        title += f" | {row['participant']} | {row['trial_id']} | Algorithm: {row['algorithm_label']}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.955)
    fig.text(0.155, 0.902, "FRONTAL", ha="center", va="center", fontsize=10, fontweight="bold", color="#59368C")
    fig.text(0.61, 0.902, "POSTERIOR", ha="center", va="center", fontsize=10, fontweight="bold", color="#176B60")
    clip_count = 0
    for ax, channel in zip(axes.ravel(), PLOT_LAYOUT):
        values = epoch[PLOT_CHANNELS.index(channel)]
        color = "#6F42A1" if channel in FRONTAL else "#187A6D"
        ax.plot(times, values, color=color, linewidth=0.9)
        ax.axvspan(-PRE_SECONDS, 0, color="#F3F4F6", zorder=-2)
        ax.axvline(0, color="#D62828", linestyle="--", linewidth=1.0)
        ax.axhline(0, color="#A6A6A6", linewidth=0.5)
        ax.set_title(channel, fontsize=11, fontweight="bold", color=color)
        ax.set_xlim(-PRE_SECONDS, POST_SECONDS)
        ax.set_ylim(-y_limit, y_limit)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
        ax.tick_params(labelsize=8)
        clip_count += int(np.sum(np.abs(values) > y_limit))
    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude (uV)", fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time from image onset (s)", fontsize=8)
    subtitle = f"Shared participant-level y-axis: +/-{y_limit:.0f} uV | 0 s = image onset | display baseline: -0.2 to 0 s"
    fig.text(0.5, 0.885, subtitle, ha="center", fontsize=9, color="#333333")
    if clip_count:
        fig.text(0.5, 0.045, f"Display note: {clip_count} sample points exceed the shared robust scale and are clipped at the plot edge.", ha="center", fontsize=8, color="#A61B1B")
    fig.text(0.985, 0.025, f"Page {page_number}/{total_pages}", ha="right", fontsize=8, color="#666666")
    pdf.savefig(fig, dpi=160)
    plt.close(fig)


def write_markdown_outputs(output: Path, manifest: pd.DataFrame, characteristics: pd.DataFrame, metadata: list[dict], scales: dict):
    counts = manifest.groupby(["participant", "algorithm_label"]).size().unstack(fill_value=0)
    participant_summary = characteristics.groupby("participant").agg(
        std_uv_median=("std_uv", "median"),
        robust_range_uv_median=("p5_p95_range_uv", "median"),
        mad_uv_median=("mad_uv", "median"),
        slow_rms_uv_median=("slow_0p1_1_rms_uv", "median"),
        fp_corr=("fp1_fp2_correlation", "first"),
        threshold_uv=("detector_threshold_uv", "first"),
        positive_epochs=("detector_positive_image_epochs", "first"),
        impedance=("impedance_metadata_present", "first"),
    )
    readme = f"""# Ocular detector manual validation - reviewer instructions

## Purpose

This package lets a human reviewer check whether the diagnostic ocular detector's labels agree with visible EEG patterns. No epochs have been deleted or modified in the source XDF files.

## Recommended blinded workflow

1. Open `manual_review_blinded.pdf` first. It hides participant, trial, and algorithm status; it shows only the opaque review ID.
2. For each page, find the same `epoch_id` in `manual_labels.csv`.
3. Enter exactly one of these values in `manual_label`:
   - `CLEAR`: large synchronous Fp1/Fp2 deflection consistent with a blink or vertical eye movement, with plausible simultaneous spread toward posterior electrodes.
   - `AMBIGUOUS`: frontal activity is present, but ocular origin or posterior contamination is uncertain.
   - `NONE`: no convincing blink/vertical-eye-movement pattern.
4. Use `notes` for brief reasoning, uncertainty, clipping, movement, or unusual morphology.
5. Do not open `manual_review_unblinded.pdf` until the blinded labels are complete. That PDF reveals POSITIVE/NEGATIVE status for comparison.

## Plot conventions

- Epoch window: -0.2 to +0.8 s around image onset.
- Red dashed line: image onset at 0 s.
- Units: microvolts (uV).
- Display-only baseline: mean of -0.2 to 0 s removed from each channel.
- Each participant uses one shared robust y-axis across all sampled epochs and all ten displayed channels. The scale is identical in blinded and unblinded PDFs. Rare points beyond the robust scale are visibly clipped and counted on that page.
- Shared scales: {json.dumps(scales, sort_keys=True)} uV.

## Reproducibility

- Fixed master seed: `{SEED}`.
- Sampling uses independent deterministic NumPy seed streams for each participant and algorithm class.
- Up to 20 POSITIVE and 20 NEGATIVE image epochs were sampled per participant. If fewer than 20 positives existed, all positives were included.
- PDF page order was shuffled deterministically using the same master seed; `epoch_id` follows PDF page order.
- The detector and filtering match the previous cohort diagnostic: 0.1-40 Hz zero-phase FIR, then synchronized Fp1/Fp2 detector low-passed at 8 Hz with a 4 robust-sigma height, 1.5 robust-sigma prominence, and 0.40 s minimum separation.

## Limitation

`algorithm_label` is not ground truth and must not be used as the manual answer. Precision, recall, false-positive rate, and false-negative rate are not calculated while `manual_label` is blank.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    lines = [
        "# Manual validation of the cohort ocular-artifact detector",
        "",
        "## 1. What is being manually validated?",
        "",
        "The diagnostic detector marks an image epoch POSITIVE when at least one strong, same-polarity synchronized Fp1/Fp2 event falls inside -0.2 to +0.8 s around image onset. This package asks a human reviewer to decide whether the displayed pattern is clearly ocular, ambiguous, or not clearly ocular.",
        "",
        "## 2. Why POSITIVE does not automatically mean a true blink",
        "",
        "The detector recognizes a signal shape and threshold crossing, not the biological cause. Frontal slow waves, movement, electrode shifts, or other large synchronized activity can resemble blinks. Manual review is therefore required before using detector labels for final trial rejection.",
        "",
        "## 3. Reproducible sample",
        "",
        f"Fixed random seed: `{SEED}`. Total review epochs: **{len(manifest)}**.",
        "",
        "| Participant | POSITIVE sampled | NEGATIVE sampled | Total |",
        "|---|---:|---:|---:|",
    ]
    for participant in PARTICIPANTS:
        positive = int(counts.loc[participant, "POSITIVE"]) if "POSITIVE" in counts.columns else 0
        negative = int(counts.loc[participant, "NEGATIVE"]) if "NEGATIVE" in counts.columns else 0
        lines.append(f"| {participant} | {positive} | {negative} | {positive + negative} |")
    lines.extend(
        [
            "",
            "## 4. Frontal-signal scale comparison",
            "",
            "These characteristics were measured on the same in-memory 0.1-40 Hz detector input, in microvolts. Values summarize Fp1/Fp2; they are not impedance estimates.",
            "",
            "| Participant | Median SD (uV) | Median 5th-95th range (uV) | Median MAD (uV) | Median slow 0.1-1 Hz RMS (uV) | Fp1/Fp2 correlation | Detector threshold (uV) | Positive image epochs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for participant, row in participant_summary.loc[PARTICIPANTS].iterrows():
        lines.append(
            f"| {participant} | {row.std_uv_median:.1f} | {row.robust_range_uv_median:.1f} | {row.mad_uv_median:.1f} | {row.slow_rms_uv_median:.1f} | {row.fp_corr:.3f} | {row.threshold_uv:.1f} | {int(row.positive_epochs)} |"
        )
    any_impedance = any(item["impedance_metadata_present"] for item in metadata)
    lines.extend(
        [
            "",
            "The frontal signal scales differ notably. P008 has the largest absolute frontal variability/range and also the highest participant-specific detector threshold, yet it has the fewest positive image epochs. P011 does not have the largest overall frontal scale, despite having the highest positive-epoch count. Therefore, the burden difference is not a simple monotonic consequence of larger baseline amplitude alone.",
            "",
            "## 5. Is there evidence of over-detection in high-burden participants?",
            "",
            "Not enough evidence yet to decide. The scale comparison argues against a simple 'larger amplitude means more positives' bias, because P008 is the counterexample. However, a participant-specific waveform pattern, slow activity, or repeated non-ocular transient could still cross the detector criteria often in P007/P011. Only blinded human labels can establish whether those positives are true ocular artifacts or over-detections.",
            "",
            "## 6. Impedance metadata",
            "",
            ("At least one explicit impedance/resistance metadata field was found; exact entries are preserved in `frontal_signals.csv`." if any_impedance else "No explicit impedance/resistance field was found in the SAGA XDF stream metadata for these four participants. Impedance is therefore reported as unavailable and is not inferred from EEG amplitude."),
            "",
            "## 7. Human labels still required",
            "",
            "All `manual_label` and `notes` cells are blank. A reviewer must label every sampled epoch as CLEAR, AMBIGUOUS, or NONE using the blinded PDF. Until then, detector precision, sensitivity/recall, false-positive rate, and false-negative rate cannot be calculated and the detector should not be frozen as a final rejection rule.",
            "",
            "## 8. Safety and scope",
            "",
            "The XDF SHA-256 hash for every source was checked before and after processing and was unchanged. No epoch rejection, ICA, interpolation, rereferencing, final preprocessing, or condition statistics were performed.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sample_blocks = []
    characteristic_rows = []
    metadata = []
    scales = {}
    for participant in PARTICIPANTS:
        print(f"Loading and sampling {participant}...", flush=True)
        epochs, table, participant_characteristics, participant_metadata = load_participant(args.root, participant)
        scale = rounded_scale(epochs)
        scales[participant] = scale
        sample_blocks.append({"participant": participant, "epochs": epochs, "table": table, "scale": scale})
        characteristic_rows.extend(participant_characteristics)
        metadata.append(participant_metadata)
        print(
            f"  detected {participant_metadata['detector_positive_epochs']}/432; sampled "
            f"{participant_metadata['sampled_positive']} positive + {participant_metadata['sampled_negative']} negative",
            flush=True,
        )

    global_rows = []
    for block_index, block in enumerate(sample_blocks):
        for local_index, row in block["table"].iterrows():
            global_rows.append(
                {
                    "block_index": block_index,
                    "local_index": int(local_index),
                    "participant": block["participant"],
                    **row.to_dict(),
                }
            )
    order_rng = np.random.default_rng(np.random.SeedSequence([SEED, 999]))
    shuffled = [global_rows[index] for index in order_rng.permutation(len(global_rows))]
    manifest_rows = []
    review_items = []
    for page_index, item in enumerate(shuffled, start=1):
        epoch_id = f"REV-{page_index:03d}"
        trial_id = f"run-{item['run']}_trial-{item['trial_in_run']:02d}_global-{item['global_trial']:03d}"
        manifest_rows.append(
            {
                "participant": item["participant"],
                "epoch_id": epoch_id,
                "trial_id": trial_id,
                "algorithm_label": item["algorithm_label"],
                "manual_label": "",
                "notes": "",
            }
        )
        block = sample_blocks[item["block_index"]]
        review_items.append(
            {
                **manifest_rows[-1],
                "epoch": block["epochs"][item["local_index"]],
                "scale": block["scale"],
                "sfreq": metadata[item["block_index"]]["sfreq_hz"],
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    characteristics = pd.DataFrame(characteristic_rows)
    manifest.to_csv(args.output / "manual_labels.csv", index=False, encoding="utf-8")
    characteristics.to_csv(args.output / "frontal_signals.csv", index=False, encoding="utf-8")
    (args.output / "sampling_metadata.json").write_text(
        json.dumps({"seed": SEED, "participants": metadata, "shared_plot_scales_uv": scales}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total_pages = len(review_items)
    blinded_path = args.output / "manual_review_blinded.pdf"
    unblinded_path = args.output / "manual_review_unblinded.pdf"
    print(f"Writing {total_pages}-page blinded and unblinded PDFs...", flush=True)
    with PdfPages(blinded_path, metadata={"Title": "Blinded ocular detector manual review"}) as blinded_pdf, PdfPages(
        unblinded_path, metadata={"Title": "Unblinded ocular detector manual review"}
    ) as unblinded_pdf:
        for page_number, item in enumerate(review_items, start=1):
            times = np.arange(item["epoch"].shape[1]) / item["sfreq"] - PRE_SECONDS
            plot_epoch(blinded_pdf, item, item["epoch"], times, item["scale"], True, page_number, total_pages)
            plot_epoch(unblinded_pdf, item, item["epoch"], times, item["scale"], False, page_number, total_pages)
            if page_number % 20 == 0 or page_number == total_pages:
                print(f"  wrote page {page_number}/{total_pages}", flush=True)

    write_markdown_outputs(args.output, manifest, characteristics, metadata, scales)
    print(json.dumps({"output": str(args.output), "pages_each_pdf": total_pages, "rows": len(manifest), "seed": SEED}, indent=2))


if __name__ == "__main__":
    main()
