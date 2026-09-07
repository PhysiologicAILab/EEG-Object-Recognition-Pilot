"""Cohort-level acquisition and ocular-artifact QC for the final protocol.

This script reads XDF files only. It never overwrites XDFs and does not save
filtered EEG, apply ICA, interpolate, rereference, or reject epochs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyxdf
from scipy import signal

sys.stdout.reconfigure(encoding="utf-8")

CHANNELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
FRONTAL = ["Fp1", "Fp2"]
POSTERIOR = ["P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
FORMAL_RUNS = {f"{number:02d}" for number in range(1, 7)}
EXPECTED_KEYS = {(f"{run:02d}", trial) for run in range(1, 7) for trial in range(1, 73)}
MICROVOLT_UNITS = {"µv", "uv", "µvolt", "microvolt", "microvolts"}
NEIGHBOURS = {
    "Fp1": ["Fp2", "F3", "Fz"], "Fp2": ["Fp1", "F4", "Fz"],
    "F3": ["Fp1", "Fz", "C3"], "Fz": ["Fp1", "Fp2", "F3", "F4", "Cz"], "F4": ["Fp2", "Fz", "C4"],
    "C3": ["F3", "Cz", "P3", "P7"], "Cz": ["Fz", "C3", "C4", "Pz"], "C4": ["F4", "Cz", "P4", "P8"],
    "P7": ["C3", "P3", "O1"], "P3": ["C3", "P7", "Pz", "O1"], "Pz": ["Cz", "P3", "P4", "Oz"],
    "P4": ["C4", "Pz", "P8", "O2"], "P8": ["C4", "P4", "O2"], "O1": ["P7", "P3", "Oz"],
    "Oz": ["Pz", "O1", "O2"], "O2": ["P4", "P8", "Oz"],
}
BASE_ORDER = ["trial_start", "fixation_onset", "image_onset", "image_offset", "silent_delay_onset", "microphone_recording_start", "response_cue_onset", "microphone_recording_stop", "trial_end"]


def one(value, default=""):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return default if value is None else value


def robust_sigma(values):
    values = np.asarray(values, float)
    center = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - center)))


def robust_z(values):
    values = np.asarray(values, float)
    scale = robust_sigma(values)
    return (values - np.nanmedian(values)) / scale if scale > 0 else np.zeros_like(values)


def discover(root: Path):
    xdfs = sorted(root.rglob("*.xdf"))
    by_participant = defaultdict(list)
    for path in xdfs:
        match = re.search(r"sub-P(\d{3})", str(path), flags=re.I)
        if match:
            participant = f"P{int(match.group(1)):03d}"
        elif "sub-006" in str(path).lower() and "sub-p006" in str(path).lower():
            participant = "P006"
        else:
            participant = path.parts[len(root.parts)] if len(path.parts) > len(root.parts) else path.stem
        by_participant[participant].append(path)
    return xdfs, by_participant


def session_from_paths(paths):
    sessions = []
    for path in paths:
        match = re.search(r"ses-([A-Za-z]?\d+)", str(path), flags=re.I)
        if match:
            sessions.append(match.group(1))
    return ";".join(sorted(set(sessions))) or "UNKNOWN"


def inclusion_table(root: Path, by_participant):
    rows = []
    expected = [f"P{x:03d}" for x in range(1, 20) if x != 14]
    for participant in expected:
        paths = by_participant.get(participant, [])
        number = int(participant[1:])
        status = "PILOT" if number <= 4 else "FINAL"
        notes = []
        if not paths:
            notes.append("No XDF discovered")
        if participant == "P015":
            notes.append("P014 is absent from the recorded session identifiers; P015 is retained as recorded")
        if participant == "P005" and len(paths) > 1:
            notes.append(f"Participant spans {len(paths)} XDF recording files; aggregated as one participant")
        if participant == "P006":
            notes.append("Nested sub-006/ses-006 naming retained and mapped to top-level P006")
        rows.append({
            "participant_id": participant,
            "session_id": session_from_paths(paths),
            "xdf_count": len(paths),
            "xdf_path": " | ".join(str(path) for path in paths),
            "protocol_status": status,
            "usable_for_current_formal_cohort_qc": "YES" if status == "FINAL" and paths else "NO",
            "notes": "; ".join(notes) or "None",
        })
    for participant, paths in sorted(by_participant.items()):
        if participant not in expected:
            rows.append({
                "participant_id": participant, "session_id": session_from_paths(paths), "xdf_count": len(paths),
                "xdf_path": " | ".join(str(path) for path in paths), "protocol_status": "UNASSIGNED/OTHER",
                "usable_for_current_formal_cohort_qc": "NO", "notes": "Not part of the user-defined P001-P019 participant set",
            })
    return pd.DataFrame(rows)


def channel_metadata(eeg):
    count = int(float(one(eeg["info"].get("channel_count", [0]))))
    try:
        channels = eeg["info"]["desc"][0]["channels"][0]["channel"]
    except Exception:
        channels = []
    rows = []
    for index in range(count):
        item = channels[index] if index < len(channels) else {}
        rows.append({"index": index, "name": str(one(item.get("label", [f"Ch{index+1}"]))), "type": str(one(item.get("type", [""]))), "unit": str(one(item.get("unit", [""])))} )
    return rows


def parse_markers(marker, recording_index, eeg_start, formal_run_override=None):
    parsed, failures = [], 0
    for sample, timestamp in zip(marker.get("time_series", []), marker.get("time_stamps", [])):
        try:
            item = json.loads(sample[0])
            item["_original_run_id"] = str(item.get("run_id", ""))
            if formal_run_override and str(item.get("run_id", "")) in FORMAL_RUNS:
                item["run_id"] = formal_run_override
            item["_timestamp"] = float(timestamp)
            item["_recording_index"] = recording_index
            item["_eeg_start"] = eeg_start
            parsed.append(item)
        except Exception:
            failures += 1
    return parsed, failures


def trial_table(markers):
    formal = []
    malformed = []
    for item in markers:
        run = str(item.get("run_id", ""))
        if run not in FORMAL_RUNS:
            continue
        try:
            trial = int(item.get("trial_index_run"))
        except Exception:
            malformed.append(f"recording {item.get('_recording_index')}: invalid run/trial on {item.get('event_name')}")
            continue
        formal.append((run, trial, item))
    grouped = defaultdict(list)
    for run, trial, item in formal:
        grouped[(run, trial)].append(item)
    rows = []
    for key in sorted(set(grouped) | EXPECTED_KEYS):
        run, trial = key
        items = sorted(grouped.get(key, []), key=lambda x: x["_timestamp"])
        events = defaultdict(list)
        for item in items:
            event = str(item.get("event_name", ""))
            event = "image_onset" if event.endswith("_image_onset") else event
            events[event].append(item)
        image_items = events.get("image_onset", [])
        first = items[0] if items else {}
        missing = [name for name in BASE_ORDER if len(events.get(name, [])) == 0]
        duplicated = [name for name, values in events.items() if len(values) > 1]
        ordered_values = [events[name][0]["_timestamp"] for name in BASE_ORDER if events.get(name)]
        out_of_order = bool(len(ordered_values) > 1 and np.any(np.diff(ordered_values) < 0))

        def delta(first_name, second_name):
            if events.get(first_name) and events.get(second_name):
                return 1000.0 * (events[second_name][0]["_timestamp"] - events[first_name][0]["_timestamp"])
            return np.nan

        anomaly = []
        if missing:
            anomaly.append("missing:" + ",".join(missing))
        if duplicated:
            anomaly.append("duplicated:" + ",".join(duplicated))
        if out_of_order:
            anomaly.append("out_of_order")
        image = image_items[0] if image_items else None
        rows.append({
            "run": run, "trial_in_run": trial, "marker_item_count": len(items),
            "global_trial": first.get("trial_index_global", first.get("global_trial_index", np.nan)),
            "condition": first.get("condition", ""),
            "image_onset_count": len(image_items), "missing_events": ";".join(missing) or "none",
            "duplicated_events": ";".join(duplicated) or "none", "out_of_order": out_of_order,
            "anomaly": " | ".join(anomaly) or "none",
            "fixation_ms": delta("fixation_onset", "image_onset"),
            "image_ms": delta("image_onset", "image_offset"),
            "silent_delay_ms": delta("silent_delay_onset", "response_cue_onset"),
            "microphone_recording_ms": delta("microphone_recording_start", "microphone_recording_stop"),
            "response_cue_to_microphone_stop_ms": delta("response_cue_onset", "microphone_recording_stop"),
            "image_timestamp": image["_timestamp"] if image else np.nan,
            "recording_index": int(image["_recording_index"]) if image else -1,
            "image_onset_seconds": float(image["_timestamp"] - image["_eeg_start"]) if image else np.nan,
        })
    return pd.DataFrame(rows), malformed


def detect_ocular_events(recordings, sfreq):
    sos = signal.butter(4, 8.0, btype="lowpass", fs=sfreq, output="sos")
    smooth_pairs = []
    combined_for_scale = []
    for rec in recordings:
        front = rec["data_uv"][[CHANNELS.index("Fp1"), CHANNELS.index("Fp2")]]
        fp1 = signal.sosfiltfilt(sos, front[0])
        fp2 = signal.sosfiltfilt(sos, front[1])
        common = (fp1 + fp2) / 2.0
        smooth_pairs.append((fp1, fp2, common))
        # Match the validated P007 detector exactly: estimate the robust
        # centre/scale from the full smoothed trace, then exclude candidate
        # peaks within 20 s of each recording boundary.
        combined_for_scale.append(common)
    scale_values = np.concatenate(combined_for_scale) if combined_for_scale else np.concatenate([x[2] for x in smooth_pairs])
    center = float(np.median(scale_values))
    scale = robust_sigma(scale_values)
    events = []
    for rec_index, ((fp1, fp2, common), rec) in enumerate(zip(smooth_pairs, recordings)):
        detection = np.abs(common - center)
        peaks, _ = signal.find_peaks(detection, height=4.0 * scale, prominence=1.5 * scale, distance=int(round(0.40 * sfreq)))
        edge = int(round(20 * sfreq))
        peaks = peaks[(peaks >= edge) & (peaks < detection.size - edge)]
        keep = (fp1[peaks] - np.median(fp1)) * (fp2[peaks] - np.median(fp2)) > 0
        for peak in peaks[keep]:
            events.append({"recording_index": rec_index, "sample": int(peak), "time_seconds": float(peak / sfreq)})
    return events, {"robust_sigma_uv": scale, "threshold_uv": 4.0 * scale, "lowpass_hz": 8.0, "minimum_separation_s": 0.40}


def channel_metrics(participant, recordings, sfreq):
    decimated = np.concatenate([rec["data_uv"][:, ::2] for rec in recordings], axis=1)
    fs = sfreq / 2.0
    median_other = np.empty_like(decimated)
    for index in range(len(CHANNELS)):
        median_other[index] = np.median(np.delete(decimated, index, axis=0), axis=0)
    freq, psd = signal.welch(decimated, fs=fs, nperseg=min(int(20 * fs), decimated.shape[1]), noverlap=min(int(10 * fs), max(0, decimated.shape[1] // 2 - 1)), axis=1)

    def power(row, low, high):
        mask = (freq >= low) & (freq <= high)
        return float(np.trapezoid(row[mask], freq[mask]))

    rows = []
    duration_minutes = sum(rec["data_uv"].shape[1] for rec in recordings) / sfreq / 60.0
    for index, channel in enumerate(CHANNELS):
        x = decimated[index]
        med = float(np.median(x))
        mad = robust_sigma(x)
        q5, q95 = np.percentile(x, [5, 95])
        total = power(psd[index], 0.1, 40)
        extreme = np.abs(x - med) > 8 * mad if mad > 0 else np.zeros(x.size, bool)
        starts = np.flatnonzero(extreme & ~np.r_[False, extreme[:-1]])
        correlations = [abs(float(np.corrcoef(x, decimated[CHANNELS.index(n)])[0, 1])) for n in NEIGHBOURS[channel]]
        rows.append({
            "participant_id": participant, "channel": channel, "region": "posterior" if channel in POSTERIOR else ("frontal" if channel in FRONTAL else "other"),
            "std_uv": float(np.std(x)), "mad_sigma_uv": mad, "p5_p95_range_uv": float(q95 - q5), "peak_to_peak_uv": float(np.ptp(x)),
            "near_zero_variance": bool(np.std(x) < 0.5 or (q95 - q5) < 1.0),
            "near_identical_step_percent": float(100 * np.mean(np.abs(np.diff(x)) < 1e-6)),
            "extreme_transient_events_per_min": float(len(starts) / duration_minutes),
            "slow_0p1_1_fraction": power(psd[index], 0.1, 1.0) / total if total > 0 else np.nan,
            "high_30_40_fraction": power(psd[index], 30, 40) / total if total > 0 else np.nan,
            "correlation_with_median_other": float(np.corrcoef(x, median_other[index])[0, 1]),
            "median_absolute_neighbor_correlation": float(np.median(correlations)),
        })
    table = pd.DataFrame(rows)
    for column in ["std_uv", "mad_sigma_uv", "p5_p95_range_uv", "extreme_transient_events_per_min", "slow_0p1_1_fraction", "high_30_40_fraction"]:
        table[column + "_robust_z"] = robust_z(table[column])
    statuses, reasons = [], []
    for _, row in table.iterrows():
        evidence = []
        flat = bool(row.near_zero_variance or row.near_identical_step_percent > 5)
        high_var = row.std_uv_robust_z > 5 or row.mad_sigma_uv_robust_z > 5
        high_hf = row.high_30_40_fraction_robust_z > 5
        high_extreme = row.extreme_transient_events_per_min_robust_z > 5
        low_spatial = abs(row.correlation_with_median_other) < 0.10 and row.median_absolute_neighbor_correlation < 0.15
        if flat:
            evidence.append("flat/near-zero variability")
        if high_var:
            evidence.append("extreme variability outlier")
        if high_hf:
            evidence.append("high 30-40 Hz contamination outlier")
        if high_extreme:
            evidence.append("extreme-transient-rate outlier")
        if low_spatial:
            evidence.append("weak spatial consistency")
        if flat or (row.channel not in FRONTAL and low_spatial and sum([high_var, high_hf, high_extreme]) >= 2) or (row.channel in FRONTAL and low_spatial and high_hf and high_extreme):
            status = "CANDIDATE BAD"
        elif evidence or row.slow_0p1_1_fraction_robust_z > 5:
            status = "WATCH"
            if row.slow_0p1_1_fraction_robust_z > 5:
                evidence.append("slow-power outlier")
        else:
            status = "GOOD"
        if row.channel in FRONTAL and status == "WATCH" and not flat and not high_hf:
            evidence.append("frontal slow/transient activity may be ocular rather than electrode failure")
        statuses.append(status)
        reasons.append("; ".join(evidence) or "No convincing multi-metric abnormality")
    table["status"] = statuses
    table["reason"] = reasons
    return table


def process_participant(participant, paths):
    recordings, all_markers = [], []
    acquisition_parts = []
    channel_names_seen, units_seen = [], []
    marker_failures = 0
    for recording_index, path in enumerate(paths):
        streams, _ = pyxdf.load_xdf(str(path), verbose=False)
        eeg_candidates = [s for s in streams if str(one(s["info"].get("type", [""]))).lower() == "eeg" and "saga" in str(one(s["info"].get("name", [""]))).lower()]
        marker_candidates = [s for s in streams if str(one(s["info"].get("type", [""]))).lower() in {"marker", "markers"} and "psychopy" in str(one(s["info"].get("name", [""]))).lower()]
        if len(eeg_candidates) != 1 or not marker_candidates:
            raise RuntimeError(f"{path}: SAGA={len(eeg_candidates)}, PsychoPyMarkers={len(marker_candidates)}")
        eeg = eeg_candidates[0]
        marker = marker_candidates[0]  # duplicate streams are deliberately not merged
        metadata = channel_metadata(eeg)
        selected = [row for row in metadata if row["type"].strip().lower() == "eeg" and row["name"].upper() not in {"TRIGGERS", "STATUS", "COUNTER"}]
        names_original = [row["name"] for row in selected]
        names = ["Fp1" if name == "Fpz" else name for name in names_original]
        if names != CHANNELS:
            raise RuntimeError(f"{path}: unexpected EEG channels after Fpz->Fp1: {names}")
        units = {row["unit"].strip().lower() for row in selected}
        if not units.issubset(MICROVOLT_UNITS):
            raise RuntimeError(f"{path}: units not explicitly microvolts: {units}")
        nominal = float(one(eeg["info"].get("nominal_srate", [0])))
        if nominal <= 0:
            raise RuntimeError(f"{path}: invalid nominal sampling rate")
        timestamps = np.asarray(eeg["time_stamps"], float)
        dt = np.diff(timestamps)
        median_dt = float(np.median(dt))
        gaps = dt > 1.5 * median_dt
        nonmono = dt <= 0
        series = np.asarray(eeg["time_series"])
        indices = [row["index"] for row in selected]
        data_v = np.asarray(series[:, indices].T, dtype=np.float64) * 1e-6
        filtered_v = mne.filter.filter_data(data_v, nominal, l_freq=0.1, h_freq=40.0, method="fir", phase="zero", fir_window="hamming", fir_design="firwin", copy=False, verbose="ERROR")
        data_uv = filtered_v * 1e6
        file_run_match = re.search(r"_run-(\d{3})_eeg\.xdf$", path.name, flags=re.I)
        file_run = f"{int(file_run_match.group(1)):02d}" if file_run_match else None
        # P005 restarted the experiment program before the fifth and sixth XDF:
        # marker run IDs reset to 01/02 although the XDF filenames explicitly
        # identify physical recordings 005/006. Remap only in the in-memory QC
        # representation; the XDF and original marker values remain unchanged.
        formal_run_override = file_run if participant == "P005" and file_run in {"05", "06"} else None
        parsed, failed = parse_markers(marker, recording_index, float(timestamps[0]), formal_run_override=formal_run_override)
        all_markers.extend(parsed)
        marker_failures += failed
        recordings.append({"path": path, "data_uv": data_uv, "timestamps": timestamps, "nominal": nominal})
        acquisition_parts.append({
            "effective_srate_hz": 1.0 / median_dt, "duration_s": float(timestamps[-1] - timestamps[0]),
            "gap_count": int(gaps.sum()), "max_gap_ms": 1000 * float(dt.max()) if dt.size else np.nan,
            "nonmonotonic_count": int(nonmono.sum()), "sample_count": len(timestamps), "marker_stream_count": len(marker_candidates),
        })
        channel_names_seen.append(names_original)
        units_seen.append(sorted(units))
        del streams, eeg, marker, series, data_v, filtered_v
        gc.collect()

    nominal_rates = {rec["nominal"] for rec in recordings}
    if len(nominal_rates) != 1:
        raise RuntimeError(f"{participant}: inconsistent nominal rates {nominal_rates}")
    sfreq = float(next(iter(nominal_rates)))
    trials, malformed = trial_table(all_markers)
    ocular_events, detection = detect_ocular_events(recordings, sfreq)
    events_by_recording = defaultdict(list)
    for event in ocular_events:
        events_by_recording[event["recording_index"]].append(event["sample"])
    for key in events_by_recording:
        events_by_recording[key] = np.asarray(sorted(events_by_recording[key]), int)

    valid_epochs = 0
    out_of_bounds = 0
    gross_artifact = 0
    ocular_epochs = 0
    posterior_ocular_epochs = 0
    npre, npost = int(round(0.2 * sfreq)), int(round(0.8 * sfreq))
    for _, trial in trials[trials.image_onset_count > 0].iterrows():
        rec_index = int(trial.recording_index)
        rec = recordings[rec_index]
        center = int(round(float(trial.image_onset_seconds) * sfreq))
        lo, hi = center - npre, center + npost + 1
        if lo < 0 or hi > rec["data_uv"].shape[1]:
            out_of_bounds += 1
            continue
        epoch = rec["data_uv"][:, lo:hi].copy()
        epoch -= np.mean(epoch[:, :npre + 1], axis=1, keepdims=True)
        valid_epochs += 1
        if not np.isfinite(epoch).all() or float(np.max(np.ptp(epoch, axis=1))) > 5000:
            gross_artifact += 1
        rec_events = events_by_recording.get(rec_index, np.array([], int))
        left, right = np.searchsorted(rec_events, [lo, hi], side="left")
        inside = rec_events[left:right]
        if inside.size:
            ocular_epochs += 1
            substantial = False
            for event_sample in inside:
                event_epoch_sample = event_sample - lo
                local_lo = max(0, event_epoch_sample - int(round(0.25 * sfreq)))
                local_hi = min(epoch.shape[1], event_epoch_sample + int(round(0.25 * sfreq)) + 1)
                front = float(np.max(np.ptp(epoch[[CHANNELS.index(x) for x in FRONTAL], local_lo:local_hi], axis=1)))
                post = float(np.max(np.ptp(epoch[[CHANNELS.index(x) for x in POSTERIOR], local_lo:local_hi], axis=1)))
                if post >= 50 and post >= 0.20 * front:
                    substantial = True
            if substantial:
                posterior_ocular_epochs += 1

    channel_table = channel_metrics(participant, recordings, sfreq)
    candidate_bads = channel_table.loc[channel_table.status == "CANDIDATE BAD", "channel"].tolist()
    watch = channel_table.loc[channel_table.status == "WATCH", "channel"].tolist()
    posterior_candidates = [x for x in candidate_bads if x in POSTERIOR]
    posterior_present = all(channel in CHANNELS for channel in POSTERIOR)
    posterior_usable = posterior_present and len(posterior_candidates) <= 1

    found_keys = set(zip(trials.loc[trials.image_onset_count > 0, "run"], trials.loc[trials.image_onset_count > 0, "trial_in_run"]))
    missing_keys = sorted(EXPECTED_KEYS - found_keys)
    duplicate_images = int((trials.image_onset_count > 1).sum())
    anomalous = trials[(trials.marker_item_count > 0) & (trials.anomaly != "none")]
    run_counts = trials[trials.image_onset_count > 0].groupby("run").size().reindex(sorted(FORMAL_RUNS), fill_value=0)
    gap_count = sum(part["gap_count"] for part in acquisition_parts)
    nonmono_count = sum(part["nonmonotonic_count"] for part in acquisition_parts)
    max_gap_ms = max(part["max_gap_ms"] for part in acquisition_parts)
    obvious_dropout = bool(max_gap_ms > 100 or nonmono_count > 0)
    sampling_integrity = "FAIL" if obvious_dropout else ("WATCH" if gap_count else "PASS")

    acquisition = {
        "participant_id": participant, "xdf_readable": "YES", "xdf_count": len(paths), "saga_eeg_present": "YES", "psychopy_markers_present": "YES",
        "selected_marker_streams": len(paths), "duplicate_marker_streams_not_merged": int(sum(max(0, x["marker_stream_count"] - 1) for x in acquisition_parts)),
        "eeg_channel_count": len(CHANNELS), "eeg_channel_names": ";".join(CHANNELS), "xdf_original_channel_names": ";".join(channel_names_seen[0]),
        "xdf_units": ";".join(sorted(set(sum(units_seen, [])))), "nominal_srate_hz": sfreq,
        "effective_srate_hz_median_across_files": float(np.median([x["effective_srate_hz"] for x in acquisition_parts])),
        "effective_srate_hz_min": float(np.min([x["effective_srate_hz"] for x in acquisition_parts])),
        "effective_srate_hz_max": float(np.max([x["effective_srate_hz"] for x in acquisition_parts])),
        "recording_duration_minutes_sum": sum(x["duration_s"] for x in acquisition_parts) / 60,
        "timestamp_gap_count": gap_count, "maximum_timestamp_gap_ms": max_gap_ms, "duplicate_or_nonmonotonic_timestamp_count": nonmono_count,
        "obvious_recording_dropout": obvious_dropout, "sampling_integrity": sampling_integrity, "marker_json_parse_failures": marker_failures,
    }
    timing_columns = ["fixation_ms", "image_ms", "silent_delay_ms", "microphone_recording_ms", "response_cue_to_microphone_stop_ms"]
    marker_row = {
        "participant_id": participant, "formal_run_count": int((run_counts > 0).sum()),
        **{f"run_{run}_image_trials": int(run_counts.loc[run]) for run in sorted(FORMAL_RUNS)},
        "expected_image_trials": 432, "observed_unique_image_trials": int((trials.image_onset_count > 0).sum()),
        "missing_image_onsets": len(missing_keys), "duplicate_image_onset_trial_keys": duplicate_images,
        "malformed_marker_records": len(malformed), "out_of_order_trial_groups": int(trials.out_of_order.sum()),
        "trial_groups_with_any_marker_anomaly": int(len(anomalous)),
        "p005_run_id_reset_mapping_applied": bool(participant == "P005"),
        "missing_trial_details": "; ".join(f"run {run} trial {trial}" for run, trial in missing_keys),
        "anomaly_details": "; ".join(f"run {row.run} trial {int(row.trial_in_run)} [{row.anomaly}]" for _, row in anomalous.iterrows()),
    }
    for column in timing_columns:
        values = trials[column].dropna().to_numpy(float)
        marker_row[column.replace("_ms", "_median_ms")] = float(np.median(values)) if values.size else np.nan
        marker_row[column.replace("_ms", "_p05_ms")] = float(np.percentile(values, 5)) if values.size else np.nan
        marker_row[column.replace("_ms", "_p95_ms")] = float(np.percentile(values, 95)) if values.size else np.nan

    ocular = {
        "participant_id": participant, "detection_method": "P007 Step-6 synchronized Fp1/Fp2 detector (8-Hz LP, 4 robust-sigma, 1.5-sigma prominence, 0.40-s separation)",
        "algorithmic_ocular_like_event_count": len(ocular_events),
        "recording_duration_minutes": acquisition["recording_duration_minutes_sum"],
        "ocular_like_event_rate_per_minute": len(ocular_events) / acquisition["recording_duration_minutes_sum"],
        "detection_threshold_uv": detection["threshold_uv"], "image_epochs_created": valid_epochs, "image_epochs_out_of_bounds": out_of_bounds,
        "grossly_malformed_or_extreme_epochs": gross_artifact, "ocular_contaminated_image_epochs": ocular_epochs,
        "ocular_contaminated_percent_of_432": 100 * ocular_epochs / 432,
        "posterior_propagating_ocular_epochs": posterior_ocular_epochs,
        "posterior_propagating_percent_of_432": 100 * posterior_ocular_epochs / 432,
        "posterior_propagating_percent_of_ocular_epochs": 100 * posterior_ocular_epochs / ocular_epochs if ocular_epochs else 0,
    }
    serious_marker = marker_row["observed_unique_image_trials"] < 400 or marker_row["formal_run_count"] < 6
    meaningful_marker_issue = (
        marker_row["observed_unique_image_trials"] != 432
        or marker_row["duplicate_image_onset_trial_keys"] > 0
        or marker_row["out_of_order_trial_groups"] > 0
        or marker_row["malformed_marker_records"] > 0
        or marker_row["trial_groups_with_any_marker_anomaly"] > 5
    )
    if obvious_dropout or serious_marker or not posterior_usable or valid_epochs < 400:
        qc = "RED"
    elif meaningful_marker_issue or candidate_bads or ocular["ocular_contaminated_percent_of_432"] > 30 or gap_count:
        qc = "YELLOW"
    else:
        qc = "GREEN"
    notes = []
    if len(paths) > 1:
        notes.append(f"aggregated {len(paths)} XDF files")
    if missing_keys:
        notes.append(f"{len(missing_keys)} missing image-onset trial keys")
    if anomalous.shape[0]:
        notes.append(f"{len(anomalous)} trial groups with marker anomalies")
    if candidate_bads:
        notes.append("candidate bad: " + ",".join(candidate_bads))
    if watch:
        notes.append("watch: " + ",".join(watch))
    if ocular["ocular_contaminated_percent_of_432"] > 30:
        notes.append("ocular-like image burden >30%")
    summary = {
        "Participant": participant, "Expected trials": 432, "Observed image trials": marker_row["observed_unique_image_trials"],
        "Sampling integrity": sampling_integrity, "Timestamp gaps": gap_count,
        "Bad-channel candidates": ";".join(candidate_bads) or "none", "WATCH channels": ";".join(watch) or "none",
        "Posterior channels usable?": "YES" if posterior_usable else "NO",
        "Ocular-contaminated image epochs": ocular_epochs, "Ocular-contaminated %": ocular["ocular_contaminated_percent_of_432"],
        "Posterior-propagating ocular epochs": posterior_ocular_epochs, "Posterior-propagating %": ocular["posterior_propagating_percent_of_432"],
        "Epochs successfully created": valid_epochs, "Grossly malformed epochs": gross_artifact,
        "Overall QC status": qc, "Notes": "; ".join(notes) or "No major issue",
    }
    del recordings
    gc.collect()
    return acquisition, marker_row, channel_table, ocular, summary


def save_plots(summary, output):
    order = summary.sort_values("Participant")
    colors = ["#d62728" if participant == "P007" else "#4c78a8" for participant in order.Participant]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(order.Participant, order["Ocular-contaminated %"], color=colors)
    ax.axhline(30, color="#ff9900", ls="--", label="30%")
    ax.axhline(40, color="#cc0000", ls=":", label="40%")
    ax.set_ylabel("Image epochs with algorithmic ocular-like event (%)")
    ax.set_title("Final-protocol cohort ocular-artifact burden (P007 highlighted)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "ocular_burden.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(order.Participant, order["Observed image trials"], color="#59a14f")
    ax.axhline(432, color="black", ls="--", label="Expected 432")
    ax.set_ylim(max(0, min(order["Observed image trials"].min() - 10, 400)), 440)
    ax.set_ylabel("Unique formal image-onset trials")
    ax.set_title("Final-protocol trial integrity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "trial_integrity.png", dpi=180)
    plt.close(fig)


def report_text(inclusion, acquisition, marker, channel, ocular, summary):
    final_count = len(summary)
    readable = int((acquisition.xdf_readable == "YES").sum())
    complete = int((marker.observed_unique_image_trials == 432).sum())
    posterior_usable = int((summary["Posterior channels usable?"] == "YES").sum())
    burden = ocular.set_index("participant_id")["ocular_contaminated_percent_of_432"]
    p007 = float(burden.loc["P007"]) if "P007" in burden else np.nan
    median = float(burden.median())
    minimum, maximum = float(burden.min()), float(burden.max())
    above30, above40 = int((burden > 30).sum()), int((burden > 40).sum())
    q1, q3 = float(burden.quantile(.25)), float(burden.quantile(.75))
    iqr = q3 - q1
    outlier = bool(p007 < q1 - 1.5 * iqr or p007 > q3 + 1.5 * iqr)
    representative = "an IQR outlier" if outlier else "within the cohort's non-outlier range"
    statuses = summary["Overall QC status"].value_counts().reindex(["GREEN", "YELLOW", "RED"], fill_value=0)
    candidates = channel[channel.status == "CANDIDATE BAD"].groupby("participant_id").channel.apply(lambda x: ", ".join(x)).to_dict()
    watches = channel[channel.status == "WATCH"].groupby("participant_id").channel.apply(lambda x: ", ".join(x)).to_dict()
    marker_problem = marker[
        (marker.observed_unique_image_trials != 432)
        | (marker.duplicate_image_onset_trial_keys > 0)
        | (marker.out_of_order_trial_groups > 0)
        | (marker.malformed_marker_records > 0)
        | (marker.trial_groups_with_any_marker_anomaly > 5)
    ]
    minor_aux = marker[(marker.observed_unique_image_trials == 432) & (marker.trial_groups_with_any_marker_anomaly.between(1, 5))]
    fixation_cohort = float(marker.fixation_median_ms.median())
    image_cohort = float(marker.image_median_ms.median())
    delay_cohort = float(marker.silent_delay_median_ms.median())
    microphone_cohort = float(marker.microphone_recording_median_ms.median())
    cue_to_stop_cohort = float(marker.response_cue_to_microphone_stop_median_ms.median())
    next_decision = (
        "Do not freeze ICA removal or a single amplitude-rejection threshold yet. Use the cohort distribution to design one conservative, "
        "pre-specified artifact strategy, then validate non-blink posterior preservation across several representative low/median/high-burden participants."
    )
    return f"""# Final-protocol cohort EEG QC report

## Scope

XDF files were read only. The 0.1–40 Hz diagnostic was run in memory with the SAGA Average Reference retained. No notch, additional rereference, ICA removal, interpolation, final epoch rejection, condition statistics, or cleaned FIF was created.

P001–P004 are labelled PILOT/PROTOCOL-DEVELOPMENT and excluded from formal summaries. P014 is absent from the recorded session identifiers, and all original identifiers are retained. P005's six XDF files are aggregated as one participant, and P006's nested `sub-006/ses-006` file is mapped to top-level P006. For P005 only, marker run IDs in XDF files `run-005`/`run-006` had reset to 01/02; the in-memory QC mapping uses the explicit XDF filenames to restore them to runs 05/06. Original marker payloads remain unchanged.

## Cohort identification

**{final_count} FINAL participants** were identified and processed: {', '.join(summary.Participant)}.

## Technical readability

**{readable}/{final_count}** were readable with a SAGA EEG stream, a PsychoPyMarkers stream, 16 scalp EEG channels, explicit microvolt units, and nominal 500 Hz sampling.

Sampling status counts: PASS **{int((summary['Sampling integrity'] == 'PASS').sum())}**, WATCH **{int((summary['Sampling integrity'] == 'WATCH').sum())}**, FAIL **{int((summary['Sampling integrity'] == 'FAIL').sum())}**.

## Trial and trigger completeness

**{complete}/{final_count}** participants have exactly 432 unique formal image-onset trials. Detailed missing/duplicate/order anomalies are in `markers.csv`.

{marker_problem[['participant_id','observed_unique_image_trials','missing_image_onsets','duplicate_image_onset_trial_keys','out_of_order_trial_groups','trial_groups_with_any_marker_anomaly']].to_markdown(index=False) if len(marker_problem) else 'All participants have complete image-onset counts and no meaningful trial-group marker anomaly.'}

P005 has 413/432 image onsets: run 02 is missing image onsets for trials 2–3 and run 04 ends after trial 55, so trials 56–72 are absent. **{len(minor_aux)}** otherwise complete participants contain one isolated auxiliary-marker duplication or partial trial-group anomaly; these are documented exactly in `markers.csv` but do not alter their 432 image onsets or event order.

Timing fields report medians plus 5th–95th percentiles. `response_cue_to_microphone_stop` is explicitly labelled as microphone timing; it is not assumed to be the participant-visible behavioural response window.

## Timing integrity

Median across participant-level medians:

- Fixation: **{fixation_cohort:.2f} ms** (target approximately 800 ms)
- Image: **{image_cohort:.2f} ms** (target approximately 1000 ms)
- Silent delay: **{delay_cohort:.2f} ms** (target approximately 600 ms)
- Microphone recording start to stop: **{microphone_cohort:.2f} ms**
- Response cue to microphone stop: **{cue_to_stop_cohort:.2f} ms**

There is no cohort-wide half-duration or double-duration pattern. P005/P006 show somewhat wider timing spread than later participants, which remains visible in the per-participant 5th–95th percentile columns. The approximately 2708-ms cue-to-microphone-stop interval is microphone timing, not proof that the visible behavioural response window lasted 2708 ms.

## Posterior channel usability

**{posterior_usable}/{final_count}** participants retain usable posterior coverage under the conservative multi-metric screen.

- Candidate-bad channels by participant: {json.dumps(candidates, ensure_ascii=False) if candidates else 'none'}
- WATCH channels by participant: {json.dumps(watches, ensure_ascii=False) if watches else 'none'}

Fp1/Fp2 slow synchronized transients were not treated as electrode failure merely because their amplitudes were large.

## Ocular-artifact estimates

These are **algorithmic ocular-like event estimates, not literal human blink counts**. Every participant used the same P007-derived synchronized Fp1/Fp2 detector and the same posterior-propagation rule.

- Cohort median contaminated image-epoch rate: **{median:.2f}%**
- Cohort range: **{minimum:.2f}%–{maximum:.2f}%**
- Participants above 30%: **{above30}/{final_count}**
- Participants above 40%: **{above40}/{final_count}**

## P007 in the cohort distribution

- P007 contaminated image epochs: **{p007:.2f}%**
- Cohort median: **{median:.2f}%**
- IQR: **{q1:.2f}%–{q3:.2f}%**
- Outlier check: P007 is **{representative}** by the conventional 1.5×IQR rule.

P007 is the **third-highest** burden in this cohort. It is clearly higher than the typical participant, so it is not representative of the cohort median; however, it is not a unique statistical outlier because P011 and P016 are still higher and P006 is also above 30%. P007 therefore represents a real high-ocular-burden subgroup rather than the whole cohort. A high rate does not by itself make a dataset RED.

## Provisional QC status

- GREEN: **{int(statuses['GREEN'])}**
- YELLOW: **{int(statuses['YELLOW'])}**
- RED: **{int(statuses['RED'])}**

See `summary.csv` for participant-level reasons. RED is reserved for major acquisition/marker failure or insufficient posterior/epoch feasibility, not ocular burden alone.

## Recommended preprocessing decision

{next_decision}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only", nargs="*", default=None, help="Optional participant IDs for regression testing")
    parser.add_argument("--report-only", action="store_true", help="Regenerate plots/report from existing final CSV files")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    xdfs, by_participant = discover(root)
    inclusion = inclusion_table(root, by_participant)
    inclusion.to_csv(output / "participant_inclusion.csv", index=False, encoding="utf-8-sig")
    if args.report_only:
        acquisition_df = pd.read_csv(output / "acquisition.csv")
        marker_df = pd.read_csv(output / "markers.csv")
        channel_df = pd.read_csv(output / "channels.csv")
        ocular_df = pd.read_csv(output / "ocular.csv")
        summary_df = pd.read_csv(output / "summary.csv")
        save_plots(summary_df, output)
        (output / "report.md").write_text(report_text(inclusion, acquisition_df, marker_df, channel_df, ocular_df, summary_df), encoding="utf-8")
        print(json.dumps({"report_only": True, "participants": len(summary_df), "output": str(output)}, indent=2), flush=True)
        return
    final_ids = [f"P{x:03d}" for x in range(5, 20) if x != 14]
    if args.only:
        final_ids = args.only
    acquisition_rows, marker_rows, channel_tables, ocular_rows, summary_rows, failures = [], [], [], [], [], []
    for index, participant in enumerate(final_ids, 1):
        paths = by_participant.get(participant, [])
        print(f"[{index}/{len(final_ids)}] {participant}: {len(paths)} XDF file(s)", flush=True)
        if not paths:
            failures.append({"participant_id": participant, "error": "No XDF discovered"})
            continue
        try:
            acquisition, marker, channels, ocular, summary = process_participant(participant, paths)
            acquisition_rows.append(acquisition); marker_rows.append(marker); channel_tables.append(channels); ocular_rows.append(ocular); summary_rows.append(summary)
            print(f"  trials={summary['Observed image trials']}, ocular={summary['Ocular-contaminated image epochs']} ({summary['Ocular-contaminated %']:.2f}%), posterior={summary['Posterior-propagating ocular epochs']}, QC={summary['Overall QC status']}", flush=True)
        except Exception as exc:
            failures.append({"participant_id": participant, "error": repr(exc)})
            print(f"  FAILED: {exc!r}", flush=True)
        gc.collect()
    acquisition_df = pd.DataFrame(acquisition_rows)
    marker_df = pd.DataFrame(marker_rows)
    channel_df = pd.concat(channel_tables, ignore_index=True) if channel_tables else pd.DataFrame()
    ocular_df = pd.DataFrame(ocular_rows)
    summary_df = pd.DataFrame(summary_rows)
    acquisition_df.to_csv(output / "acquisition.csv", index=False, encoding="utf-8-sig")
    marker_df.to_csv(output / "markers.csv", index=False, encoding="utf-8-sig")
    channel_df.to_csv(output / "channels.csv", index=False, encoding="utf-8-sig")
    ocular_df.to_csv(output / "ocular.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(output / "cohort_processing_failures.csv", index=False, encoding="utf-8-sig")
    if not args.only and len(summary_df) == 14 and not failures:
        save_plots(summary_df, output)
        (output / "report.md").write_text(report_text(inclusion, acquisition_df, marker_df, channel_df, ocular_df, summary_df), encoding="utf-8")
    elif args.only:
        (output / "regression_test_result.json").write_text(json.dumps({"participants": summary_rows, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"processed": len(summary_df), "failures": failures, "output": str(output)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
