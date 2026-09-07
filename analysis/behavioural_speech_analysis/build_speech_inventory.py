from __future__ import annotations

import hashlib
import os
import re
import wave
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
RAW_BEHAVIOURAL = Path(
    os.environ.get(
        "EEG_RAW_BEHAVIOURAL_ROOT",
        PROJECT_ROOT / "raw_data" / "behavioural_audio",
    )
).resolve()
OUT = Path(
    os.environ.get(
        "BEHAVIOURAL_ANALYSIS_ROOT",
        PROJECT_ROOT / "processed_data" / "behavioural_speech_analysis",
    )
).resolve()
VALID_LABELS = {"bowl", "cup", "plate", "knife", "fork", "spoon"}
CONDITIONS = [
    "visible_none", "visible_congruent", "visible_incongruent",
    "occluded_none", "occluded_congruent", "occluded_incongruent",
]
SEED = 20260829


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_num(value):
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def inspect_wav(path: Path) -> dict:
    result = {
        "wav_readable": False, "wav_error": "", "wav_file_size_bytes": path.stat().st_size,
        "wav_sha256": "", "wav_sample_rate_hz": np.nan, "wav_channels": np.nan,
        "wav_sample_width_bytes": np.nan, "wav_frames": np.nan, "wav_duration_s_header": np.nan,
    }
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            frames = w.getnframes()
            result.update({
                "wav_readable": True, "wav_sample_rate_hz": rate, "wav_channels": w.getnchannels(),
                "wav_sample_width_bytes": w.getsampwidth(), "wav_frames": frames,
                "wav_duration_s_header": frames / rate if rate else np.nan,
            })
        result["wav_sha256"] = file_sha256(path)
    except Exception as exc:
        result["wav_error"] = f"{type(exc).__name__}: {exc}"
    return result


def event_summary(event_path: Path) -> pd.DataFrame:
    ev = pd.read_csv(event_path, dtype=str, low_memory=False)
    ev = ev[ev.event_name.isin(["microphone_recording_start", "response_cue_onset"])].copy()
    ev["trial_index_global"] = pd.to_numeric(ev.trial_index_global, errors="coerce")
    ev = ev[ev.trial_index_global > 0]
    rows = []
    for (run_id, global_trial), g in ev.groupby(["run_id", "trial_index_global"], dropna=False):
        row = {"run_id": str(run_id), "trial_index_global": int(global_trial)}
        for event_name, prefix in [("microphone_recording_start", "mic"), ("response_cue_onset", "cue")]:
            x = g[g.event_name.eq(event_name)]
            row[f"{prefix}_event_count"] = len(x)
            row[f"{prefix}_psychopy_timestamp"] = pd.to_numeric(x.psychopy_timestamp, errors="coerce").iloc[0] if len(x) else np.nan
            row[f"{prefix}_lsl_timestamp"] = pd.to_numeric(x.lsl_timestamp, errors="coerce").iloc[0] if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_source(participant: str, root: Path, run_offset: int, source_label: str) -> pd.DataFrame:
    behaviour_path = root / "behaviour.csv"
    audio_path = root / "audio_manifest.csv"
    event_path = root / "event_log.csv"
    audio_dir = root / "audio"
    behaviour = pd.read_csv(behaviour_path, dtype=str, low_memory=False)
    behaviour = behaviour[pd.to_numeric(behaviour.is_practice, errors="coerce").eq(0)].copy()
    audio = pd.read_csv(audio_path, dtype=str, low_memory=False)
    audio = audio[pd.to_numeric(audio.trial_index_global, errors="coerce").gt(0)].copy()
    events = event_summary(event_path)

    behaviour["trial_index_global_num"] = pd.to_numeric(behaviour.trial_index_global, errors="coerce")
    behaviour["trial_index_run_num"] = pd.to_numeric(behaviour.trial_index_run, errors="coerce")
    audio["trial_index_global_num"] = pd.to_numeric(audio.trial_index_global, errors="coerce")
    event_key = events.rename(columns={"trial_index_global": "trial_index_global_num"})

    audio_cols = [
        "run_id", "trial_index_global_num", "final_wav_path", "recording_segment_start_time",
        "response_cue_time", "recording_segment_stop_time", "wav_duration_ms", "sample_count",
        "peak_amplitude", "wav_sample_rate", "wav_channels", "recording_status",
    ]
    audio_small = audio[audio_cols].copy()
    audio_small = audio_small.rename(columns={c: f"audio_manifest_{c}" for c in audio_cols if c not in {"run_id", "trial_index_global_num"}})
    merged = behaviour.merge(audio_small, on=["run_id", "trial_index_global_num"], how="left", validate="one_to_one")
    merged = merged.merge(event_key, on=["run_id", "trial_index_global_num"], how="left", validate="one_to_one")

    wavs = list(audio_dir.glob("*.wav"))
    wav_by_name = {}
    for path in wavs:
        wav_by_name.setdefault(path.name, []).append(path)

    rows = []
    for _, r in merged.iterrows():
        stored_run = int(r.run_id)
        actual_run = stored_run + run_offset
        trial_in_run = int(r.trial_index_run_num)
        actual_global = (actual_run - 1) * 72 + trial_in_run
        candidate_names = []
        for col in ["response_wav_filename", "response_wav_path", "audio_manifest_final_wav_path"]:
            value = r.get(col)
            if pd.notna(value) and str(value).strip():
                candidate_names.append(Path(str(value)).name)
        candidate_names = list(dict.fromkeys(candidate_names))
        matches = [p for name in candidate_names for p in wav_by_name.get(name, [])]
        matches = list(dict.fromkeys(matches))
        wav_path = matches[0] if len(matches) == 1 else None
        condition = f"{r.visibility}_{r.audio_condition}"
        target = str(r.correct_response).strip().lower()
        row = {
            "participant": participant, "actual_run": actual_run, "trial_in_run": trial_in_run,
            "actual_global_trial": actual_global, "stored_run_id": r.run_id,
            "stored_global_trial": int(r.trial_index_global_num), "source_label": source_label,
            "source_root": str(root), "behaviour_csv": str(behaviour_path),
            "audio_manifest_csv": str(audio_path), "event_log_csv": str(event_path),
            "wav_file": str(wav_path) if wav_path else "", "wav_basename": wav_path.name if wav_path else (candidate_names[0] if candidate_names else ""),
            "wav_candidate_count": len(matches), "mapping_status": "MATCHED" if len(matches) == 1 else ("DUPLICATE_CANDIDATES" if len(matches) > 1 else "NO_WAV_MATCH"),
            "target_label": target, "target_label_valid": target in VALID_LABELS,
            "visibility": str(r.visibility), "audio_condition": str(r.audio_condition), "condition": condition,
            "image_object": str(r.image_object), "accepted_responses": str(r.accepted_responses),
            "microphone_start_psychopy_s": as_num(r.recording_segment_start_time),
            "response_cue_psychopy_s": as_num(r.response_cue_time),
            "microphone_stop_psychopy_s": as_num(r.recording_segment_stop_time),
            "microphone_start_lsl_s": as_num(r.get("mic_lsl_timestamp")),
            "response_cue_lsl_s": as_num(r.get("cue_lsl_timestamp")),
            "mic_event_count": as_num(r.get("mic_event_count")), "cue_event_count": as_num(r.get("cue_event_count")),
            "behaviour_recording_status": str(r.recording_status),
            "audio_manifest_recording_status": str(r.get("audio_manifest_recording_status")),
            "manifest_wav_duration_ms": as_num(r.get("audio_manifest_wav_duration_ms")),
            "manifest_sample_count": as_num(r.get("audio_manifest_sample_count")),
            "manifest_sample_rate_hz": as_num(r.get("audio_manifest_wav_sample_rate")),
            "manifest_channels": as_num(r.get("audio_manifest_wav_channels")),
            "run_offset_rule_applied": run_offset,
        }
        row["cue_position_psychopy_s"] = row["response_cue_psychopy_s"] - row["microphone_start_psychopy_s"]
        row["cue_position_lsl_s"] = row["response_cue_lsl_s"] - row["microphone_start_lsl_s"]
        row["cue_position_clock_difference_ms"] = 1000 * (row["cue_position_psychopy_s"] - row["cue_position_lsl_s"])
        if wav_path:
            row.update(inspect_wav(wav_path))
            name_target = re.search(r"_object-([a-z]+)_", wav_path.name, flags=re.I)
            name_condition = re.search(r"_condition-([a-z_]+)\.wav$", wav_path.name, flags=re.I)
            row["filename_target_label"] = name_target.group(1).lower() if name_target else ""
            row["filename_condition"] = name_condition.group(1).lower() if name_condition else ""
        else:
            row.update({"wav_readable": False, "wav_error": "missing WAV match", "wav_file_size_bytes": np.nan,
                        "wav_sha256": "", "wav_sample_rate_hz": np.nan, "wav_channels": np.nan,
                        "wav_sample_width_bytes": np.nan, "wav_frames": np.nan, "wav_duration_s_header": np.nan,
                        "filename_target_label": "", "filename_condition": ""})
        row["filename_target_matches_manifest"] = row["filename_target_label"] == target
        row["filename_condition_matches_manifest"] = row["filename_condition"] == condition
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    sources = []
    # P005's first four actual runs are in source_1; run 4 ended after trial 54 for WAV recording.
    sources.append(load_source("P005", RAW_BEHAVIOURAL / "P005" / "source_1" / "ses-005", 0, "P005_actual_runs_01_04"))
    # P005's later acquisition reset stored run IDs to 01/02; the documented project mapping restores actual runs 05/06.
    sources.append(load_source("P005", RAW_BEHAVIOURAL / "P005" / "source_2", 4, "P005_stored_runs_01_02_mapped_to_actual_05_06"))
    for number in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20]:
        session_dir = RAW_BEHAVIOURAL / f"P{number:03d}" / "session_source"
        sources.append(load_source(f"P{number:03d}", session_dir, 0, "standard_final_session"))
    inventory = pd.concat(sources, ignore_index=True)
    inventory = inventory.sort_values(["participant", "actual_run", "trial_in_run"]).reset_index(drop=True)

    key_cols = ["participant", "actual_run", "trial_in_run"]
    inventory["duplicate_trial_key"] = inventory.duplicated(key_cols, keep=False)
    inventory["duplicate_audio_content"] = inventory.wav_sha256.ne("") & inventory.duplicated("wav_sha256", keep=False)
    inventory["cue_position_valid"] = inventory.cue_position_psychopy_s.between(0.2, 1.0)
    inventory["event_timestamp_pair_complete"] = inventory.microphone_start_lsl_s.notna() & inventory.response_cue_lsl_s.notna()
    inventory["metadata_qc_flag"] = "OK"
    flag_rules = [
        (inventory.mapping_status.ne("MATCHED"), "unmatched_wav"),
        (~inventory.wav_readable, "unreadable_wav"),
        (inventory.duplicate_trial_key, "duplicate_trial_key"),
        (inventory.duplicate_audio_content, "duplicate_audio_content"),
        (~inventory.target_label_valid, "invalid_target_label"),
        (~inventory.filename_target_matches_manifest, "filename_target_mismatch"),
        (~inventory.filename_condition_matches_manifest, "filename_condition_mismatch"),
        (~inventory.cue_position_valid, "implausible_cue_position"),
        (~inventory.event_timestamp_pair_complete, "missing_lsl_event_timestamp_pair"),
    ]
    for mask, label in flag_rules:
        inventory.loc[mask, "metadata_qc_flag"] = inventory.loc[mask, "metadata_qc_flag"].map(lambda x: label if x == "OK" else x + ";" + label)

    expected_keys = pd.DataFrame(
        [(f"P{p:03d}", run, trial) for p in [5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20] for run in range(1, 7) for trial in range(1, 73)],
        columns=key_cols,
    )
    present_keys = inventory[key_cols].drop_duplicates()
    missing = expected_keys.merge(present_keys, on=key_cols, how="left", indicator=True)
    missing = missing[missing._merge.eq("left_only")].drop(columns="_merge")
    missing["expected_global_trial"] = (missing.actual_run - 1) * 72 + missing.trial_in_run
    missing["reason"] = "No completed behaviour/audio-manifest row and no formal WAV found"

    audit_rows = []
    for participant, g in inventory.groupby("participant"):
        miss = missing[missing.participant.eq(participant)]
        audit_rows.append({
            "participant": participant, "formal_wavs_found": len(g), "successfully_matched_wavs": int(g.mapping_status.eq("MATCHED").sum()),
            "unmatched_wavs": int(g.mapping_status.ne("MATCHED").sum()), "expected_formal_trials": 432,
            "missing_expected_wavs": len(miss), "unreadable_or_corrupt_wavs": int((~g.wav_readable).sum()),
            "duplicate_trial_keys": int(g.duplicate_trial_key.sum()), "byte_identical_duplicate_wavs": int(g.duplicate_audio_content.sum()),
            "target_mismatches_filename_vs_manifest": int((~g.filename_target_matches_manifest).sum()),
            "condition_mismatches_filename_vs_manifest": int((~g.filename_condition_matches_manifest).sum()),
            "missing_lsl_event_timestamp_pairs": int((~g.event_timestamp_pair_complete).sum()),
            "cue_position_median_ms": 1000 * g.cue_position_psychopy_s.median(),
            "cue_position_min_ms": 1000 * g.cue_position_psychopy_s.min(), "cue_position_max_ms": 1000 * g.cue_position_psychopy_s.max(),
            "conditions_present": ";".join(sorted(g.condition.unique())),
            "data_sources": ";".join(sorted(g.source_root.unique())),
        })
    audit = pd.DataFrame(audit_rows)

    rng = np.random.default_rng(SEED)
    selected_indices = []
    eligible = inventory[inventory.mapping_status.eq("MATCHED") & inventory.wav_readable & inventory.condition.isin(CONDITIONS)]
    for participant in sorted(eligible.participant.unique()):
        for condition in CONDITIONS:
            pool = eligible[(eligible.participant.eq(participant)) & (eligible.condition.eq(condition))]
            if len(pool):
                selected_indices.append(int(rng.choice(pool.index.to_numpy())))
    remaining = eligible.drop(index=selected_indices)
    extra_n = max(0, 100 - len(selected_indices))
    if extra_n:
        selected_indices.extend(rng.choice(remaining.index.to_numpy(), size=extra_n, replace=False).astype(int).tolist())
    validation = inventory.loc[selected_indices].copy().sort_values(["participant", "condition", "actual_run", "trial_in_run"])
    validation.insert(0, "validation_id", [f"VAL{i:03d}" for i in range(1, len(validation) + 1)])
    validation["validation_random_seed"] = SEED
    validation["selection_rule"] = "one WAV per participant×condition, plus 10 seeded random extras"

    OUT.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT / "wav_inventory_and_mapping.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "data_structure_audit.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(OUT / "missing_expected_wavs.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUT / "validation_subset_manifest.csv", index=False, encoding="utf-8-sig")

    report = f"""# Behavioural speech data-structure report

## Scope

Only FINAL-protocol participants P005–P020 (P014 absent) were included. Pilot sessions P003/P004 and practice WAVs were excluded. Original WAV/CSV/XDF files were read only.

## Trial correspondence

- Primary mapping source: each session's `behaviour.csv`, joined one-to-one to `audio_manifest.csv` and the actual WAV basename.
- Target object: `correct_response` in `behaviour.csv` (cross-checked against `image_object` and the WAV filename object field).
- Visibility and Audio conditions: `visibility` and `audio_condition` in `behaviour.csv`.
- Microphone start and response cue: PsychoPy timestamps in `behaviour.csv`; LSL timestamps were independently attached from `event_log.csv` when available.
- Cue position in the WAV: `response_cue_time - recording_segment_start_time`; it was not assumed to be exactly 0.5 s.

## P005 mapping exception

P005 is split across `raw_data/behavioural_audio/P005/source_1` and `source_2`. The second source contains stored run IDs 01/02; the documented project mapping identifies these as actual runs 5/6. A fixed +4 run offset was applied only to those records. Original filenames and CSVs were not changed.

## Counts

- Formal WAV files found: **{len(inventory):,}**.
- Successfully matched to behaviour/audio-manifest trials: **{int(inventory.mapping_status.eq('MATCHED').sum()):,}**.
- Unmatched formal WAVs: **{int(inventory.mapping_status.ne('MATCHED').sum()):,}**.
- Expected formal trials without a WAV: **{len(missing):,}**.
- Unreadable/corrupted WAVs: **{int((~inventory.wav_readable).sum()):,}**.
- Duplicate participant/run/trial keys: **{int(inventory.duplicate_trial_key.sum()):,}**.
- Byte-identical duplicate WAV records: **{int(inventory.duplicate_audio_content.sum()):,}**.

The 100-trial validation subset was selected with seed {SEED}: one trial from every participant × six-condition cell (90 trials), then 10 additional random eligible trials. No ASR outcome or expected accuracy was used in sampling.
"""
    (OUT / "step1_data_structure_report.md").write_text(report, encoding="utf-8")
    print({
        "formal_wavs": len(inventory), "matched": int(inventory.mapping_status.eq("MATCHED").sum()),
        "unmatched": int(inventory.mapping_status.ne("MATCHED").sum()), "missing_expected": len(missing),
        "unreadable": int((~inventory.wav_readable).sum()), "duplicate_keys": int(inventory.duplicate_trial_key.sum()),
        "duplicate_audio": int(inventory.duplicate_audio_content.sum()), "validation_n": len(validation),
        "output": str(OUT),
    })


if __name__ == "__main__":
    main()
