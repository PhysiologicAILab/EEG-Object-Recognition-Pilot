"""Crash-resistant CSV and session output management."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import config


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def safe_id(value: str) -> str:
    cleaned = "".join(char for char in value.strip() if char.isalnum() or char in "-_")
    if not cleaned:
        raise ValueError("Participant and session IDs cannot be empty")
    return cleaned


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Iterable[str] | None = None) -> None:
    records = list(rows)
    columns = list(fieldnames or [])
    if not columns and records:
        columns = list(records[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


class SessionDataManager:
    def __init__(
        self, participant_id: str, session_id: str, output_root: Path | None = None
    ) -> None:
        self.participant_id = safe_id(participant_id)
        self.session_id = safe_id(session_id)
        root = output_root or config.OUTPUTS_DIR
        self.session_dir = root / f"sub-{self.participant_id}" / f"ses-{self.session_id}"
        self.audio_dir = self.session_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.behaviour_rows: list[dict] = []
        self.event_rows: list[dict] = []
        self.audio_rows: list[dict] = []
        self.run_timing_summaries: dict[str, dict] = {}
        self.event_log_dirty = False
        self.event_rows_saved = 0
        self.event_log_initialised = False
        self.started_iso = iso_now()

    def initialise(self, manifest: list[dict], experiment_config: dict, seed: int) -> None:
        write_csv(self.session_dir / "trial_manifest.csv", manifest)
        shutil.copy2(config.TRIGGER_CODEBOOK_FILE, self.session_dir / "trigger_codebook.csv")
        (self.session_dir / "experiment_config.json").write_text(
            json.dumps(experiment_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.session_dir / "random_seed.txt").write_text(f"{seed}\n", encoding="ascii")
        (self.session_dir / "protocol_version.txt").write_text(
            f"{config.PROTOCOL_VERSION}\n", encoding="ascii"
        )
        notes = self.session_dir / "session_notes_template.txt"
        if not notes.exists():
            notes.write_text(
                "Participant comfort:\nEEG notes:\nTrigger notes:\nAudio/microphone notes:\n"
                "Stimulus issues:\nBreaks or interruptions:\nOther observations:\n",
                encoding="utf-8",
            )
        self.save_behaviour(autosave=True)
        self.save_audio_manifest()
        if not self.event_log_initialised:
            write_csv(self.session_dir / "event_log.csv", [], config.EVENT_FIELDS)
            self.event_log_initialised = True

    def save_asset_report(self, rows: list[dict]) -> None:
        versioned_rows = [{"protocol_version": config.PROTOCOL_VERSION, **row} for row in rows]
        write_csv(
            self.session_dir / "asset_check_report.csv",
            versioned_rows,
            ("protocol_version", "asset_type", "path", "exists", "format_ok", "psychopy_load_ok", "status", "details"),
        )

    def add_event(self, row: dict) -> None:
        record = {field: row.get(field, "") for field in config.EVENT_FIELDS}
        self.event_rows.append(record)
        self.event_log_dirty = True

    def save_event_log(self) -> None:
        """Flush markers at trial/run boundaries, never inside a timed display phase."""
        path = self.session_dir / "event_log.csv"
        if not self.event_log_initialised:
            write_csv(path, [], config.EVENT_FIELDS)
            self.event_log_initialised = True
        if not self.event_log_dirty:
            return
        pending = self.event_rows[self.event_rows_saved:]
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=config.EVENT_FIELDS, extrasaction="ignore")
            writer.writerows(pending)
        self.event_rows_saved = len(self.event_rows)
        self.event_log_dirty = False

    def add_behaviour(self, row: dict) -> None:
        record = {field: row.get(field, "") for field in config.BEHAVIOUR_FIELDS}
        self.behaviour_rows.append(record)
        self.save_behaviour(autosave=True)

    def add_audio(self, row: dict) -> None:
        self.audio_rows.append(row)
        self.save_audio_manifest()

    def save_behaviour(self, autosave: bool = False) -> None:
        filename = "behaviour_autosave.csv" if autosave else "behaviour.csv"
        write_csv(self.session_dir / filename, self.behaviour_rows, config.BEHAVIOUR_FIELDS)

    def save_audio_manifest(self) -> None:
        write_csv(self.session_dir / "audio_manifest.csv", self.audio_rows, config.AUDIO_MANIFEST_FIELDS)

    def save_run_timing_summary(self, run_id: str, summary: dict) -> None:
        self.run_timing_summaries[str(run_id)] = summary
        (self.session_dir / "run_timing_summary.json").write_text(
            json.dumps(self.run_timing_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def finalise(self, status: str, summary: dict) -> None:
        self.save_behaviour(autosave=False)
        self.save_behaviour(autosave=True)
        self.save_audio_manifest()
        self.save_event_log()
        payload = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "protocol_version": config.PROTOCOL_VERSION,
            "data_compatibility_warning": (
                "Do not pool this 70% Latin-square microphone-fixed protocol with previous 65% "
                "pilot sessions without an explicit analysis decision."
            ),
            "status": status,
            "started_iso": self.started_iso,
            "finished_iso": iso_now(),
            "completed_behaviour_rows": len(self.behaviour_rows),
            "event_rows": len(self.event_rows),
            "audio_rows": len(self.audio_rows),
            **summary,
        }
        (self.session_dir / "session_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
