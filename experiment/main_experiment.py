"""PsychoPy Coder entry point for the delayed object-naming EEG experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
import platform
import secrets
import statistics
import sys
import traceback
from datetime import datetime, timezone

from PIL import Image

import config
from data_manager import SessionDataManager, iso_now, write_csv
from stimulus_manager import OcclusionApprovalPending
from trial_generator import (
    build_manifests,
    latin_square_order,
    participant_number_from_id,
    load_objects,
)


class AbortExperiment(Exception):
    """Raised after the researcher confirms an Escape-key abort."""


def measure_refresh_rate(win, warmup_frames: int = 30, sample_frames: int = 180) -> dict:
    """Measure refresh directly from flip timestamps; PsychoPy's estimate can halve high-Hz displays."""
    for _ in range(warmup_frames):
        win.flip()
    flip_times = [win.flip() for _ in range(sample_frames)]
    intervals = [right - left for left, right in zip(flip_times, flip_times[1:])]
    plausible = [interval for interval in intervals if 0.002 <= interval <= 0.05]
    if len(plausible) < sample_frames * 0.8:
        raise RuntimeError("Display refresh calibration failed: too many invalid flip intervals")
    median_interval = statistics.median(plausible)
    refresh_hz = 1.0 / median_interval
    absolute_deviations = [abs(interval - median_interval) for interval in plausible]
    relative_mad = statistics.median(absolute_deviations) / median_interval
    if not config.MIN_PLAUSIBLE_REFRESH_HZ <= refresh_hz <= config.MAX_PLAUSIBLE_REFRESH_HZ:
        raise RuntimeError(f"Display refresh calibration returned an invalid rate: {refresh_hz:.3f} Hz")
    return {
        "refresh_hz": refresh_hz,
        "median_frame_interval_ms": median_interval * 1000,
        "minimum_frame_interval_ms": min(plausible) * 1000,
        "maximum_frame_interval_ms": max(plausible) * 1000,
        "valid_interval_count": len(plausible),
        "sample_frames": sample_frames,
        "relative_median_absolute_deviation": relative_mad,
    }


def validate_session_refresh_rate(calibration: dict) -> None:
    """Accept any plausible stable fullscreen rate; no monitor rate is hard-coded."""
    measured = float(calibration["refresh_hz"])
    relative_mad = float(calibration.get("relative_median_absolute_deviation", 0.0))
    if not config.MIN_PLAUSIBLE_REFRESH_HZ <= measured <= config.MAX_PLAUSIBLE_REFRESH_HZ:
        raise RuntimeError(
            f"Fullscreen refresh-rate check failed: implausible measured rate {measured:.3f} Hz."
        )
    if relative_mad > config.REFRESH_CALIBRATION_MAX_RELATIVE_MAD:
        raise RuntimeError(
            f"Fullscreen refresh-rate check failed: unstable flip intervals "
            f"(relative MAD {relative_mad:.4f})."
        )


def validate_observed_session_refresh(rows: list[dict], session_refresh_hz: float) -> float:
    """Passively detect substantial drift from trial flip counts; performs no recalibration."""
    estimates = []
    phase_fields = (
        ("fixation_frame_count", "actual_fixation_ms"),
        ("image_frame_count", "actual_image_ms"),
        ("silent_delay_frame_count", "actual_silent_delay_ms"),
        ("response_frame_count", "actual_response_ms"),
    )
    for row in rows:
        for count_field, duration_field in phase_fields:
            count = float(row.get(count_field, 0) or 0)
            duration = float(row.get(duration_field, 0) or 0)
            if count > 0 and duration > 0:
                estimates.append(count / (duration / 1000.0))
    if not estimates:
        return float(session_refresh_hz)
    observed = statistics.median(estimates)
    drift_percent = abs(observed - session_refresh_hz) / session_refresh_hz * 100.0
    if drift_percent > config.SESSION_REFRESH_DRIFT_TOLERANCE_PERCENT:
        raise RuntimeError(
            f"Session refresh changed substantially: baseline {session_refresh_hz:.3f} Hz, "
            f"passive observed estimate {observed:.3f} Hz ({drift_percent:.1f}% drift)."
        )
    return observed


def evaluate_trial_timing(actual_by_phase: dict[str, float], tolerance_ms: float) -> dict:
    """Return analysis flags without changing or fabricating observed timestamps."""
    planned_by_phase = {
        "fixation": float(config.FIXATION_MS),
        "image": float(config.IMAGE_DURATION_MS),
        "silent_delay": float(config.SILENT_DELAY_MS),
        "response": float(config.RESPONSE_WINDOW_MS),
    }
    violations = []
    for phase, planned in planned_by_phase.items():
        actual = float(actual_by_phase[phase])
        error = actual - planned
        if abs(error) > tolerance_ms:
            violations.append({"phase": phase, "error_ms": error, "planned_ms": planned, "actual_ms": actual})
    largest = max(violations, key=lambda item: abs(item["error_ms"])) if violations else None
    return {
        "timing_violation": int(bool(violations)),
        "timing_violation_phase": "|".join(item["phase"] for item in violations),
        "timing_error_ms": round(largest["error_ms"], 3) if largest else 0.0,
        "timing_violation_reason": "display_duration_outside_tolerance" if violations else "",
        "violations": violations,
    }


def build_timing_qc(rows: list[dict]) -> dict:
    formal = [row for row in rows if not bool(int(row.get("is_practice", 0) or 0))]
    affected = []
    for row in formal:
        if int(row.get("timing_violation", 0) or 0) != 1:
            continue
        phase = str(row.get("timing_violation_phase", ""))
        planned_lookup = {
            "fixation": row.get("planned_fixation_ms", ""),
            "image": row.get("planned_image_ms", ""),
            "silent_delay": row.get("planned_silent_delay_ms", ""),
            "response": row.get("planned_response_ms", ""),
        }
        actual_lookup = {
            "fixation": row.get("actual_fixation_ms", ""),
            "image": row.get("actual_image_ms", ""),
            "silent_delay": row.get("actual_silent_delay_ms", ""),
            "response": row.get("actual_response_ms", ""),
        }
        primary_phase = phase.split("|")[0] if phase else ""
        affected.append({
            "run": row.get("run_id", ""),
            "trial_in_run": row.get("trial_index_run", ""),
            "global_trial": row.get("trial_index_global", ""),
            "condition": f"{row.get('visibility', '')}_{row.get('audio_condition', '')}",
            "object": row.get("image_object", ""),
            "image_filename": row.get("image_path", ""),
            "phase": phase,
            "planned_duration_ms": planned_lookup.get(primary_phase, ""),
            "actual_duration_ms": actual_lookup.get(primary_phase, ""),
            "error_ms": row.get("timing_error_ms", ""),
        })
    total = len(formal)
    violation_count = len(affected)
    return {
        "total_formal_trials": total,
        "timing_valid_trials": total - violation_count,
        "timing_violation_trials": violation_count,
        "timing_valid_percentage": round((total - violation_count) / total * 100, 3) if total else 0.0,
        "affected_trials": affected,
    }


def select_run_start_validation_trials(main_trials: list[dict]) -> list[dict]:
    """Keep the first formal trials of every run without renumbering the manifest."""
    limit = config.RUN_START_VALIDATION_TRIALS_PER_RUN
    selected = [row for row in main_trials if int(row["trial_index_run"]) <= limit]
    counts = Counter(str(row["run_id"]) for row in selected)
    expected = {f"{run_id:02d}": limit for run_id in range(1, config.FULL_RUNS + 1)}
    if counts != expected:
        raise RuntimeError(f"Invalid run-start validation subset: {dict(counts)}")
    return selected


def pure_validation() -> None:
    """Run checks that do not open a PsychoPy window or audio device."""
    from stimulus_manager import assert_preflight_passed, run_asset_preflight

    rows = run_asset_preflight(psychopy_load=False, require_approved_occlusions=False)
    assert_preflight_passed(rows)
    for mode, expected in (("pilot", config.PILOT_MAIN_TRIALS), ("full", config.FULL_MAIN_TRIALS)):
        practice, main = build_manifests("P001", "01", mode, 240513)
        assert len(practice) == config.PRACTICE_TRIALS
        assert len(main) == expected
    print(json.dumps({
        "validation": "passed",
        "visible_images": sum(row["asset_type"] == "visible_image" for row in rows),
        "occluded_images": sum(row["asset_type"] == "occluded_image" for row in rows),
        "spoken_audio": sum(row["asset_type"] == "spoken_audio" for row in rows),
        "practice_trials": config.PRACTICE_TRIALS,
        "pilot_main_trials": config.PILOT_MAIN_TRIALS,
        "full_main_trials": config.FULL_MAIN_TRIALS,
        "full_runs": config.FULL_RUNS,
        "occlusion": f"{config.OCCLUSION_STYLE} {config.OCCLUSION_LEVEL}%",
        "protocol_version": config.PROTOCOL_VERSION,
    }, indent=2))


def get_setup(gui) -> dict | None:
    dialog = gui.Dlg(title="EEG Object Naming - Researcher Setup")
    dialog.addField("Participant ID:", "")
    dialog.addField("Session ID:", "01")
    dialog.addField("Language:", choices=["English", "中文"])
    dialog.addField("Mode:", choices=["Pilot", "Full", "Run-start validation"])
    dialog.addField("Fullscreen:", choices=["on", "off"])
    values = dialog.show()
    if not dialog.OK:
        return None
    participant_id = str(values[0]).strip()
    mode = str(values[3]).lower().replace("-", "_").replace(" ", "_")
    latin_row_override = None
    if mode in ("full", "run_start_validation") and participant_number_from_id(participant_id) is None:
        row_dialog = gui.Dlg(title="Latin-square assignment required")
        row_dialog.addText("Participant ID has no numeric index. Enter the assigned row (1-6).")
        row_dialog.addField("Latin-square row:", choices=["1", "2", "3", "4", "5", "6"])
        row_values = row_dialog.show()
        if not row_dialog.OK:
            return None
        latin_row_override = int(row_values[0])
    return {
        "participant_id": participant_id,
        "session_id": str(values[1]).strip(),
        "language": "zh" if values[2] == "中文" else "en",
        "mode": mode,
        "fullscreen": values[4] == "on",
        "latin_row_override": latin_row_override,
    }


class ExperimentRunner:
    def __init__(self, setup: dict) -> None:
        from psychopy import core, event, logging, visual

        self.core = core
        self.event = event
        self.visual = visual
        self.logging = logging
        self.setup = dict(setup)
        self.validation_mode = self.setup["mode"] == "run_start_validation"
        self.requested_session_id = self.setup["session_id"]
        if self.validation_mode:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.setup["session_id"] = f"{self.requested_session_id}-validation-{stamp}"
        self.seed = secrets.randbits(31)
        manifest_mode = "full" if self.validation_mode else self.setup["mode"]
        self.practice, self.main_trials = build_manifests(
            self.setup["participant_id"], self.setup["session_id"], manifest_mode, self.seed,
            self.setup.get("latin_row_override"),
        )
        self.validation_source_manifest: list[dict] = []
        if self.validation_mode:
            self.validation_source_manifest = list(self.main_trials)
            self.practice = []
            self.main_trials = select_run_start_validation_trials(self.main_trials)
        self.objects = load_objects()
        output_root = config.VALIDATION_OUTPUTS_DIR if self.validation_mode else None
        self.data = SessionDataManager(
            self.setup["participant_id"], self.setup["session_id"], output_root=output_root
        )
        self.global_clock = core.Clock()
        self.win = None
        self.refresh_calibration: dict = {}
        self.frame_rate = 60.0
        self.frame_duration = 1.0 / self.frame_rate
        self.pending_events: list[dict] = []
        self.marker_errors: list[str] = []
        self.marker = None
        self.microphone = None
        self.stimuli = None
        self.cached_sounds: dict[str, object] = {}
        self.prepared_run_id: str | None = None
        self.run_warmup_records: list[dict] = []
        self.run_dropped_frames: dict[str, int] = {}
        self.session_window_id: int | None = None
        self.current_audio_path = None
        self.current_trial = None
        self.current_recording_started_iso = ""
        self.current_recording_start_time = ""
        self.last_focus = None
        self.last_fullscreen = False
        self.timing_warning_runs: set[str] = set()
        self.log_file = logging.LogFile(
            str(self.data.session_dir / "psychopy_log.txt"), level=logging.INFO, filemode="w"
        )
        logging.console.setLevel(logging.WARNING)
        logging.info(f"protocol_version={config.PROTOCOL_VERSION}")

        self.open_window(fullscreen=False, calibrate=False)

    def create_visual_components(self) -> None:
        """Create window-bound stimuli after each safe window recreation."""
        visual = self.visual

        self.fixation = visual.TextStim(self.win, text="+", color=config.TEXT_COLOUR, height=0.11)
        self.heading = visual.TextStim(
            self.win, text="", color=config.TEXT_COLOUR, height=0.075,
            wrapWidth=1.65, pos=(0, 0.66), alignText="center", anchorHoriz="center",
        )
        self.message = visual.TextStim(
            self.win, text="", color=config.TEXT_COLOUR, height=0.055,
            wrapWidth=1.55, alignText="center", anchorHoriz="center",
        )
        self.small_message = visual.TextStim(
            self.win, text="", color=config.TEXT_COLOUR, height=0.035,
            wrapWidth=1.6, pos=(0, -0.75), alignText="center",
        )
        self.mic_body = visual.Rect(
            self.win, width=0.11, height=0.22, lineColor="white", fillColor=None,
            lineWidth=4, pos=(0, 0.1)
        )
        self.mic_stem = visual.Line(
            self.win, start=(0, -0.05), end=(0, -0.18), lineColor="white", lineWidth=4
        )
        self.mic_base = visual.Line(
            self.win, start=(-0.09, -0.18), end=(0.09, -0.18), lineColor="white", lineWidth=4
        )

    def open_window(self, fullscreen: bool, calibrate: bool) -> None:
        self.win = self.visual.Window(
            size=config.WINDOW_SIZE,
            fullscr=fullscreen,
            monitor=config.MONITOR_NAME,
            units="norm",
            color=config.BACKGROUND_COLOUR,
            colorSpace="rgb",
            allowGUI=not fullscreen,
            waitBlanking=True,
        )
        self.last_focus = None
        self.last_fullscreen = bool(self.win.fullscr)
        if calibrate:
            self.refresh_calibration = measure_refresh_rate(self.win)
            self.frame_rate = self.refresh_calibration["refresh_hz"]
            self.frame_duration = 1.0 / self.frame_rate
        self.create_visual_components()

    def close_window(self) -> None:
        if self.stimuli is not None:
            self.stimuli.stop_all()
            self.stimuli = None
        if self.win is not None:
            self.win.close()
            self.win = None

    def open_fullscreen_for_run(self, trials: list[dict], formal_run_id: str | None = None) -> None:
        """Open the one participant window and calibrate it once for the session."""
        from stimulus_manager import StimulusCache

        try:
            self.close_window()
            participant_fullscreen = (
                True if self.setup["mode"] in ("full", "run_start_validation")
                else self.setup["fullscreen"]
            )
            self.open_window(fullscreen=participant_fullscreen, calibrate=False)
            image_paths = list(dict.fromkeys(trial["image_path"] for trial in trials))
            self.stimuli = StimulusCache(
                self.win,
                image_paths=image_paths,
                sounds=self.cached_sounds,
            )
            self.cached_sounds = self.stimuli.sounds
            self.stimuli.prepare_textures(image_paths)
            if formal_run_id is None:
                self.refresh_calibration = measure_refresh_rate(
                    self.win,
                    warmup_frames=config.RUN_WARMUP_BLANK_FRAMES,
                    sample_frames=config.RUN_REFRESH_SAMPLE_FRAMES,
                )
                if self.setup["mode"] in ("full", "run_start_validation"):
                    validate_session_refresh_rate(self.refresh_calibration)
                self.frame_rate = self.refresh_calibration["refresh_hz"]
                self.frame_duration = 1.0 / self.frame_rate
                self.session_window_id = id(self.win)
                self.prepared_run_id = None
            else:
                self.prepare_formal_run(trials, formal_run_id)
        except Exception as exc:
            self.close_window()
            self.open_window(fullscreen=False, calibrate=False)
            raise RuntimeError(
                f"Fullscreen/display preparation failed before the run began: {type(exc).__name__}: {exc}"
            ) from exc
        self.local_event(
            "fullscreen_ready", trials[0] if trials else {},
            (
                f"refresh_hz={self.frame_rate:.6f}; preloaded_images={self.stimuli.preloaded_image_count}; "
                f"outlet_source_id={self.marker.info['source_id']}"
            ),
        )

    def open_operator_window(self) -> None:
        self.close_window()
        self.open_window(fullscreen=False, calibrate=False)

    def prepare_formal_run(self, trials: list[dict], run_id: str) -> None:
        """Finish cold-start work before run_start without presenting formal stimuli."""
        if self.session_window_id is None or id(self.win) != self.session_window_id:
            raise RuntimeError("The session fullscreen window changed before a formal run")
        image_paths = list(dict.fromkeys(trial["image_path"] for trial in trials))
        decode_count_before = self.stimuli.image_decode_count
        self.stimuli.prepare_textures(image_paths)
        if self.stimuli.image_decode_count != decode_count_before:
            raise RuntimeError("Run warm-up discovered an image that was not preloaded")

        gc.collect()

        # Finish all pending disk writes before the final blank-flip calibration.
        self.data.save_behaviour(autosave=True)
        self.data.save_audio_manifest()
        self.data.save_event_log()
        record = {
            "run_id": str(run_id),
            "preloaded_image_count": self.stimuli.preloaded_image_count,
            "image_decode_count_during_warmup": self.stimuli.image_decode_count - decode_count_before,
            "prepared_audio_count": len(self.stimuli.sounds),
            "microphone_rewarmed": False,
            "session_refresh_hz": self.frame_rate,
            "refresh_measurements_this_run": 0,
            "session_window_id": self.session_window_id,
            "formal_markers_sent": 0,
        }
        self.run_warmup_records.append(record)
        self.prepared_run_id = str(run_id)
        self.local_event("run_warmup_complete", trials[0], json.dumps(record, separators=(",", ":")))

    def frames(self, milliseconds: int) -> int:
        return max(1, round(milliseconds / 1000.0 * self.frame_rate))

    def text(self, key: str) -> str:
        return config.TEXT[self.setup["language"]][key]

    def draw_text(self, text: str, footer: str = "") -> None:
        self.message.text = text
        self.message.draw()
        if footer:
            self.small_message.text = footer
            self.small_message.draw()

    def draw_page(self, heading: str, body: str, footer: str = "") -> None:
        self.heading.text = heading
        self.heading.draw()
        self.message.text = body
        self.message.pos = (0, 0.05)
        self.message.draw()
        self.message.pos = (0, 0)
        if footer:
            self.small_message.text = footer
            self.small_message.draw()

    def show_until_key(self, text: str, allowed=("space", "return"), footer: str | None = None) -> str:
        self.event.clearEvents()
        footer = self.text("continue") if footer is None else footer
        while True:
            self.draw_text(text, footer)
            self.win.flip()
            keys = self.event.getKeys(keyList=list(allowed) + ["escape"])
            if "escape" in keys:
                self.confirm_abort()
                self.event.clearEvents()
            for key in allowed:
                if key in keys:
                    return key

    def show_page_until_key(self, heading: str, body: str) -> None:
        self.event.clearEvents()
        while True:
            self.draw_page(heading, body, self.text("continue"))
            self.win.flip()
            keys = self.event.getKeys(keyList=["space", "return", "escape"])
            if "escape" in keys:
                self.confirm_abort()
            if "space" in keys or "return" in keys:
                return

    def confirm_abort(self) -> None:
        prompt = (
            "Abort experiment?\n\nPress Y to abort and save, or N to continue."
            if self.setup["language"] == "en"
            else "是否中止实验？\n\n按 Y 中止并保存，按 N 继续。"
        )
        self.event.clearEvents()
        while True:
            self.draw_text(prompt, "")
            self.win.flip()
            keys = self.event.getKeys(keyList=["y", "n"])
            if "y" in keys:
                raise AbortExperiment()
            if "n" in keys:
                return

    def poll_escape_and_window(self, context: dict) -> None:
        if "escape" in self.event.getKeys(keyList=["escape"]):
            self.confirm_abort()
        try:
            focused = bool(self.win.winHandle.hasFocus)
        except Exception:
            focused = None
        if focused is not None and self.last_focus is not None and focused != self.last_focus:
            self.local_event("window_focus" if focused else "window_blur", context, "Window focus changed")
        self.last_focus = focused
        fullscreen = bool(self.win.fullscr)
        if self.last_fullscreen and not fullscreen:
            self.local_event("fullscreen_exit", context, "Window left fullscreen mode")
        self.last_fullscreen = fullscreen

    def local_event(self, name: str, context: dict, notes: str) -> None:
        self.data.add_event({
            "participant_id": self.setup["participant_id"],
            "participant_number": context.get("participant_number", ""),
            "session_id": self.setup["session_id"],
            "protocol_version": config.PROTOCOL_VERSION,
            "latin_square_row": context.get("latin_square_row", ""),
            "source_list_id": context.get("source_list_id", ""),
            "presented_run_number": context.get("presented_run_number", ""),
            "run_id": context.get("run_id", ""),
            "trial_index_global": context.get("trial_index_global", ""),
            "trial_index_run": context.get("trial_index_run", ""),
            "condition": (
                f"{context.get('visibility', '')}_{context.get('audio_condition', '')}"
                if context.get("visibility") else ""
            ),
            "event_name": name,
            "event_code": "",
            "condition_code": context.get("condition_code", ""),
            "object_label": context.get("image_label", ""),
            "psychopy_timestamp": self.global_clock.getTime(),
            "lsl_timestamp": "",
            "browser_or_system_datetime_iso": iso_now(),
            "marker_sent": False,
            "notes": notes,
        })

    def _push_marker_callback(self, event_name: str, event_code: int, context: dict, capture: dict | None = None) -> None:
        psychopy_time = self.global_clock.getTime()
        try:
            row = self.marker.push(event_name, event_code, context, psychopy_time)
        except Exception as exc:
            row = {
                "participant_id": self.setup["participant_id"],
                "participant_number": context.get("participant_number", ""),
                "session_id": self.setup["session_id"],
                "protocol_version": config.PROTOCOL_VERSION,
                "latin_square_row": context.get("latin_square_row", ""),
                "source_list_id": context.get("source_list_id", ""),
                "presented_run_number": context.get("presented_run_number", ""),
                "run_id": context.get("run_id", ""),
                "trial_index_global": context.get("trial_index_global", ""),
                "trial_index_run": context.get("trial_index_run", ""),
                "condition": (
                    f"{context.get('visibility', '')}_{context.get('audio_condition', '')}"
                    if context.get("visibility") else ""
                ),
                "event_name": event_name,
                "event_code": event_code,
                "condition_code": context.get("condition_code", ""),
                "object_label": context.get("image_label", ""),
                "psychopy_timestamp": psychopy_time,
                "lsl_timestamp": "",
                "browser_or_system_datetime_iso": iso_now(),
                "marker_sent": False,
                "notes": f"LSL error: {type(exc).__name__}: {exc}",
            }
            self.marker_errors.append(row["notes"])
            self.logging.error(row["notes"])
        self.pending_events.append(row)
        if capture is not None:
            capture[event_name] = row

    def schedule_marker(self, event_name: str, context: dict, capture: dict | None = None) -> None:
        self.win.callOnFlip(
            self._push_marker_callback,
            event_name,
            config.TRIGGER_CODES[event_name],
            context,
            capture,
        )

    def send_marker(self, event_name: str, context: dict) -> None:
        self._push_marker_callback(event_name, config.TRIGGER_CODES[event_name], context)
        self.flush_events()

    def flush_events(self) -> None:
        while self.pending_events:
            self.data.add_event(self.pending_events.pop(0))
        if self.marker_errors:
            self.data.save_event_log()
            message = self.marker_errors.pop(0)
            raise RuntimeError(message)

    def draw_response_cue(self) -> None:
        self.mic_body.draw()
        self.mic_stem.draw()
        self.mic_base.draw()
        self.message.text = self.text("speak_now")
        self.message.pos = (0, -0.38)
        self.message.draw()
        self.message.pos = (0, 0)

    def frame_phase(self, frame_count: int, draw, context: dict, first_marker: str | None = None) -> list[float]:
        """Draw to a flip deadline so one delayed frame is not followed by stale extra frames."""
        flips: list[float] = []
        target_seconds = frame_count * self.frame_duration
        while not flips or flips[-1] < flips[0] + target_seconds - 1.5 * self.frame_duration:
            draw()
            if not flips and first_marker:
                self.schedule_marker(first_marker, context)
            flips.append(self.win.flip())
            self.flush_events()
            self.poll_escape_and_window(context)
        return flips

    def audio_filename(self, trial: dict) -> str:
        trial_token = (
            f"practice-{int(trial['trial_index_run']):03d}"
            if trial["is_practice"]
            else f"{int(trial['trial_index_global']):04d}"
        )
        condition = f"{trial['visibility']}_{trial['audio_condition']}"
        return (
            f"sub-{self.data.participant_id}_ses-{self.data.session_id}_run-{trial['run_id']}_"
            f"trial-{trial_token}_object-{trial['image_object']}_exemplar-{trial['exemplar_id']}_"
            f"condition-{condition}.wav"
        )

    def run_trial(self, trial: dict) -> dict:
        self.current_trial = trial
        trial_start_iso = iso_now()
        self.stimuli.stop_all()
        self.event.clearEvents()
        self.stimuli.set_image(trial["image_path"])
        audio = self.stimuli.get_sound(trial["audio_path"]) if trial["audio_path"] else None
        audio_duration = self.stimuli.sound_duration_seconds(trial["audio_path"]) if audio else 0.0
        self.send_marker("trial_start", trial)

        fixation_frames = self.frames(config.FIXATION_MS)
        fixation_flips = self.frame_phase(fixation_frames, self.fixation.draw, trial, "fixation_onset")

        planned_audio_onset = ""
        capture: dict = {}
        self.stimuli.image.draw()
        self.schedule_marker(f"{trial['visibility']}_{trial['audio_condition']}_image_onset", trial, capture)
        if audio is not None:
            planned_audio_onset = self.win.getFutureFlipTime(clock="ptb")
            audio.play(when=planned_audio_onset)
            self.schedule_marker(f"{trial['audio_condition']}_audio_onset", trial, capture)
        image_onset = self.win.flip()
        self.flush_events()
        self.poll_escape_and_window(trial)
        image_flips = [image_onset]
        image_deadline = image_onset + config.IMAGE_DURATION_MS / 1000.0
        while image_flips[-1] < image_deadline - 1.5 * self.frame_duration:
            self.stimuli.image.draw()
            image_flips.append(self.win.flip())
            self.poll_escape_and_window(trial)

        self.fixation.draw()
        self.schedule_marker("image_offset", trial, capture)
        self.schedule_marker("silent_delay_onset", trial, capture)
        image_offset = self.win.flip()
        self.flush_events()
        self.poll_escape_and_window(trial)
        silent_flips = [image_offset]

        actual_image_duration_ms = (image_offset - image_onset) * 1000

        image_offset_time = float(capture["image_offset"]["psychopy_timestamp"])
        condition_audio_onset_time = ""
        condition_audio_offset_time = ""
        if audio is not None:
            audio_event = capture[f"{trial['audio_condition']}_audio_onset"]
            condition_audio_onset_time = float(audio_event["psychopy_timestamp"])
            condition_audio_offset_time = condition_audio_onset_time + audio_duration
        desired_segment_start = image_offset_time + (
            config.SILENT_DELAY_MS - config.DESIRED_PRE_ROLL_MS
        ) / 1000.0
        safe_segment_start = (
            float(condition_audio_offset_time) + config.RECORDING_GUARD_MS / 1000.0
            if condition_audio_offset_time != "" else desired_segment_start
        )
        segment_target_time = max(desired_segment_start, safe_segment_start)

        response_path = self.data.audio_dir / self.audio_filename(trial)
        self.current_audio_path = response_path
        recording_start_info: dict = {}
        recording_segment_start_time = ""
        microphone_start_block_ms = 0.0
        silent_deadline = image_offset + config.SILENT_DELAY_MS / 1000.0
        while silent_flips[-1] < silent_deadline - 1.5 * self.frame_duration:
            self.fixation.draw()
            silent_flips.append(self.win.flip())
            self.poll_escape_and_window(trial)
            if recording_segment_start_time == "" and self.global_clock.getTime() >= segment_target_time:
                microphone_start_call = self.core.Clock()
                recording_start_info = self.microphone.start()
                microphone_start_block_ms = microphone_start_call.getTime() * 1000.0
                recording_segment_start_time = self.global_clock.getTime()
                self.current_recording_start_time = recording_segment_start_time
                self.current_recording_started_iso = recording_start_info["recording_started_iso"]
                self.send_marker("microphone_recording_start", trial)

        if recording_segment_start_time == "":
            raise RuntimeError(
                "A clean response recording could not begin before the response cue. "
                "Check condition-audio duration and microphone status."
            )

        self.draw_response_cue()
        self.schedule_marker("response_cue_onset", trial, capture)
        response_onset = self.win.flip()
        self.flush_events()
        self.poll_escape_and_window(trial)
        response_cue_time = float(capture["response_cue_onset"]["psychopy_timestamp"])
        clean_pre_roll_ms = (response_cue_time - float(recording_segment_start_time)) * 1000
        if clean_pre_roll_ms < config.MIN_CLEAN_PRE_ROLL_MS:
            self.local_event("recording_pre_roll_failure", trial, f"clean_pre_roll_ms={clean_pre_roll_ms:.3f}")
            raise RuntimeError(
                f"Clean microphone pre-roll was only {clean_pre_roll_ms:.1f} ms; "
                f"minimum is {config.MIN_CLEAN_PRE_ROLL_MS} ms. Formal timing cannot continue."
            )

        response_flips = [response_onset]
        response_deadline = response_onset + config.RESPONSE_WINDOW_MS / 1000.0
        while response_flips[-1] < response_deadline - 1.5 * self.frame_duration:
            self.draw_response_cue()
            response_flips.append(self.win.flip())
            self.poll_escape_and_window(trial)

        iti_clock = self.core.Clock()
        self.win.callOnFlip(iti_clock.reset)
        if trial["is_practice"]:
            self.draw_text(self.text("recorded"), "")
        response_offset = self.win.flip()
        self.poll_escape_and_window(trial)
        response_window_end_time = self.global_clock.getTime()
        stop_after_ms = min(config.RECORDING_TAIL_MS, int(trial["iti_ms"]) - 100)
        if stop_after_ms <= 0:
            raise RuntimeError("ITI is too short to stop recording 100 ms before the next fixation")

        iti_flips = [response_offset]
        while iti_clock.getTime() * 1000 < stop_after_ms:
            if trial["is_practice"] and iti_clock.getTime() * 1000 < config.FEEDBACK_MS:
                self.draw_text(self.text("recorded"), "")
            iti_flips.append(self.win.flip())
            self.poll_escape_and_window(trial)

        segment = self.microphone.stop_capture()
        recording_segment_stop_time = self.global_clock.getTime()
        self.send_marker("microphone_recording_stop", trial)
        recording_duration_ms = (recording_segment_stop_time - float(recording_segment_start_time)) * 1000
        tail_ms = (recording_segment_stop_time - response_window_end_time) * 1000
        audio_result = self.microphone.save_segment(segment, response_path, recording_duration_ms)
        self.current_audio_path = None
        self.current_recording_started_iso = ""
        self.current_recording_start_time = ""

        while iti_clock.getTime() * 1000 < int(trial["iti_ms"]):
            iti_flips.append(self.win.flip())
            self.poll_escape_and_window(trial)
        iti_actual_ms = iti_clock.getTime() * 1000
        if iti_actual_ms > int(trial["iti_ms"]) + max(25.0, self.frame_duration * 2000):
            self.local_event("iti_overrun_failure", trial, f"iti_actual_ms={iti_actual_ms:.3f}")
            raise RuntimeError("WAV saving overran the ITI and would delay the next fixation")

        actual_by_phase = {
            "fixation": (image_onset - fixation_flips[0]) * 1000,
            "image": actual_image_duration_ms,
            "silent_delay": (response_onset - image_offset) * 1000,
            "response": (response_offset - response_onset) * 1000,
        }
        timing = evaluate_trial_timing(actual_by_phase, config.DISPLAY_TIMING_TOLERANCE_MS)
        for violation in timing["violations"]:
            self.local_event(
                "display_timing_violation",
                trial,
                (
                    f"phase={violation['phase']}; planned_ms={violation['planned_ms']:.3f}; "
                    f"actual_ms={violation['actual_ms']:.3f}; error_ms={violation['error_ms']:.3f}; "
                    "trial_flagged_for_exclusion=1"
                ),
            )

        relative_wav = response_path.relative_to(config.PROJECT_ROOT).as_posix()
        condition = f"{trial['visibility']}_{trial['audio_condition']}"
        audio_row = {
            **{field: trial.get(field, "") for field in (
                "participant_id", "participant_number", "session_id", "protocol_version",
                "latin_square_row", "source_list_id", "presented_run_number", "run_id",
                "trial_index_global", "trial_index_run",
            )},
            "condition": condition,
            "microphone_device_name": self.microphone.device_name,
            "microphone_device_index": self.microphone.device_index,
            "microphone_sample_rate": self.microphone.sample_rate,
            "microphone_stream_initialised_time": self.microphone.stream_initialised_time,
            "condition_audio_onset_time": condition_audio_onset_time,
            "condition_audio_offset_time": condition_audio_offset_time,
            "image_offset_time": image_offset_time,
            "recording_segment_start_time": recording_segment_start_time,
            "response_cue_time": response_cue_time,
            "response_window_end_time": response_window_end_time,
            "recording_segment_stop_time": recording_segment_stop_time,
            "clean_pre_roll_ms": round(clean_pre_roll_ms, 3),
            "tail_ms": round(tail_ms, 3),
            "recording_duration_ms": round(recording_duration_ms, 3),
            "recording_started_iso": recording_start_info.get("recording_started_iso", ""),
            "recording_stopped_iso": audio_result.get("recording_stopped_iso", ""),
            "recording_status": audio_result.get("recording_status", "save_failed"),
            "final_wav_path": relative_wav,
            **{key: audio_result.get(key, "") for key in (
                "wav_duration_ms", "sample_count", "peak_amplitude", "wav_sample_rate", "wav_channels",
            )},
        }
        self.data.add_audio(audio_row)
        if audio_row["recording_status"] != "saved":
            self.local_event("recording_validation_failure", trial, json.dumps(audio_row, default=str))
            self.show_until_key(
                f"{self.text('recording_warning')}\n\nStatus: {audio_row['recording_status']}",
                allowed=("return",), footer="Press ENTER to stop and save the session",
            )
            raise RuntimeError(f"Recording validation failed: {audio_row['recording_status']}")

        self.send_marker("trial_end", trial)
        self.marker.validate_trial(trial)
        row = dict(trial)
        row.update({
            "trial_start_iso": trial_start_iso,
            "fixation_ms_actual": round((image_onset - fixation_flips[0]) * 1000, 3),
            "fixation_frame_count": len(fixation_flips),
            "fixation_first_flip": fixation_flips[0],
            "fixation_last_flip": fixation_flips[-1],
            "image_duration_ms_actual": round((image_offset - image_onset) * 1000, 3),
            "image_frame_count": len(image_flips),
            "image_first_flip": image_onset,
            "image_last_flip": image_flips[-1],
            "planned_audio_onset_ptb": planned_audio_onset,
            "condition_audio_onset_time": condition_audio_onset_time,
            "condition_audio_offset_time": condition_audio_offset_time,
            "audio_marker_timestamp": capture.get(f"{trial['audio_condition']}_audio_onset", {}).get("lsl_timestamp", "") if audio else "",
            "image_offset_time": image_offset_time,
            "silent_delay_ms_actual": round((response_onset - image_offset) * 1000, 3),
            "silent_delay_frame_count": len(silent_flips),
            "response_window_ms_actual": round((response_offset - response_onset) * 1000, 3),
            "response_window_end_time": response_window_end_time,
            "response_frame_count": len(response_flips),
            "response_cue_flip": response_onset,
            "recording_segment_start_time": recording_segment_start_time,
            "response_cue_time": response_cue_time,
            "recording_segment_stop_time": recording_segment_stop_time,
            "clean_pre_roll_ms": round(clean_pre_roll_ms, 3),
            "recording_duration_ms": round(recording_duration_ms, 3),
            "tail_ms": round(tail_ms, 3),
            "wav_duration_ms": audio_result.get("wav_duration_ms", ""),
            "sample_count": audio_result.get("sample_count", ""),
            "peak_amplitude": audio_result.get("peak_amplitude", ""),
            "response_wav_path": relative_wav,
            "timeout": "",
            "iti_ms_actual": round(iti_actual_ms, 3),
            "iti_frame_count": len(iti_flips),
            "recording_status": audio_row["recording_status"],
            "microphone_start_block_ms": round(microphone_start_block_ms, 3),
            "session": trial["session_id"],
            "run": trial["run_id"],
            "trial_in_run": trial["trial_index_run"],
            "global_trial": trial["trial_index_global"],
            "object": trial["image_object"],
            "image_filename": trial["image_path"],
            "audio_filename": trial["audio_path"],
            "spoken_condition_word": trial["audio_label"],
            "response_cue_timestamp": response_cue_time,
            "microphone_segment_start": recording_segment_start_time,
            "microphone_segment_stop": recording_segment_stop_time,
            "response_wav_filename": response_path.name,
            "audio_status": audio_row["recording_status"],
            "planned_fixation_ms": config.FIXATION_MS,
            "actual_fixation_ms": round(actual_by_phase["fixation"], 3),
            "planned_image_ms": config.IMAGE_DURATION_MS,
            "actual_image_ms": round(actual_by_phase["image"], 3),
            "planned_silent_delay_ms": config.SILENT_DELAY_MS,
            "actual_silent_delay_ms": round(actual_by_phase["silent_delay"], 3),
            "planned_response_ms": config.RESPONSE_WINDOW_MS,
            "actual_response_ms": round(actual_by_phase["response"], 3),
            "timing_violation": timing["timing_violation"],
            "timing_violation_phase": timing["timing_violation_phase"],
            "timing_error_ms": timing["timing_error_ms"],
            "timing_violation_reason": timing["timing_violation_reason"],
        })
        self.data.add_behaviour(row)
        self.data.save_event_log()
        self.current_trial = None
        return row

    def initialise_devices_and_assets(self) -> dict:
        import psychopy
        from psychopy import prefs
        from lsl_markers import MarkerOutlet
        from microphone_recorder import TrialMicrophone
        from stimulus_manager import StimulusCache, assert_preflight_passed, run_asset_preflight

        self.marker = MarkerOutlet(self.data.participant_id, self.data.session_id)
        self.pending_events.append(self.marker.self_test(self.global_clock.getTime()))
        self.flush_events()
        self.data.save_event_log()
        report = run_asset_preflight(
            self.win,
            psychopy_load=True,
            require_approved_occlusions=self.setup["mode"] in ("full", "run_start_validation"),
        )
        self.data.save_asset_report(report)
        assert_preflight_passed(report)
        self.stimuli = StimulusCache(self.win)
        self.stimuli.warm_up_audio(self.core)
        self.cached_sounds = self.stimuli.sounds
        self.microphone = TrialMicrophone(
            self.data.audio_dir,
            allow_mock=self.setup["mode"] == "pilot",
        )
        self.microphone.initialise()
        self.microphone.warm_up()
        return {
            "psychopy_version": psychopy.__version__,
            "python_version": sys.version,
            "operating_system": platform.platform(),
            "monitor_name": config.MONITOR_NAME,
            "requested_fullscreen": self.setup["fullscreen"],
            "initial_setup_window_fullscreen": bool(self.win.fullscr),
            "actual_frame_rate": self.frame_rate,
            "refresh_calibration": self.refresh_calibration,
            "estimated_frame_duration_ms": self.frame_duration * 1000,
            "dropped_frame_count_at_start": getattr(self.win, "nDroppedFrames", 0),
            "audio_backend_requested": config.AUDIO_BACKEND,
            "audio_backend_actual": self.stimuli.audio_backend,
            "speaker_device": self.stimuli.speaker,
            "prepared_condition_audio_count": len(self.cached_sounds),
            "audio_library_preferences": prefs.hardware.get("audioLib"),
            "microphone_device": self.microphone.device_name,
            "microphone_device_index": self.microphone.device_index,
            "microphone_sample_rate": self.microphone.sample_rate,
            "microphone_stream_initialised_time": self.microphone.stream_initialised_time,
            "microphone_mock": self.microphone.mock,
            "microphone_initialise_count": self.microphone.initialise_count,
            "microphone_stream_warmed": self.microphone.warmed_up,
            "lsl_outlet": self.marker.info,
            "occlusion_style": config.OCCLUSION_STYLE,
            "occlusion_level": config.OCCLUSION_LEVEL,
            "random_seed": self.seed,
            "canonical_master_seed": config.CANONICAL_MASTER_SEED,
            "protocol_version": config.PROTOCOL_VERSION,
            "latin_square_row": self.main_trials[0].get("latin_square_row", "") if self.main_trials else "",
            "source_list_order": [
                trial["source_list_id"] for trial in self.main_trials
                if int(trial["trial_index_run"]) == 1
            ],
            "mode": self.setup["mode"],
            "language": self.setup["language"],
            "formal_objects": list(config.OBJECTS),
            "object_count": len(config.OBJECTS),
            "timing_ms": {
                "fixation": config.FIXATION_MS,
                "audio_lead": config.AUDIO_LEAD_MS,
                "image": config.IMAGE_DURATION_MS,
                "silent_delay": config.SILENT_DELAY_MS,
                "response_window": config.RESPONSE_WINDOW_MS,
                "recording_tail": config.RECORDING_TAIL_MS,
                "iti_min": config.ITI_MIN_MS,
                "iti_max": config.ITI_MAX_MS,
            },
        }

    def labrecorder_setup_screen(self, next_run: str | None = None) -> None:
        """Keep the one marker outlet alive while the operator controls LabRecorder."""
        context = {"run_id": next_run or "", "presented_run_number": next_run or ""}
        event_name = "labrecorder_setup_start" if next_run is None else "run_transition_start"
        self.local_event(event_name, context, f"outlet_source_id={self.marker.info['source_id']}")
        self.data.save_event_log()
        if next_run is None:
            heading = "LabRecorder Setup"
            body = (
                "1. Open LabRecorder.\n"
                "2. Select SAGA EEG.\n"
                "3. Select PsychoPyMarkers.\n"
                "4. Press Start in LabRecorder.\n"
                "5. Return here and press SPACE.\n\n"
                "ESC = Abort and save"
            )
        else:
            heading = f"Run {int(next_run) - 1:02d} complete"
            body = (
                "1. Stop and save the current recording in LabRecorder.\n"
                "2. Start a new LabRecorder recording for the next run.\n"
                "3. Confirm that SAGA EEG and PsychoPyMarkers are selected.\n"
                "4. Return here and press SPACE.\n\n"
                f"SPACE = Start Run {int(next_run):02d}\n"
                "ESC = Safely end experiment"
            )
        self.event.clearEvents()
        while True:
            self.draw_page(heading, body, "")
            self.win.flip()
            keys = self.event.getKeys(keyList=["space", "escape"])
            if "escape" in keys:
                self.data.save_behaviour(autosave=True)
                self.data.save_audio_manifest()
                self.data.save_event_log()
                self.confirm_abort()
            if "space" in keys:
                complete_name = "labrecorder_setup_complete" if next_run is None else "run_transition_complete"
                self.local_event(complete_name, context, f"outlet_source_id={self.marker.info['source_id']}")
                return

    def run_break_screen(self, next_run: str) -> None:
        """Participant-controlled break while the one LabRecorder recording continues."""
        heading = f"Break - Run {int(next_run) - 1:02d} complete"
        body = (
            "LabRecorder must continue recording. Do not stop or restart it.\n\n"
            "Take as long as needed. Press SPACE when ready for the next run.\n\n"
            "ESC = Safely end experiment"
        )
        self.event.clearEvents()
        while True:
            self.draw_page(heading, body, "")
            self.win.flip()
            keys = self.event.getKeys(keyList=["space", "escape"])
            if "escape" in keys:
                self.confirm_abort()
            if "space" in keys:
                return

    def labrecorder_final_screen(self) -> None:
        """Let the operator stop and save the final XDF after experiment_end."""
        self.local_event(
            "final_labrecorder_setup_start", {},
            f"outlet_source_id={self.marker.info['source_id']}",
        )
        self.data.save_event_log()
        body = (
            "All six runs are complete.\n\n"
            "1. Stop and save the final recording in LabRecorder.\n"
            "2. Confirm that the XDF file was created.\n"
            "3. Return here and press SPACE to close."
        )
        self.event.clearEvents()
        while True:
            self.draw_page("Experiment complete", body, "SPACE = Close")
            self.win.flip()
            keys = self.event.getKeys(keyList=["space", "escape"])
            if "space" in keys:
                self.local_event(
                    "final_labrecorder_setup_complete", {},
                    f"outlet_source_id={self.marker.info['source_id']}",
                )
                return
            if "escape" in keys:
                self.data.save_behaviour(autosave=True)
                self.data.save_audio_manifest()
                self.data.save_event_log()
                self.confirm_abort()

    def researcher_ready_screen(self) -> None:
        while True:
            body = self.text("asset_ready")
            self.draw_text(body, self.text("start_hint"))
            self.win.flip()
            keys = self.event.getKeys(keyList=["return", "a", "escape"])
            if "escape" in keys:
                self.confirm_abort()
            if "return" in keys:
                return
            if "a" in keys:
                from stimulus_manager import assert_preflight_passed, run_asset_preflight
                report = run_asset_preflight(
                    self.win,
                    psychopy_load=True,
                    require_approved_occlusions=self.setup["mode"] == "full",
                )
                self.data.save_asset_report(report)
                assert_preflight_passed(report)

    def run_instruction_pages(self) -> None:
        for heading, body in config.INSTRUCTION_PAGES[self.setup["language"]]:
            self.show_page_until_key(heading, body)

    def run_microphone_preflight(self) -> dict:
        from psychopy import sound

        path = self.data.audio_dir / "microphone_preflight_test.wav"
        last_result: dict = {}
        while True:
            self.event.clearEvents()
            started: dict = {}

            def start_on_flip() -> None:
                started.update(self.microphone.start())
                started["recording_segment_start_time"] = self.global_clock.getTime()

            self.draw_response_cue()
            self.small_message.text = self.text("mic_test_prompt")
            self.small_message.draw()
            self.win.callOnFlip(start_on_flip)
            first_flip = self.win.flip()
            while (
                self.global_clock.getTime() - started["recording_segment_start_time"]
                < config.MICROPHONE_PREFLIGHT_MS / 1000.0
            ):
                self.draw_response_cue()
                self.small_message.text = self.text("mic_test_prompt")
                self.small_message.draw()
                self.win.flip()
                self.poll_escape_and_window({"run_id": "preflight"})
            segment = self.microphone.stop_capture()
            stopped_time = self.global_clock.getTime()
            duration_ms = (stopped_time - started["recording_segment_start_time"]) * 1000
            last_result = self.microphone.save_segment(
                segment, path, config.MICROPHONE_PREFLIGHT_MS
            )
            passed = last_result.get("recording_status") == "saved"
            payload = {
                "protocol_version": config.PROTOCOL_VERSION,
                "participant_id": self.data.participant_id,
                "device_name": self.microphone.device_name,
                "device_index": self.microphone.device_index,
                "sample_rate": self.microphone.sample_rate,
                "requested_duration_ms": config.MICROPHONE_PREFLIGHT_MS,
                "recording_duration_ms": round(duration_ms, 3),
                "first_flip": first_flip,
                "passed": passed,
                **{key: value for key, value in last_result.items() if key != "clip"},
            }
            (self.data.session_dir / "microphone_preflight_result.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            status = "PASS" if passed else "FAIL"
            body = (
                f"{status}\n\nDevice: {self.microphone.device_name}\n"
                f"Sample rate: {self.microphone.sample_rate} Hz\n"
                f"Duration: {last_result.get('wav_duration_ms', 0):.1f} ms\n"
                f"Peak amplitude: {last_result.get('peak_amplitude', 0):.5f}\n\n"
                "Use P to listen. Continue only if the immediately spoken word is complete."
            )
            while True:
                self.draw_page(self.text("mic_test_title"), body, self.text("mic_test_controls"))
                self.win.flip()
                keys = self.event.getKeys(keyList=["r", "p", "return", "escape"])
                if "escape" in keys:
                    self.confirm_abort()
                if "p" in keys and path.exists():
                    playback = sound.Sound(str(path), stereo=self.microphone.channels == 2)
                    playback.play()
                if "r" in keys:
                    break
                if "return" in keys and passed:
                    return payload
            # R repeats the complete recording/validation cycle.

    def _image_stim(self, relative_path: str, pos: tuple[float, float], height: float):
        absolute = config.PROJECT_ROOT / relative_path
        with Image.open(absolute) as image:
            aspect = image.width / image.height
        return self.visual.ImageStim(
            self.win, image=str(absolute), units="norm", pos=pos,
            size=(height * aspect, height), interpolate=True,
        )

    def run_familiarisation(self) -> None:
        self.show_until_key(self.text("stage_familiarisation"))
        positions = ((-0.42, 0.30), (0.42, 0.30), (-0.42, -0.25), (0.42, -0.25))
        label_stim = self.visual.TextStim(self.win, color="white", height=0.085, pos=(0, -0.58))
        for item in self.objects:
            images = [
                self._image_stim(exemplar["visible_path"], position, 0.39)
                for exemplar, position in zip(item["exemplars"], positions)
            ]
            spoken = self.stimuli.get_sound(item["audio_path"])
            label_stim.text = item["label"]

            def draw_object_page() -> None:
                for stimulus in images:
                    stimulus.draw()
                label_stim.draw()
                self.small_message.text = self.text("familiarisation_hint")
                self.small_message.draw()

            draw_object_page()
            self.win.flip()
            spoken.play()
            self.event.clearEvents()
            while True:
                draw_object_page()
                self.win.flip()
                keys = self.event.getKeys(keyList=["r", "space", "return", "escape"])
                if "escape" in keys:
                    self.confirm_abort()
                if "r" in keys:
                    spoken.stop()
                    spoken.play()
                if "space" in keys or "return" in keys:
                    spoken.stop()
                    break

        x_positions = (-0.6, 0.0, 0.6)
        y_positions = (0.28, -0.40)
        summary_items = []
        for index, item in enumerate(self.objects):
            pos = (x_positions[index % 3], y_positions[index // 3])
            image = self._image_stim(item["exemplars"][0]["visible_path"], (pos[0], pos[1] + 0.09), 0.27)
            label = self.visual.TextStim(self.win, text=item["label"], color="white", height=0.045,
                                         pos=(pos[0], pos[1] - 0.14))
            summary_items.append((image, label))
        self.event.clearEvents()
        while True:
            self.heading.text = self.text("familiarisation_summary")
            self.heading.draw()
            for image, label in summary_items:
                image.draw()
                label.draw()
            self.small_message.text = self.text("continue")
            self.small_message.draw()
            self.win.flip()
            keys = self.event.getKeys(keyList=["space", "return", "escape"])
            if "escape" in keys:
                self.confirm_abort()
            if "space" in keys or "return" in keys:
                return

    def save_run_timing_summary(self, run_id: str) -> dict:
        rows = [
            row for row in self.data.behaviour_rows
            if str(row.get("run_id", "")) == str(run_id) and int(row.get("is_practice", 0) or 0) == 0
        ]
        qc = build_timing_qc(rows)
        summary = {
            "total_trials": qc["total_formal_trials"],
            "valid_timing_trials": qc["timing_valid_trials"],
            "timing_violation_trials": qc["timing_violation_trials"],
            "percentage_timing_valid": qc["timing_valid_percentage"],
            "affected_trials": qc["affected_trials"],
            "dropped_or_late_frames": self.run_dropped_frames.get(str(run_id), 0),
            "session_refresh_hz": self.frame_rate,
            "session_window_id": self.session_window_id,
        }
        summary["passive_observed_refresh_hz"] = validate_observed_session_refresh(
            rows, self.frame_rate
        )
        self.data.save_run_timing_summary(f"run_{int(run_id):02d}", summary)
        return summary

    def timing_safety_reached(self, run_rows: list[dict], consecutive: int) -> bool:
        violations = sum(int(row.get("timing_violation", 0) or 0) for row in run_rows)
        percentage = violations / len(run_rows) * 100 if run_rows else 0.0
        return (
            consecutive >= config.TIMING_WARNING_CONSECUTIVE_TRIALS
            or (
                violations >= config.TIMING_WARNING_MIN_VIOLATIONS
                and percentage > config.TIMING_WARNING_RUN_PERCENT
            )
        )

    def show_timing_safety_warning(self, run_id: str, run_rows: list[dict]) -> None:
        self.data.save_behaviour(autosave=True)
        self.data.save_audio_manifest()
        self.data.save_event_log()
        summary = self.save_run_timing_summary(run_id)
        context = run_rows[-1]
        self.local_event(
            "timing_system_warning", context,
            f"violations={summary['timing_violation_trials']}; completed={summary['total_trials']}",
        )
        body = (
            "Repeated display-timing violations were detected.\n\n"
            f"Run {int(run_id):02d}: {summary['timing_violation_trials']} violations in "
            f"{summary['total_trials']} completed trials.\n\n"
            "The current trial and all data have been saved.\n"
            "ENTER = continue\nESC = safely abort"
        )
        self.event.clearEvents()
        while True:
            self.draw_page("Experimenter timing warning", body, "")
            self.win.flip()
            keys = self.event.getKeys(keyList=["return", "escape"])
            if "return" in keys:
                return
            if "escape" in keys:
                self.confirm_abort()

    def run_trials(self, trials: list[dict], practice: bool) -> None:
        if practice:
            self.send_marker("practice_start", trials[0])
        grouped: dict[str, list[dict]] = {}
        for trial in trials:
            grouped.setdefault(trial["run_id"], []).append(trial)
        run_ids = list(grouped)
        for run_index, run_id in enumerate(run_ids):
            context = dict(grouped[run_id][0])
            run_rows: list[dict] = []
            consecutive_violations = 0
            if not practice:
                if self.prepared_run_id != str(run_id):
                    self.prepare_formal_run(grouped[run_id], run_id)
                gc.disable()
                self.core.rush(True)
                self.send_marker("run_start", context)
                dropped_before = int(getattr(getattr(self, "win", None), "nDroppedFrames", 0) or 0)
            for trial in grouped[run_id]:
                row = self.run_trial(trial)
                run_rows.append(row)
                if int(row.get("timing_violation", 0) or 0) == 1:
                    consecutive_violations += 1
                else:
                    consecutive_violations = 0
                if (
                    not practice
                    and run_id not in self.timing_warning_runs
                    and self.timing_safety_reached(run_rows, consecutive_violations)
                ):
                    self.timing_warning_runs.add(run_id)
                    self.show_timing_safety_warning(run_id, run_rows)
            if not practice:
                self.send_marker("run_end", context)
                dropped_after = int(getattr(getattr(self, "win", None), "nDroppedFrames", 0) or 0)
                if not hasattr(self, "run_dropped_frames"):
                    self.run_dropped_frames = {}
                self.run_dropped_frames[str(run_id)] = max(0, dropped_after - dropped_before)
                self.data.save_event_log()
                self.data.save_behaviour(autosave=True)
                self.data.save_audio_manifest()
                self.save_run_timing_summary(run_id)
                self.prepared_run_id = None
                if not gc.isenabled():
                    gc.enable()
                self.core.rush(False)
                gc.collect()
                if run_index < len(run_ids) - 1:
                    self.send_marker("break_start", context)
                    next_context = grouped[run_ids[run_index + 1]][0]
                    self.run_break_screen(run_ids[run_index + 1])
                    self.prepare_formal_run(
                        grouped[run_ids[run_index + 1]], run_ids[run_index + 1]
                    )
                    self.send_marker("break_end", next_context)
        if practice:
            self.send_marker("practice_end", trials[-1])

    def safe_abort_devices(self) -> None:
        if self.stimuli is not None:
            self.stimuli.stop_all()
        if self.microphone is not None:
            try:
                if self.microphone.recording and self.current_audio_path is not None:
                    trial = self.current_trial or {}
                    segment = self.microphone.stop_capture()
                    if self.marker is not None:
                        try:
                            self.send_marker("microphone_recording_stop", trial)
                        except Exception:
                            pass
                    elapsed_ms = max(
                        1.0,
                        (self.global_clock.getTime() - float(self.current_recording_start_time or self.global_clock.getTime())) * 1000,
                    )
                    result = self.microphone.save_segment(segment, self.current_audio_path, elapsed_ms)
                    self.data.add_audio({
                        **{field: trial.get(field, "") for field in config.AUDIO_MANIFEST_FIELDS},
                        "protocol_version": config.PROTOCOL_VERSION,
                        "final_wav_path": self.current_audio_path.relative_to(config.PROJECT_ROOT).as_posix(),
                        "recording_status": f"aborted_partial_{result.get('recording_status', 'save_failed')}",
                        "recording_started_iso": self.current_recording_started_iso,
                        "recording_stopped_iso": result.get("recording_stopped_iso", ""),
                    })
                else:
                    self.microphone.abort()
            except Exception:
                pass

    def run(self) -> None:
        all_manifest = self.practice + self.main_trials
        base_config = {
            "participant_id": self.setup["participant_id"],
            "session_id": self.setup["session_id"],
            "mode": self.setup["mode"],
            "language": self.setup["language"],
            "random_seed": self.seed,
            "protocol_version": config.PROTOCOL_VERSION,
            "canonical_master_seed": config.CANONICAL_MASTER_SEED,
            "latin_square_row_override": self.setup.get("latin_row_override"),
        }
        self.data.initialise(all_manifest, base_config, self.seed)
        if self.validation_mode:
            write_csv(
                self.data.session_dir / "full_432_trial_source_manifest.csv",
                self.validation_source_manifest,
            )
        status = "aborted"
        summary: dict = {}
        try:
            runtime_config = self.initialise_devices_and_assets()
            self.data.initialise(all_manifest, {**base_config, **runtime_config}, self.seed)
            self.labrecorder_setup_screen()
            self.open_fullscreen_for_run(all_manifest)
            runtime_config.update({
                "actual_fullscreen": bool(self.win.fullscr),
                "actual_frame_rate": self.frame_rate,
                "refresh_calibration": self.refresh_calibration,
                "estimated_frame_duration_ms": self.frame_duration * 1000,
                "lsl_outlet_persisted_across_window_recreation": True,
            })
            self.data.initialise(all_manifest, {**base_config, **runtime_config}, self.seed)
            self.send_marker("experiment_start", {})
            if self.validation_mode:
                heading = "Run-start hardware validation"
                body = (
                    "This is a 30-trial hardware test: trials 1-5 from each of six runs.\n\n"
                    "Results are stored separately and are not formal participant data."
                )
                self.show_page_until_key(heading, body)
                summary["microphone_preflight"] = self.run_microphone_preflight()
            else:
                self.run_instruction_pages()
                preflight = self.run_microphone_preflight()
                summary["microphone_preflight"] = preflight
                self.run_familiarisation()
                self.show_page_until_key(self.text("stage_practice"), self.text("practice"))
                self.run_trials(self.practice, practice=True)
                self.show_page_until_key(self.text("stage_formal"), self.text("practice_done"))
            self.run_trials(self.main_trials, practice=False)
            self.send_marker("experiment_end", {})
            status = "complete"
            self.data.save_behaviour(autosave=True)
            self.data.save_audio_manifest()
            self.labrecorder_final_screen()
        except OcclusionApprovalPending as exc:
            status = "blocked_pending_stimulus_approval"
            summary["error"] = str(exc)
            if self.setup["language"] == "zh":
                message = (
                    "70% 遮挡图片尚未审批\n\n"
                    "48 张图片都存在并且可以正常加载，这不是文件丢失或程序损坏。\n\n"
                    "如果只是本地测试流程，请重新运行并选择 Pilot。\n"
                    "如果准备运行 Full，请先检查 contact_sheets，并在 QA 表中完成审批。"
                )
                footer = "按空格键或回车键关闭"
            else:
                message = (
                    "70% occlusion stimuli are awaiting approval\n\n"
                    "All 48 files exist and load correctly. This is not a missing-file error.\n\n"
                    "For a local flow test, restart and select Pilot. For Full mode, review the "
                    "contact sheets and complete the QA approval first."
                )
                footer = "Press SPACE or ENTER to close"
            self.show_until_key(message, footer=footer)
        except AbortExperiment:
            self.safe_abort_devices()
            if self.marker is not None:
                try:
                    self.send_marker("experiment_abort", {})
                except Exception:
                    pass
            status = "aborted_by_researcher"
        except Exception as exc:
            self.safe_abort_devices()
            if self.marker is not None:
                try:
                    self.send_marker("experiment_abort", {})
                except Exception:
                    pass
            status = "error"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            summary["traceback"] = traceback.format_exc()
            try:
                self.show_until_key(
                    f"Researcher error\n\n{type(exc).__name__}: {exc}\n\nData collected so far has been saved.",
                    allowed=("space",), footer="Press SPACE to close",
                )
            except AbortExperiment:
                pass
        finally:
            summary.update({
                "random_seed": self.seed,
                "protocol_version": config.PROTOCOL_VERSION,
                "canonical_master_seed": config.CANONICAL_MASTER_SEED,
                "latin_square_row": self.main_trials[0].get("latin_square_row", "") if self.main_trials else "",
                "source_list_order": [
                    trial["source_list_id"] for trial in self.main_trials
                    if int(trial["trial_index_run"]) == 1
                ],
                "mode": self.setup["mode"],
                "planned_main_trials": len(self.main_trials),
                "planned_practice_trials": len(self.practice),
                "formal_objects": list(config.OBJECTS),
                "object_count": len(config.OBJECTS),
                "dropped_frame_count": getattr(self.win, "nDroppedFrames", ""),
                "actual_frame_rate": self.frame_rate,
                "timing_qc": build_timing_qc(self.data.behaviour_rows),
                "run_timing_summaries": self.data.run_timing_summaries,
                "lsl_outlet_source_id": self.marker.info["source_id"] if self.marker is not None else "",
                "run_warmup_records": self.run_warmup_records,
                "session_refresh_hz": self.frame_rate,
                "session_window_id": self.session_window_id,
                "run_dropped_or_late_frames": self.run_dropped_frames,
                "requested_session_id": self.requested_session_id,
                "validation_output_isolated": self.validation_mode,
            })
            if self.validation_mode:
                marker_counts = Counter(
                    str(row.get("event_name", "")) for row in self.data.event_rows
                    if row.get("marker_sent") in (True, 1, "True", "true", "1")
                )
                expected_marker_counts = Counter({
                    "lsl_self_test": 1,
                    "experiment_start": 1,
                    "experiment_end": 1,
                    "run_start": config.FULL_RUNS,
                    "run_end": config.FULL_RUNS,
                    "break_start": config.FULL_RUNS - 1,
                    "break_end": config.FULL_RUNS - 1,
                })
                for trial in self.main_trials:
                    expected_marker_counts.update((
                        "trial_start", "fixation_onset",
                        f"{trial['visibility']}_{trial['audio_condition']}_image_onset",
                        "image_offset", "silent_delay_onset", "microphone_recording_start",
                        "response_cue_onset", "microphone_recording_stop", "trial_end",
                    ))
                    if trial["audio_condition"] in ("congruent", "incongruent"):
                        expected_marker_counts.update((f"{trial['audio_condition']}_audio_onset",))
                phase_fields = {
                    "fixation": "actual_fixation_ms",
                    "image": "actual_image_ms",
                    "silent_delay": "actual_silent_delay_ms",
                    "response": "actual_response_ms",
                }
                actual_durations = {}
                for phase, field in phase_fields.items():
                    values = [float(row[field]) for row in self.data.behaviour_rows if row.get(field) != ""]
                    actual_durations[phase] = {
                        "mean_ms": round(statistics.mean(values), 3) if values else None,
                        "minimum_ms": round(min(values), 3) if values else None,
                        "maximum_ms": round(max(values), 3) if values else None,
                    }
                validation_report = {
                    "mode": "run_start_validation",
                    "formal_data": False,
                    "output_directory": str(self.data.session_dir),
                    "planned_trials": len(self.main_trials),
                    "trials_per_run": config.RUN_START_VALIDATION_TRIALS_PER_RUN,
                    "actual_phase_durations": actual_durations,
                    "timing_qc": build_timing_qc(self.data.behaviour_rows),
                    "timing_violations_by_run": self.data.run_timing_summaries,
                    "dropped_or_late_frames_by_run": self.run_dropped_frames,
                    "trigger_counts": dict(sorted(marker_counts.items())),
                    "expected_trigger_counts": dict(sorted(expected_marker_counts.items())),
                    "trigger_count_differences": {
                        name: marker_counts.get(name, 0) - expected
                        for name, expected in sorted(expected_marker_counts.items())
                    },
                    "run_warmup_records": self.run_warmup_records,
                    "session_refresh_hz": self.frame_rate,
                    "session_refresh_rate_used_by_run": {
                        f"{run_id:02d}": self.frame_rate
                        for run_id in range(1, config.FULL_RUNS + 1)
                    },
                    "trial_phase_timings": [
                        {
                            "run": row.get("run_id"),
                            "trial_in_run": row.get("trial_index_run"),
                            "fixation_ms": row.get("actual_fixation_ms"),
                            "image_ms": row.get("actual_image_ms"),
                            "silent_delay_ms": row.get("actual_silent_delay_ms"),
                            "response_ms": row.get("actual_response_ms"),
                            "iti_ms": row.get("iti_ms_actual"),
                            "timing_violation": row.get("timing_violation"),
                            "timing_violation_phase": row.get("timing_violation_phase"),
                        }
                        for row in self.data.behaviour_rows
                        if int(row.get("is_practice", 0) or 0) == 0
                    ],
                }
                (self.data.session_dir / "hardware_validation_report.json").write_text(
                    json.dumps(validation_report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            self.data.finalise(status, summary)
            if not gc.isenabled():
                gc.enable()
            self.core.rush(False)
            if self.microphone is not None:
                self.microphone.close()
            if self.stimuli is not None:
                self.stimuli.stop_all()
            if self.win is not None:
                self.win.close()
            self.core.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        pure_validation()
        return

    from psychopy import prefs
    prefs.hardware["audioLib"] = [config.AUDIO_BACKEND]
    prefs.hardware["audioLatencyMode"] = 3
    from psychopy import gui

    setup = get_setup(gui)
    if setup is None:
        return
    if not setup["participant_id"] or not setup["session_id"]:
        error = gui.Dlg(title="Setup error")
        error.addText("Participant ID and Session ID are required.")
        error.show()
        return
    ExperimentRunner(setup).run()


if __name__ == "__main__":
    main()
