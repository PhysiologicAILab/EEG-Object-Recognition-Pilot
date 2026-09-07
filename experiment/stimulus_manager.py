"""Stimulus manifest loading and strict startup asset checks."""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

from PIL import Image

import config


class OcclusionApprovalPending(RuntimeError):
    """Raised when valid 70% candidates have not yet been approved for Full mode."""


def _exact_case_exists(path: Path) -> bool:
    if not path.exists():
        return False
    return any(child.name == path.name for child in path.parent.iterdir())


def all_asset_paths() -> tuple[list[Path], list[Path], list[Path]]:
    with config.OBJECTS_FILE.open(encoding="utf-8") as handle:
        objects = json.load(handle)
    visible: list[Path] = []
    occluded: list[Path] = []
    audio: list[Path] = []
    for item in objects:
        audio.append(config.PROJECT_ROOT / item["audio_path"])
        for exemplar in item["exemplars"]:
            visible.append(config.PROJECT_ROOT / exemplar["visible_path"])
            occluded.extend(config.PROJECT_ROOT / path for path in exemplar["occluded_paths"])
    return visible, occluded, audio


def _occlusion_qa() -> dict[str, dict]:
    if not config.OCCLUSION_QA_FILE.exists():
        return {}
    with config.OCCLUSION_QA_FILE.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        relative = row.get("output_path")
        if not relative and row.get("occluded_filename"):
            relative = f"stimuli_final_candidates/{row['occluded_filename']}"
        if not relative and row.get("category") and row.get("exemplar_id"):
            relative = (
                f"stimuli_final_candidates/occluded_70/{row['category']}/"
                f"{row['exemplar_id']}_occ70.png"
            )
        if relative:
            result[relative] = row
    return result


def run_asset_preflight(
    win=None,
    psychopy_load: bool = False,
    require_approved_occlusions: bool = False,
) -> list[dict]:
    visible, occluded, audio = all_asset_paths()
    qa_by_path = _occlusion_qa()
    rows: list[dict] = []
    image_stim = None
    if psychopy_load:
        from psychopy import visual
        image_stim = visual.ImageStim(win, image=None, units="norm")

    for asset_type, paths in (("visible_image", visible), ("occluded_image", occluded)):
        for path in paths:
            exists = path.exists() and _exact_case_exists(path)
            format_ok = False
            load_ok = False
            details = ""
            try:
                if exists:
                    with Image.open(path) as image:
                        format_ok = image.format == "PNG" and image.width > 0 and image.height > 0
                    if psychopy_load:
                        image_stim.setImage(str(path), log=False)
                    load_ok = True
            except Exception as exc:
                details = f"{type(exc).__name__}: {exc}"
            qa_ok = True
            if asset_type == "occluded_image":
                relative = path.relative_to(config.PROJECT_ROOT).as_posix()
                qa = qa_by_path.get(relative)
                if qa is None:
                    qa_ok = False
                    details = f"{details}; missing QA row".strip("; ")
                else:
                    measured = float(
                        qa.get("measured_foreground_coverage_percent")
                        or qa["estimated_actual_foreground_coverage_percent"]
                    )
                    coverage_ok = 68.0 <= measured <= 72.0
                    # The final-candidate report is the frozen approved source for v3.
                    approved = (
                        qa.get("approved", "").strip().lower() == "true"
                        or "estimated_actual_foreground_coverage_percent" in qa
                    )
                    qa_ok = coverage_ok and (approved or not require_approved_occlusions)
                    details = (
                        f"{details}; coverage={measured:.2f}%; approved={approved}; "
                        f"status={qa.get('approval_status', '')}"
                    ).strip("; ")
            rows.append({
                "asset_type": asset_type,
                "path": path.relative_to(config.PROJECT_ROOT).as_posix(),
                "exists": exists,
                "format_ok": format_ok,
                "psychopy_load_ok": load_ok,
                "status": "pass" if exists and format_ok and load_ok and qa_ok else "fail",
                "details": details,
            })

    for path in audio:
        exists = path.exists() and _exact_case_exists(path)
        format_ok = False
        load_ok = False
        details = ""
        try:
            if exists:
                with wave.open(str(path), "rb") as handle:
                    channels = handle.getnchannels()
                    sample_rate = handle.getframerate()
                    duration = handle.getnframes() / sample_rate
                    format_ok = channels == 1 and sample_rate > 0 and duration > 0
                    details = f"{sample_rate} Hz; {channels} channel; {duration:.3f} s"
                if psychopy_load:
                    from psychopy import core, sound
                    stimulus = sound.Sound(str(path), stereo=False)
                    stimulus.setVolume(0.0)
                    stimulus.play()
                    core.wait(0.02)
                    stimulus.stop()
                load_ok = True
        except Exception as exc:
            details = f"{type(exc).__name__}: {exc}"
        rows.append({
            "asset_type": "spoken_audio",
            "path": path.relative_to(config.PROJECT_ROOT).as_posix(),
            "exists": exists,
            "format_ok": format_ok,
            "psychopy_load_ok": load_ok,
            "status": "pass" if exists and format_ok and load_ok else "fail",
            "details": details,
        })
    return rows


def assert_preflight_passed(rows: list[dict]) -> None:
    failures = [row for row in rows if row["status"] != "pass"]
    if failures:
        approval_failures = [
            row for row in failures
            if row["asset_type"] == "occluded_image" and "approved=False" in row["details"]
        ]
        if len(approval_failures) == len(failures):
            raise OcclusionApprovalPending(
                f"All {len(failures)} 70% occlusion files exist and load correctly, but are still "
                "pending manual approval. For a local flow test, restart and select Pilot mode. "
                "For Full mode, review assets/images/occluded/70/contact_sheets and then update "
                "assets/images/occluded/70/occlusion_qa_report.csv."
            )
        paths = "\n".join(row["path"] for row in failures[:12])
        raise RuntimeError(f"Asset preflight failed for {len(failures)} files:\n{paths}")


class StimulusCache:
    def __init__(self, win, image_paths=(), sounds: dict[str, object] | None = None) -> None:
        from psychopy import sound, visual

        self.win = win
        self.visual = visual
        self.images: dict[str, object] = {}
        self.image = None
        self.image_decode_count = 0
        self.sounds = sounds if sounds is not None else {}
        if sounds is None:
            _, _, audio_paths = all_asset_paths()
            for path in audio_paths:
                self.sounds[path.relative_to(config.PROJECT_ROOT).as_posix()] = sound.Sound(
                    str(path), stereo=False
                )
        self.preload_images(image_paths)

    def preload_images(self, relative_paths) -> None:
        """Decode each PNG and create one persistent ImageStim before formal timing."""
        for relative_path in dict.fromkeys(relative_paths):
            if relative_path in self.images:
                continue
            absolute = config.PROJECT_ROOT / relative_path
            with Image.open(absolute) as source:
                source.load()
                aspect = source.width / source.height
            height = config.IMAGE_SIZE_HEIGHT
            self.images[relative_path] = self.visual.ImageStim(
                self.win,
                image=str(absolute),
                units="norm",
                size=(height * aspect, height),
                interpolate=True,
            )
            self.image_decode_count += 1

    def prepare_textures(self, relative_paths) -> None:
        """Issue draws to prepare GPU textures without presenting stimuli onscreen."""
        self.preload_images(relative_paths)
        for relative_path in dict.fromkeys(relative_paths):
            self.images[relative_path].draw()
        self.win.clearBuffer()

    def set_image(self, relative_path: str) -> None:
        """Select an already prepared image; disk access during a trial is forbidden."""
        if relative_path not in self.images:
            raise RuntimeError(
                f"Image was not preloaded before formal timing: {relative_path}"
            )
        self.image = self.images[relative_path]

    def warm_up_audio(self, core, duration_seconds: float = 0.02) -> None:
        """Prepare all persistent Sound objects silently before LabRecorder setup."""
        for stimulus in self.sounds.values():
            try:
                old_volume = float(getattr(stimulus, "volume", 1.0))
            except (TypeError, ValueError):
                old_volume = 1.0
            stimulus.setVolume(0.0)
            stimulus.play()
            core.wait(duration_seconds)
            stimulus.stop()
            stimulus.setVolume(old_volume)

    def get_sound(self, relative_path: str):
        return self.sounds[relative_path]

    def sound_duration_seconds(self, relative_path: str) -> float:
        sound = self.sounds[relative_path]
        duration = getattr(sound, "duration", None)
        if duration is None and hasattr(sound, "getDuration"):
            duration = sound.getDuration()
        return float(duration or 0.0)

    def stop_all(self) -> None:
        for stimulus in self.sounds.values():
            stimulus.stop()

    @property
    def audio_backend(self) -> str:
        first = next(iter(self.sounds.values()))
        return type(first).__module__

    @property
    def speaker(self) -> str:
        first = next(iter(self.sounds.values()))
        return str(getattr(first, "speaker", ""))

    @property
    def preloaded_image_count(self) -> int:
        return len(self.images)
