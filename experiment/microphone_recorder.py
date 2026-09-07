"""Persistent PsychoPy microphone stream and one-file-per-trial saving."""

from __future__ import annotations

import hashlib
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

import config


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class RecordingError(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def validate_wav(path: Path, expected_duration_ms: float | None = None) -> dict:
    result = {
        "file_exists": path.is_file(),
        "wav_duration_ms": 0.0,
        "sample_count": 0,
        "peak_amplitude": 0.0,
        "wav_sample_rate": "",
        "wav_channels": "",
        "reopen_ok": False,
    }
    if not result["file_exists"]:
        return result
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        result.update({
            "wav_duration_ms": round((len(samples) / sample_rate) * 1000, 3) if sample_rate else 0.0,
            "sample_count": int(samples.size),
            "peak_amplitude": float(np.max(np.abs(samples))) if samples.size else 0.0,
            "wav_sample_rate": int(sample_rate),
            "wav_channels": int(samples.shape[1]),
            "reopen_ok": True,
        })
    except Exception as exc:
        result["validation_error"] = f"{type(exc).__name__}: {exc}"
        return result
    if expected_duration_ms is not None:
        result["duration_error_ms"] = round(result["wav_duration_ms"] - expected_duration_ms, 3)
        result["duration_close"] = abs(result["duration_error_ms"]) <= config.WAV_DURATION_TOLERANCE_MS
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrialMicrophone:
    """One hardware object; its PTB device stream stays open between segments."""

    def __init__(self, recording_folder: Path, allow_mock: bool = False) -> None:
        self.recording_folder = recording_folder
        self.allow_mock = allow_mock
        self.microphone = None
        self.mock = False
        self.recording = False
        self.initialised = False
        self.warmed_up = False
        self.device_name = ""
        self.device_index = ""
        self.sample_rate = ""
        self.channels = ""
        self.stream_initialised_time = ""
        self.initialise_count = 0

    def initialise(self) -> None:
        if self.initialised:
            return
        self.recording_folder.mkdir(parents=True, exist_ok=True)
        try:
            from psychopy.sound.microphone import Microphone

            self.microphone = Microphone(
                recordingFolder=self.recording_folder,
                recordingExt="wav",
                name="trial_mic",
                streamBufferSecs=6.0,
                audioRunMode=1,
            )
            device = self.microphone.device
            profile = getattr(device, "_device", None)
            self.device_name = str(getattr(profile, "deviceName", device))
            self.device_index = getattr(profile, "deviceIndex", getattr(device, "index", ""))
            self.sample_rate = int(device.sampleRateHz)
            self.channels = int(device.channels)
        except Exception as exc:
            if not self.allow_mock:
                raise RecordingError("no_device", f"Could not initialise microphone: {exc}") from exc
            self.mock = True
            self.device_name = "PILOT_MOCK_MICROPHONE"
            self.sample_rate = 44_100
            self.channels = 1
        self.initialised = True
        self.initialise_count += 1
        self.stream_initialised_time = iso_now()

    def warm_up(self, duration_seconds: float = 0.25, force: bool = False) -> None:
        """Start once to warm the device; stop does not close PsychoPy's PTB stream."""
        if self.warmed_up and not force:
            return
        if not self.initialised:
            self.initialise()
        if self.mock:
            self.warmed_up = True
            return
        from psychopy import core

        try:
            self.microphone.start(waitForStart=1)
            core.wait(duration_seconds)
            self.microphone.stop(blockUntilStopped=True)
            self.microphone.bank(tag="hardware_warmup", transcribe=False)
            self.microphone.clips.pop("hardware_warmup", None)
            self.warmed_up = True
        except Exception as exc:
            raise RecordingError("stream_not_active", f"Microphone warm-up failed: {exc}") from exc

    def start(self) -> dict:
        if not self.initialised or not self.warmed_up:
            raise RecordingError("stream_not_active", "Microphone stream is not initialised and warm")
        if self.recording:
            raise RecordingError("segment_start_failed", "A recording segment is already active")
        started_iso = iso_now()
        try:
            hardware_start_time = "" if self.mock else self.microphone.start(waitForStart=1)
        except Exception as exc:
            raise RecordingError("segment_start_failed", f"Could not start response segment: {exc}") from exc
        self.recording = True
        return {
            "recording_started_iso": started_iso,
            "hardware_start_time": hardware_start_time,
            "status": "recording",
        }

    def stop_capture(self) -> dict:
        """Stop and bank immediately; disk saving is deliberately a separate step."""
        result = {"recording_status": "stop_failed", "recording_stopped_iso": iso_now(), "clip": None, "tag": ""}
        if not self.recording:
            result["recording_status"] = "stream_not_active"
            return result
        self.recording = False
        if self.mock:
            result["recording_status"] = "captured_mock"
            return result
        try:
            self.microphone.stop(blockUntilStopped=True)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result
        tag = f"segment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        try:
            clip = self.microphone.bank(tag=tag, transcribe=False)
        except Exception as exc:
            result.update({"recording_status": "empty_buffer", "error": f"{type(exc).__name__}: {exc}"})
            return result
        if clip is None:
            result["recording_status"] = "empty_buffer"
            return result
        result.update({"recording_status": "captured", "clip": clip, "tag": tag})
        return result

    def _remove_verified_duplicates(self, destination: Path) -> list[str]:
        """Delete only temporary files that are byte-identical to the verified canonical WAV."""
        removed = []
        final_hash = _sha256(destination)
        for temporary in destination.parent.glob("recording_trial_mic_*.wav"):
            if temporary.resolve() != destination.resolve() and _sha256(temporary) == final_hash:
                temporary.unlink()
                removed.append(str(temporary))
        return removed

    def save_segment(self, segment: dict, destination: Path, expected_duration_ms: float) -> dict:
        result = {
            "recording_status": segment.get("recording_status", "save_failed"),
            "recording_stopped_iso": segment.get("recording_stopped_iso", iso_now()),
            "final_wav_path": str(destination),
            "temporary_files_removed": "",
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.mock:
            frame_count = max(1, round(expected_duration_ms / 1000 * int(self.sample_rate)))
            with wave.open(str(destination), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(int(self.sample_rate))
                handle.writeframes(b"\x00\x00" * frame_count)
            result.update(validate_wav(destination, expected_duration_ms))
            result["recording_status"] = "zero_amplitude"
            return result

        clip = segment.get("clip")
        tag = segment.get("tag", "")
        if clip is None:
            return result
        try:
            clip.save(destination)
        except Exception as exc:
            result.update({"recording_status": "save_failed", "error": f"{type(exc).__name__}: {exc}"})
            return result
        result.update(validate_wav(destination, expected_duration_ms))
        if not result["file_exists"]:
            result["recording_status"] = "save_failed"
        elif not result.get("reopen_ok"):
            result["recording_status"] = "reopen_failed"
        elif result["sample_count"] <= 0 or result["wav_duration_ms"] <= 0:
            result["recording_status"] = "empty_buffer"
        elif result["peak_amplitude"] <= config.MICROPHONE_PEAK_THRESHOLD:
            result["recording_status"] = "zero_amplitude"
        elif not result.get("duration_close", False):
            result["recording_status"] = "duration_invalid"
        else:
            result["recording_status"] = "saved"

        # Only a verified final file permits removal from PsychoPy's clip bank.
        if result["recording_status"] == "saved":
            self.microphone.clips.pop(tag, None)
            result["temporary_files_removed"] = "|".join(self._remove_verified_duplicates(destination))
        return result

    def stop_and_save(self, destination: Path, expected_duration_ms: float) -> dict:
        return self.save_segment(self.stop_capture(), destination, expected_duration_ms)

    def abort(self, destination: Path | None = None, expected_duration_ms: float = 1000) -> dict | None:
        if not self.recording:
            return None
        segment = self.stop_capture()
        return self.save_segment(segment, destination, expected_duration_ms) if destination else segment

    def close(self) -> None:
        if self.recording:
            self.abort()
        if self.microphone is not None:
            # Failed clips remain recoverable; successful ones were removed after validation.
            if self.microphone.clips:
                self.microphone.saveClips(clear=True)
            self.microphone.close()
