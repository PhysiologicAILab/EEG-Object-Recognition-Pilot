"""LSL JSON marker outlet with duplicate prevention and trial audits."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import config


TRIAL_SINGLETONS = {
    "trial_start", "fixation_onset", "visible_none_image_onset",
    "visible_congruent_image_onset", "visible_incongruent_image_onset",
    "occluded_none_image_onset", "occluded_congruent_image_onset",
    "occluded_incongruent_image_onset", "congruent_audio_onset",
    "incongruent_audio_onset", "image_offset", "silent_delay_onset",
    "response_cue_onset", "microphone_recording_start",
    "microphone_recording_stop", "trial_end",
}


class MarkerOutlet:
    def __init__(self, participant_id: str, session_id: str) -> None:
        from pylsl import StreamInfo, StreamOutlet

        source_id = f"{config.LSL_SOURCE_PREFIX}-{participant_id}-{session_id}-{config.PROTOCOL_VERSION}"
        stream_info = StreamInfo(
            config.LSL_STREAM_NAME, config.LSL_STREAM_TYPE, 1, 0, "string", source_id,
        )
        description = stream_info.desc()
        description.append_child_value("manufacturer", "PsychoPy")
        description.append_child_value("participant_id", participant_id)
        description.append_child_value("session_id", session_id)
        description.append_child_value("protocol_version", config.PROTOCOL_VERSION)
        self.outlet = StreamOutlet(stream_info)
        self.participant_id = participant_id
        self.session_id = session_id
        self.sent_keys: set[tuple] = set()
        self.trial_payloads: dict[tuple, list[dict]] = {}
        self.info = {
            "name": config.LSL_STREAM_NAME,
            "type": config.LSL_STREAM_TYPE,
            "channel_count": 1,
            "channel_format": "string",
            "nominal_sampling_rate": 0,
            "source_id": source_id,
            "protocol_version": config.PROTOCOL_VERSION,
        }

    @staticmethod
    def _trial_key(context: dict) -> tuple:
        return (int(bool(context.get("is_practice", 0))), str(context.get("trial_index_global", "")))

    def _dedup_key(self, event_name: str, context: dict) -> tuple | None:
        if event_name in TRIAL_SINGLETONS:
            trial_key = self._trial_key(context)
            if trial_key[1] == "":
                raise RuntimeError(f"Trial-scoped marker {event_name} has no global trial index")
            return ("trial", *trial_key, event_name)
        if event_name in {"run_start", "run_end", "break_start", "break_end"}:
            return ("run", str(context.get("presented_run_number", context.get("run_id", ""))), event_name)
        if event_name in {"experiment_start", "experiment_end", "experiment_abort", "practice_start", "practice_end"}:
            return ("session", event_name)
        return None

    def push(self, event_name: str, event_code: int, context: dict, psychopy_time: float) -> dict:
        from pylsl import local_clock

        dedup_key = self._dedup_key(event_name, context)
        if dedup_key is not None and dedup_key in self.sent_keys:
            raise RuntimeError(f"Duplicate marker prevented: {event_name} / {dedup_key}")
        lsl_time = local_clock()
        condition = (
            f"{context.get('visibility', '')}_{context.get('audio_condition', '')}"
            if context.get("visibility") else ""
        )
        payload = {
            "marker_code": int(event_code),
            "event_code": int(event_code),
            "event_name": event_name,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "presented_run_number": context.get("presented_run_number", context.get("run_id", "")),
            "run_id": context.get("run_id", ""),
            "global_trial_index": context.get("trial_index_global", ""),
            "trial_index_global": context.get("trial_index_global", ""),
            "within_run_trial_index": context.get("trial_index_run", ""),
            "trial_index_run": context.get("trial_index_run", ""),
            "condition": condition,
            "condition_code": context.get("condition_code", ""),
            "visibility": context.get("visibility", ""),
            "audio_condition": context.get("audio_condition", ""),
            "object_label": context.get("image_label", context.get("image_object", "")),
            "image_object": context.get("image_object", ""),
            "audio_object": context.get("audio_object", ""),
            "spoken_condition_word": context.get("audio_label", ""),
            "image_filename": context.get("image_path", ""),
            "audio_filename": context.get("audio_path", ""),
            "source_list_id": context.get("source_list_id", ""),
            "protocol_version": config.PROTOCOL_VERSION,
            "exemplar_id": context.get("exemplar_id", ""),
            "psychopy_time": psychopy_time,
            "lsl_local_clock_time": lsl_time,
        }
        self.outlet.push_sample([json.dumps(payload, separators=(",", ":"))], timestamp=lsl_time)
        if dedup_key is not None:
            self.sent_keys.add(dedup_key)
        if event_name in TRIAL_SINGLETONS:
            self.trial_payloads.setdefault(self._trial_key(context), []).append(payload)
        return {
            "participant_id": self.participant_id,
            "participant_number": context.get("participant_number", ""),
            "session_id": self.session_id,
            "protocol_version": config.PROTOCOL_VERSION,
            "latin_square_row": context.get("latin_square_row", ""),
            "source_list_id": context.get("source_list_id", ""),
            "presented_run_number": context.get("presented_run_number", ""),
            "run_id": context.get("run_id", ""),
            "trial_index_global": context.get("trial_index_global", ""),
            "trial_index_run": context.get("trial_index_run", ""),
            "condition": condition,
            "condition_code": context.get("condition_code", ""),
            "object_label": context.get("image_label", context.get("image_object", "")),
            "event_name": event_name,
            "event_code": int(event_code),
            "psychopy_timestamp": psychopy_time,
            "lsl_timestamp": lsl_time,
            "browser_or_system_datetime_iso": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "marker_sent": True,
            "notes": "",
        }

    def validate_trial(self, context: dict) -> None:
        payloads = self.trial_payloads.get(self._trial_key(context), [])
        names = [payload["event_name"] for payload in payloads]
        expected = [
            "trial_start", "fixation_onset",
            f"{context['visibility']}_{context['audio_condition']}_image_onset",
        ]
        if context["audio_condition"] != "none":
            expected.append(f"{context['audio_condition']}_audio_onset")
        expected.extend([
            "image_offset", "silent_delay_onset", "microphone_recording_start",
            "response_cue_onset", "microphone_recording_stop", "trial_end",
        ])
        counts = Counter(names)
        missing = [name for name in expected if counts[name] != 1]
        unexpected = [name for name, count in counts.items() if count != 1 or name not in expected]
        positions = [names.index(name) for name in expected if name in names]
        if missing or unexpected or positions != sorted(positions):
            raise RuntimeError(
                f"Marker integrity failure for trial {context.get('trial_index_global')}: "
                f"missing_or_bad={missing}; unexpected_or_duplicate={unexpected}; order={names}"
            )

    def self_test(self, psychopy_time: float) -> dict:
        return self.push("lsl_self_test", config.TRIGGER_CODES["lsl_self_test"], {}, psychopy_time)
