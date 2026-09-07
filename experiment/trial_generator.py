"""Versioned balanced trial lists and participant Latin-square assignment."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import config


LIST_IDS = tuple("ABCDEF")


def load_objects(path: Path = config.OBJECTS_FILE) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        objects = json.load(handle)
    ids = [item["id"] for item in objects]
    if tuple(ids) != config.OBJECTS:
        raise ValueError(f"objects.json order/content does not match config.py: {ids}")
    return objects


def participant_number_from_id(participant_id: str) -> int | None:
    """Return the first positive numeric component, or None if absent/invalid."""
    match = re.search(r"\d+", participant_id)
    if not match:
        return None
    number = int(match.group())
    return number if number > 0 else None


def latin_square_order(row: int) -> tuple[str, ...]:
    if row not in range(1, 7):
        raise ValueError("Latin-square row must be between 1 and 6")
    offset = row - 1
    return LIST_IDS[offset:] + LIST_IDS[:offset]


def latin_square_row(participant_id: str, override: int | None = None) -> tuple[int | None, int]:
    number = participant_number_from_id(participant_id)
    if number is not None:
        return number, ((number - 1) % 6) + 1
    if override not in range(1, 7):
        raise ValueError("Participant ID has no valid number; enter a Latin-square row from 1 to 6")
    return None, int(override)


def _trial(
    participant_id: str,
    session_id: str,
    seed: int,
    run_number: int | str,
    visibility: str,
    audio_condition: str,
    image_object: dict,
    audio_object: dict | None,
    exemplar: dict,
    pattern_number: int,
    iti_ms: int,
    is_practice: bool,
) -> dict:
    condition = f"{visibility}_{audio_condition}"
    image_path = exemplar["visible_path"] if visibility == "visible" else exemplar["occluded_paths"][pattern_number - 1]
    return {
        "participant_id": participant_id,
        "participant_number": "",
        "session_id": session_id,
        "protocol_version": config.PROTOCOL_VERSION,
        "latin_square_row": "",
        "source_list_id": "",
        "presented_run_number": "",
        "canonical_list_seed": config.CANONICAL_MASTER_SEED,
        "run_id": str(run_number) if is_practice else f"{int(run_number):02d}",
        "trial_index_global": "",
        "trial_index_run": "",
        "visibility": visibility,
        "audio_condition": audio_condition,
        "condition_code": config.CONDITION_CODES[condition],
        "image_object": image_object["id"],
        "image_label": image_object["label"],
        "exemplar_id": exemplar["id"],
        "image_path": image_path,
        "occlusion_level": config.OCCLUSION_LEVEL if visibility == "occluded" else "",
        "occlusion_pattern_id": f"occ_{pattern_number:02d}" if visibility == "occluded" else "",
        "audio_object": audio_object["id"] if audio_object else "",
        "audio_label": audio_object["label"] if audio_object else "",
        "audio_path": audio_object["audio_path"] if audio_object else "",
        "is_congruent": "" if audio_condition == "none" else int(audio_condition == "congruent"),
        "correct_response": image_object["label"],
        "accepted_responses": "|".join(image_object["accepted_responses"]),
        "fixation_ms_planned": config.FIXATION_MS,
        "image_duration_ms_planned": config.IMAGE_DURATION_MS,
        "silent_delay_ms_planned": config.SILENT_DELAY_MS,
        "response_window_ms": config.RESPONSE_WINDOW_MS,
        "iti_ms": iti_ms,
        "is_practice": int(is_practice),
        "random_seed": seed,
    }


def _allowed(candidate: dict, sequence: list[dict]) -> bool:
    if not sequence:
        return True
    previous = sequence[-1]
    if candidate["exemplar_id"] == previous["exemplar_id"]:
        return False
    if candidate["audio_object"] and candidate["audio_object"] == previous["audio_object"]:
        return False
    if len(sequence) >= 2 and all(item["image_object"] == candidate["image_object"] for item in sequence[-2:]):
        return False
    if len(sequence) >= 3 and all(item["condition_code"] == candidate["condition_code"] for item in sequence[-3:]):
        return False
    return True


def constrained_shuffle(candidates: Iterable[dict], seed: int) -> list[dict]:
    source = list(candidates)
    for attempt in range(config.MAX_RANDOMISATION_ATTEMPTS):
        rng = random.Random(seed + attempt * 100_003)
        remaining = source.copy()
        rng.shuffle(remaining)
        ordered: list[dict] = []
        while remaining:
            valid = [index for index, item in enumerate(remaining) if _allowed(item, ordered)]
            if not valid:
                break
            ordered.append(remaining.pop(rng.choice(valid)))
        if len(ordered) == len(source):
            return ordered
    raise RuntimeError("Could not satisfy the constrained-randomisation rules")


def _audio_object(objects: list[dict], image_index: int, occurrence: int, audio_condition: str) -> dict | None:
    if audio_condition == "none":
        return None
    if audio_condition == "congruent":
        return objects[image_index]
    offset = 1 + (occurrence % (len(objects) - 1))
    return objects[(image_index + offset) % len(objects)]


def generate_canonical_templates(overwrite: bool = False) -> list[Path]:
    """Create the six immutable protocol templates using the fixed master seed."""
    objects = load_objects()
    config.RUN_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [config.RUN_TEMPLATE_DIR / f"list_{list_id}.csv" for list_id in LIST_IDS]
    if not overwrite and all(path.exists() for path in paths):
        return paths

    rng = random.Random(config.CANONICAL_MASTER_SEED)
    candidates_by_list: list[list[dict]] = [[] for _ in LIST_IDS]
    conditions = [(v, a) for v in config.VISIBILITIES for a in config.AUDIO_CONDITIONS]
    for object_index, image_object in enumerate(objects):
        for condition_index, (visibility, audio_condition) in enumerate(conditions):
            occurrence = 0
            for list_index in range(len(LIST_IDS)):
                for _ in range(2):
                    exemplar_index = (occurrence + object_index + condition_index) % len(image_object["exemplars"])
                    exemplar = image_object["exemplars"][exemplar_index]
                    audio_object = _audio_object(objects, object_index, occurrence, audio_condition)
                    candidates_by_list[list_index].append(_trial(
                        "", "", config.CANONICAL_MASTER_SEED, list_index + 1,
                        visibility, audio_condition, image_object, audio_object, exemplar,
                        1 + occurrence % config.OCCLUSION_PATTERNS,
                        rng.randint(config.ITI_MIN_MS, config.ITI_MAX_MS), False,
                    ))
                    occurrence += 1

    written: list[Path] = []
    for list_index, (list_id, candidates) in enumerate(zip(LIST_IDS, candidates_by_list)):
        ordered = constrained_shuffle(candidates, config.CANONICAL_MASTER_SEED + (list_index + 1) * 10_007)
        for index, trial in enumerate(ordered, start=1):
            trial.update({
                "source_list_id": list_id,
                "run_id": list_id,
                "trial_index_global": "",
                "trial_index_run": index,
                "presented_run_number": "",
            })
        validate_canonical_list(ordered, list_id)
        path = paths[list_index]
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
            writer.writeheader()
            writer.writerows(ordered)
        written.append(path)
    return written


def _coerce_template_row(row: dict) -> dict:
    integer_fields = {
        "canonical_list_seed", "trial_index_run", "condition_code", "occlusion_level",
        "is_congruent", "fixation_ms_planned", "image_duration_ms_planned",
        "silent_delay_ms_planned", "response_window_ms", "iti_ms", "is_practice", "random_seed",
    }
    result = dict(row)
    for field in integer_fields:
        if result.get(field, "") != "":
            result[field] = int(result[field])
    return result


def load_canonical_template(list_id: str) -> list[dict]:
    path = config.RUN_TEMPLATE_DIR / f"list_{list_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical run template: {path}. Run scripts/generate_run_templates.py once.")
    with path.open(encoding="utf-8-sig") as handle:
        trials = [_coerce_template_row(row) for row in csv.DictReader(handle)]
    validate_canonical_list(trials, list_id)
    return trials


def build_full_main(
    participant_id: str,
    session_id: str,
    seed: int,
    objects: list[dict],
    latin_row_override: int | None = None,
) -> list[dict]:
    del seed, objects
    participant_number, row = latin_square_row(participant_id, latin_row_override)
    manifest: list[dict] = []
    for presented_run, source_list_id in enumerate(latin_square_order(row), start=1):
        for trial in load_canonical_template(source_list_id):
            trial.update({
                "participant_id": participant_id,
                "participant_number": participant_number if participant_number is not None else "",
                "session_id": session_id,
                "protocol_version": config.PROTOCOL_VERSION,
                "latin_square_row": row,
                "source_list_id": source_list_id,
                "presented_run_number": presented_run,
                "run_id": f"{presented_run:02d}",
                "trial_index_global": len(manifest) + 1,
            })
            manifest.append(trial)
    validate_manifest(manifest, mode="full")
    return manifest


def build_pilot_main(participant_id: str, session_id: str, seed: int, objects: list[dict]) -> list[dict]:
    rng = random.Random(seed)
    candidates: list[dict] = []
    conditions = [(v, a) for v in config.VISIBILITIES for a in config.AUDIO_CONDITIONS]
    for object_index, image_object in enumerate(objects):
        for condition_index, (visibility, audio_condition) in enumerate(conditions):
            exemplar = image_object["exemplars"][(object_index + condition_index) % len(image_object["exemplars"])]
            audio_object = _audio_object(objects, object_index, condition_index, audio_condition)
            candidates.append(_trial(
                participant_id, session_id, seed, 1, visibility, audio_condition, image_object,
                audio_object, exemplar, 1 + condition_index % config.OCCLUSION_PATTERNS,
                rng.randint(config.ITI_MIN_MS, config.ITI_MAX_MS), False,
            ))
    ordered = constrained_shuffle(candidates, seed + 77)
    participant_number = participant_number_from_id(participant_id)
    for index, trial in enumerate(ordered, start=1):
        trial.update({
            "participant_number": participant_number or "",
            "source_list_id": "PILOT",
            "presented_run_number": 1,
            "trial_index_global": index,
            "trial_index_run": index,
        })
    validate_manifest(ordered, mode="pilot")
    return ordered


def build_practice(participant_id: str, session_id: str, seed: int, objects: list[dict]) -> list[dict]:
    rng = random.Random(seed + 9_001)
    conditions = [(v, a) for v in config.VISIBILITIES for a in config.AUDIO_CONDITIONS] * 2
    candidates: list[dict] = []
    for index, (visibility, audio_condition) in enumerate(conditions):
        object_index = (index + (index // len(objects)) * 2) % len(objects)
        image_object = objects[object_index]
        exemplar = image_object["exemplars"][index % len(image_object["exemplars"])]
        audio_object = _audio_object(objects, object_index, index, audio_condition)
        candidates.append(_trial(
            participant_id, session_id, seed, "practice", visibility, audio_condition,
            image_object, audio_object, exemplar, 1 + index % config.OCCLUSION_PATTERNS,
            rng.randint(config.ITI_MIN_MS, config.ITI_MAX_MS), True,
        ))
    ordered = constrained_shuffle(candidates, seed + 9_003)
    participant_number = participant_number_from_id(participant_id)
    for index, trial in enumerate(ordered, start=1):
        trial.update({
            "participant_number": participant_number or "",
            "source_list_id": "PRACTICE",
            "presented_run_number": "practice",
            "trial_index_global": -index,
            "trial_index_run": index,
        })
    return ordered


def build_manifests(
    participant_id: str,
    session_id: str,
    mode: str,
    seed: int,
    latin_row_override: int | None = None,
) -> tuple[list[dict], list[dict]]:
    objects = load_objects()
    practice = build_practice(participant_id, session_id, seed, objects)
    main = (
        build_full_main(participant_id, session_id, seed, objects, latin_row_override)
        if mode == "full" else build_pilot_main(participant_id, session_id, seed, objects)
    )
    return practice, main


def validate_canonical_list(trials: list[dict], list_id: str) -> None:
    if len(trials) != config.FULL_TRIALS_PER_RUN:
        raise AssertionError(f"List {list_id} must contain 72 trials")
    if set(Counter(t["condition_code"] for t in trials).values()) != {12}:
        raise AssertionError(f"List {list_id} condition balance failed")
    if set(Counter(t["image_object"] for t in trials).values()) != {12}:
        raise AssertionError(f"List {list_id} object balance failed")
    _validate_sequence(trials)


def _validate_sequence(trials: list[dict]) -> None:
    for index, trial in enumerate(trials):
        previous = trials[max(0, index - 3):index]
        if previous and trial["exemplar_id"] == previous[-1]["exemplar_id"]:
            raise AssertionError("Consecutive exemplar constraint failed")
        if trial["audio_object"] and previous and trial["audio_object"] == previous[-1]["audio_object"]:
            raise AssertionError("Consecutive audio-word constraint failed")
        if len(previous) >= 2 and all(item["image_object"] == trial["image_object"] for item in previous[-2:]):
            raise AssertionError("Object run-length constraint failed")
        if len(previous) >= 3 and all(item["condition_code"] == trial["condition_code"] for item in previous[-3:]):
            raise AssertionError("Condition run-length constraint failed")


def validate_manifest(manifest: list[dict], mode: str) -> None:
    expected = config.FULL_MAIN_TRIALS if mode == "full" else config.PILOT_MAIN_TRIALS
    if len(manifest) != expected:
        raise AssertionError(f"Expected {expected} trials, got {len(manifest)}")
    for trial in manifest:
        if trial["protocol_version"] != config.PROTOCOL_VERSION:
            raise AssertionError("Protocol version is missing or inconsistent")
        if trial["audio_condition"] == "congruent" and trial["audio_object"] != trial["image_object"]:
            raise AssertionError("Congruent mapping is invalid")
        if trial["audio_condition"] == "incongruent" and trial["audio_object"] == trial["image_object"]:
            raise AssertionError("Incongruent mapping is invalid")

    by_run: dict[str, list[dict]] = {}
    for trial in manifest:
        by_run.setdefault(trial["run_id"], []).append(trial)
    for run_trials in by_run.values():
        _validate_sequence(run_trials)

    if mode == "full":
        if len(by_run) != config.FULL_RUNS or any(len(items) != config.FULL_TRIALS_PER_RUN for items in by_run.values()):
            raise AssertionError("Full run structure is invalid")
        object_condition = Counter((t["image_object"], t["condition_code"]) for t in manifest)
        if set(object_condition.values()) != {config.FULL_REPETITIONS_PER_OBJECT_CONDITION}:
            raise AssertionError("Object x condition counts are not exactly balanced")
        exemplar_condition = Counter((t["exemplar_id"], t["condition_code"]) for t in manifest)
        if set(exemplar_condition.values()) != {3}:
            raise AssertionError("Each exemplar x condition must occur exactly three times")
        for run_trials in by_run.values():
            if set(Counter(t["condition_code"] for t in run_trials).values()) != {12}:
                raise AssertionError("Condition balance within a run failed")
            if set(Counter(t["image_object"] for t in run_trials).values()) != {12}:
                raise AssertionError("Object balance within a run failed")
