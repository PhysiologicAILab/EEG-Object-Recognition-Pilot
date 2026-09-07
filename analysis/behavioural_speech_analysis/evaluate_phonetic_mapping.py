from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ASR_DEPS = os.environ.get("EEG_ASR_DEPENDENCIES")
if ASR_DEPS:
    sys.path.insert(0, str(Path(ASR_DEPS).expanduser().resolve()))
import pronouncing

OUT = Path(
    os.environ.get(
        "BEHAVIOURAL_ANALYSIS_ROOT",
        PROJECT_ROOT / "processed_data" / "behavioural_speech_analysis",
    )
).resolve()
LABELS = ["bowl", "cup", "plate", "knife", "fork", "spoon"]


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def phones(word: str) -> list[str] | None:
    found = pronouncing.phones_for_word(word.lower())
    if not found:
        return None
    return [re.sub(r"\d", "", x) for x in found[0].split()]


TARGET_PHONES = {x: phones(x) for x in LABELS}


def nearest_from_text(text: str) -> tuple[str | None, float, str | None]:
    tokens = re.findall(r"[a-z]+", str(text).lower())
    candidates = []
    for token in tokens:
        source = phones(token)
        for label in LABELS:
            target = TARGET_PHONES[label]
            if source:
                distance = edit_distance(source, target) / max(len(source), len(target))
            else:
                distance = edit_distance(token, label) / max(len(token), len(label))
            candidates.append((distance, label, token))
    if not candidates:
        return None, float("nan"), None
    candidates.sort()
    return candidates[0][1], candidates[0][0], candidates[0][2]


def main():
    df = pd.read_csv(OUT / "validation_subset_constrained_results.csv", low_memory=False)
    for prefix, column in [("whisper_phonetic", "raw_asr_transcript"),
                           ("wav2vec_phonetic", "wav2vec_raw_transcript")]:
        mapped = [nearest_from_text(x) for x in df[column].fillna("")]
        df[f"{prefix}_label"] = [x[0] for x in mapped]
        df[f"{prefix}_distance"] = [x[1] for x in mapped]
        df[f"{prefix}_source_token"] = [x[2] for x in mapped]
        print(prefix, "all", df[f"{prefix}_label"].eq(df.target_label).mean())
        for threshold in [0.25, 0.34, 0.50, 0.67]:
            use = df[f"{prefix}_distance"].le(threshold)
            print(prefix, threshold, "coverage", use.mean(), "target agreement", df.loc[use, f"{prefix}_label"].eq(df.loc[use, "target_label"]).mean())
    agree = df.whisper_phonetic_label.eq(df.wav2vec_phonetic_label) & df.whisper_phonetic_label.notna()
    print("phonetic cross-model agree coverage", agree.mean(), "target agreement", df.loc[agree, "whisper_phonetic_label"].eq(df.loc[agree, "target_label"]).mean())
    cagree = df.constrained_predicted_label.eq(df.wav2vec_phonetic_label) & df.wav2vec_phonetic_label.notna()
    print("constrained/w2v-phonetic agree coverage", cagree.mean(), "target agreement", df.loc[cagree, "constrained_predicted_label"].eq(df.loc[cagree, "target_label"]).mean())
    print(df[["validation_id", "target_label", "raw_asr_transcript", "whisper_phonetic_label", "whisper_phonetic_distance", "wav2vec_raw_transcript", "wav2vec_phonetic_label", "wav2vec_phonetic_distance", "constrained_predicted_label"]].to_string(index=False))
    df.to_csv(OUT / "validation_subset_constrained_phonetic_results.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
