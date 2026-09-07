from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ASR_DEPS = os.environ.get("EEG_ASR_DEPENDENCIES")
if ASR_DEPS:
    sys.path.insert(0, str(Path(ASR_DEPS).expanduser().resolve()))
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

OUT = Path(
    os.environ.get(
        "BEHAVIOURAL_ANALYSIS_ROOT",
        PROJECT_ROOT / "processed_data" / "behavioural_speech_analysis",
    )
).resolve()
INFILE = OUT / "validation_subset_results.csv"
MODEL_CACHE = os.environ.get("EEG_ASR_MODEL_CACHE")
LABELS = ["bowl", "cup", "plate", "knife", "fork", "spoon"]
VAD_OPTIONS = VadOptions(threshold=0.50, min_speech_duration_ms=150,
                         min_silence_duration_ms=150, speech_pad_ms=30)


def ctc_log_probability(log_probs: torch.Tensor, tokens: list[int], blank: int) -> float:
    """Exact CTC sequence log probability, summing all valid alignment paths."""
    extended = [blank]
    for token in tokens:
        extended.extend([token, blank])
    s_count = len(extended)
    previous = torch.full((s_count,), -torch.inf, dtype=log_probs.dtype)
    previous[0] = log_probs[0, blank]
    if s_count > 1:
        previous[1] = log_probs[0, extended[1]]
    for t in range(1, log_probs.shape[0]):
        current = torch.full_like(previous, -torch.inf)
        for s, token in enumerate(extended):
            paths = [previous[s]]
            if s >= 1:
                paths.append(previous[s - 1])
            if s >= 2 and token != blank and token != extended[s - 2]:
                paths.append(previous[s - 2])
            current[s] = torch.logsumexp(torch.stack(paths), dim=0) + log_probs[t, token]
        previous = current
    return float(torch.logsumexp(previous[-2:], dim=0))


def main() -> None:
    source = pd.read_csv(INFILE, low_memory=False)
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h", cache_dir=MODEL_CACHE, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h", cache_dir=MODEL_CACHE, local_files_only=True)
    model.eval()
    blank = int(processor.tokenizer.pad_token_id)
    label_tokens = {
        label: processor.tokenizer(label.upper(), add_special_tokens=False).input_ids
        for label in LABELS
    }
    print({"blank": blank, "label_tokens": label_tokens}, flush=True)

    rows = []
    for batch_start in range(0, len(source), 12):
        part = source.iloc[batch_start:batch_start + 12]
        arrays = []
        vad_detected = []
        for wav_file in part.wav_file:
            audio = decode_audio(wav_file, sampling_rate=16000)
            stamps = get_speech_timestamps(audio, VAD_OPTIONS, sampling_rate=16000)
            arrays.append(np.concatenate([audio[x["start"]:x["end"]] for x in stamps]) if stamps else audio)
            vad_detected.append(bool(stamps))
        inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True,
                           return_attention_mask=True)
        input_lengths = inputs.attention_mask.sum(-1)
        with torch.inference_mode():
            logits = model(inputs.input_values).logits
            output_lengths = model._get_feat_extract_output_lengths(input_lengths).to(torch.long)
        for i, length in enumerate(output_lengths.tolist()):
            lp = torch.log_softmax(logits[i, :length], dim=-1).cpu()
            scores = {label: ctc_log_probability(lp, ids, blank) for label, ids in label_tokens.items()}
            ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            score_values = torch.tensor([scores[x] for x in LABELS], dtype=torch.float64)
            probabilities = torch.softmax(score_values, dim=0).numpy()
            prob_map = dict(zip(LABELS, probabilities.tolist()))
            rows.append({
                "constrained_predicted_label": ordered[0][0],
                "constrained_top_probability": prob_map[ordered[0][0]],
                "constrained_second_label": ordered[1][0],
                "constrained_second_probability": prob_map[ordered[1][0]],
                "constrained_log_score_margin": ordered[0][1] - ordered[1][1],
                "constrained_scores_json": json.dumps(scores),
                "constrained_vad_detected": vad_detected[i],
            })
        print(f"Constrained validation progress: {min(batch_start + 12, len(source))}/{len(source)}", flush=True)

    result = pd.concat([source.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result["constrained_matches_target"] = result.constrained_predicted_label.eq(result.target_label)
    result.to_csv(OUT / "validation_subset_constrained_results.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n": len(result),
        "target_agreement": float(result.constrained_matches_target.mean()),
        "median_top_probability": float(result.constrained_top_probability.median()),
        "p10_top_probability": float(result.constrained_top_probability.quantile(.10)),
        "median_log_margin": float(result.constrained_log_score_margin.median()),
        "errors": result.loc[~result.constrained_matches_target,
                             ["validation_id", "participant", "target_label",
                              "constrained_predicted_label", "constrained_top_probability",
                              "constrained_second_label", "raw_asr_transcript",
                              "wav2vec_raw_transcript"]].to_dict("records"),
    }
    (OUT / "validation_constrained_decision.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
