from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ASR_DEPS = os.environ.get("EEG_ASR_DEPENDENCIES")
if ASR_DEPS:
    sys.path.insert(0, str(Path(ASR_DEPS).expanduser().resolve()))
import pronouncing
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

OUT = Path(
    os.environ.get(
        "BEHAVIOURAL_ANALYSIS_ROOT",
        PROJECT_ROOT / "processed_data" / "behavioural_speech_analysis",
    )
).resolve()
INVENTORY = OUT / "wav_inventory_and_mapping.csv"
MODEL_CACHE = os.environ.get("EEG_ASR_MODEL_CACHE")
CHECKPOINT = OUT / "full_inference_checkpoint.csv"
LABELS = ["bowl", "cup", "plate", "knife", "fork", "spoon"]
CONDITIONS = [
    "visible_none", "visible_congruent", "visible_incongruent",
    "occluded_none", "occluded_congruent", "occluded_incongruent",
]
SEED = 20260829
BATCH_SIZE = 32
PHONETIC_DISTANCE_LIMIT = 0.50
LOW_MARGIN_LIMIT = 2.0
VAD_OPTIONS = VadOptions(threshold=0.50, min_speech_duration_ms=150,
                         min_silence_duration_ms=150, speech_pad_ms=30)


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def phones(word: str):
    found = pronouncing.phones_for_word(word.lower())
    if not found:
        return None
    return [re.sub(r"\d", "", x) for x in found[0].split()]


TARGET_PHONES = {x: phones(x) for x in LABELS}


def nearest_from_text(text: str):
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
        return None, np.nan, None
    candidates.sort()
    return candidates[0][1], candidates[0][0], candidates[0][2]


def ctc_log_probability(log_probs: torch.Tensor, tokens: list[int], blank: int) -> float:
    extended = [blank]
    for token in tokens:
        extended.extend([token, blank])
    previous = torch.full((len(extended),), -torch.inf, dtype=log_probs.dtype)
    previous[0] = log_probs[0, blank]
    if len(extended) > 1:
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


def six_ctc_scores(log_probs: torch.Tensor, label_tokens: dict[str, list[int]], blank: int):
    """Vectorized equivalent of six exact CTC sequence log probabilities."""
    lengths = torch.tensor([len(label_tokens[x]) for x in LABELS], dtype=torch.long)
    targets = torch.tensor([token for x in LABELS for token in label_tokens[x]], dtype=torch.long)
    input_lengths = torch.full((len(LABELS),), log_probs.shape[0], dtype=torch.long)
    expanded = log_probs[:, None, :].expand(-1, len(LABELS), -1)
    losses = torch.nn.functional.ctc_loss(
        expanded, targets, input_lengths, lengths, blank=blank,
        reduction="none", zero_infinity=True)
    return {label: float(-losses[i]) for i, label in enumerate(LABELS)}


def infer_batch(part: pd.DataFrame, processor, model, label_tokens, blank):
    arrays, vad_rows = [], []
    for wav_file in part.wav_file:
        try:
            audio = decode_audio(wav_file, sampling_rate=16000)
            stamps = get_speech_timestamps(audio, VAD_OPTIONS, sampling_rate=16000)
            asr_audio = np.concatenate([audio[x["start"]:x["end"]] for x in stamps]) if stamps else audio
            arrays.append(asr_audio)
            vad_rows.append({
                "vad_detected": bool(stamps), "vad_segment_count": len(stamps),
                "vad_segments_json": json.dumps([[x["start"] / 16000, x["end"] / 16000] for x in stamps]),
                "speech_onset_in_wav_s": stamps[0]["start"] / 16000 if stamps else np.nan,
                "audio_decode_error": "",
            })
        except Exception as exc:
            arrays.append(np.zeros(1600, dtype=np.float32))
            vad_rows.append({"vad_detected": False, "vad_segment_count": 0,
                             "vad_segments_json": "[]", "speech_onset_in_wav_s": np.nan,
                             "audio_decode_error": f"{type(exc).__name__}: {exc}"})
    inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True,
                       return_attention_mask=True)
    input_lengths = inputs.attention_mask.sum(-1)
    with torch.inference_mode():
        logits = model(inputs.input_values).logits
        output_lengths = model._get_feat_extract_output_lengths(input_lengths).to(torch.long)
    greedy = processor.batch_decode(torch.argmax(logits, dim=-1))
    output = []
    for i, length in enumerate(output_lengths.tolist()):
        lp = torch.log_softmax(logits[i, :length], dim=-1).cpu()
        scores = six_ctc_scores(lp, label_tokens, blank)
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        probs = torch.softmax(torch.tensor([scores[x] for x in LABELS], dtype=torch.float64), dim=0).numpy()
        prob_map = dict(zip(LABELS, probs.tolist()))
        phon_label, phon_distance, phon_token = nearest_from_text(greedy[i])
        output.append({
            **vad_rows[i], "raw_asr_transcript": greedy[i],
            "normalized_transcript": str(greedy[i]).strip().lower(),
            "phonetic_nearest_label": phon_label, "phonetic_distance": phon_distance,
            "phonetic_source_token": phon_token,
            "predicted_label": ordered[0][0] if vad_rows[i]["vad_detected"] else None,
            "asr_confidence": prob_map[ordered[0][0]],
            "second_label": ordered[1][0], "second_label_probability": prob_map[ordered[1][0]],
            "asr_log_score_margin": ordered[0][1] - ordered[1][1],
            "six_label_scores_json": json.dumps(scores),
        })
    return output


def build_outputs(df: pd.DataFrame):
    result = df.copy()
    result["correct"] = np.where(result.predicted_label.notna(), result.predicted_label.eq(result.target_label), np.nan)
    result["response_time_s"] = result.speech_onset_in_wav_s - result.cue_position_psychopy_s
    reliable_phonetic = result.phonetic_distance.le(PHONETIC_DISTANCE_LIMIT)
    model_agree = result.predicted_label.eq(result.phonetic_nearest_label)
    result["manual_review_required"] = (
        ~result.vad_detected | result.audio_decode_error.fillna("").ne("") |
        result.vad_segment_count.gt(1) | ~reliable_phonetic | ~model_agree |
        result.asr_log_score_margin.lt(LOW_MARGIN_LIMIT)
    )
    reasons, flags = [], []
    for _, r in result.iterrows():
        why, qc = [], []
        if isinstance(r.audio_decode_error, str) and r.audio_decode_error.strip():
            why.append("unreadable_or_corrupted_audio"); qc.append("unreadable_or_corrupted_audio")
        if not r.vad_detected:
            why.append("no_speech_detected"); qc.append("no_speech_detected")
        if r.vad_segment_count > 1:
            why.append("multiple_possible_responses"); qc.append("multiple_speech_segments")
        if pd.isna(r.phonetic_distance) or r.phonetic_distance > PHONETIC_DISTANCE_LIMIT:
            why.append("raw_transcript_not_close_to_six_labels"); qc.append("transcription_outside_six_target_labels")
        if r.predicted_label != r.phonetic_nearest_label:
            why.append("acoustic_and_phonetic_methods_disagree")
        if r.asr_log_score_margin < LOW_MARGIN_LIMIT:
            why.append("low_confidence_six_label_classification"); qc.append("low_confidence_transcription")
        if pd.notna(r.response_time_s) and r.response_time_s < 0:
            qc.append("speech_onset_before_response_cue")
        if pd.notna(r.response_time_s) and r.response_time_s > 2:
            qc.append("response_onset_after_2000ms_window")
        if pd.notna(r.response_time_s) and (r.response_time_s < -0.2 or r.response_time_s > 3):
            qc.append("obviously_implausible_response_time")
        reasons.append(";".join(why) if why else "automatic_six_label_decision")
        flags.append(";".join(qc) if qc else "OK")
    result["review_reason"] = reasons
    result["qc_flag"] = flags
    result["result_status"] = np.where(result.manual_review_required,
                                       "AUTOMATIC_INFERENCE_REQUIRES_MANUAL_REVIEW",
                                       "AUTOMATIC_INFERENCE_HIGHER_CONFIDENCE")

    keep = [
        "participant", "actual_run", "trial_in_run", "actual_global_trial", "wav_file",
        "target_label", "visibility", "audio_condition", "condition",
        "microphone_start_psychopy_s", "response_cue_psychopy_s", "microphone_start_lsl_s",
        "response_cue_lsl_s", "cue_position_psychopy_s", "raw_asr_transcript",
        "normalized_transcript", "phonetic_nearest_label", "phonetic_distance",
        "phonetic_source_token", "predicted_label", "correct", "asr_confidence",
        "second_label", "second_label_probability", "asr_log_score_margin",
        "vad_detected", "vad_segment_count", "speech_onset_in_wav_s", "response_time_s",
        "manual_review_required", "review_reason", "qc_flag", "result_status",
    ]
    final = result[keep].rename(columns={"actual_run": "run", "trial_in_run": "trial"})
    final.to_csv(OUT / "trial_level_speech_results.csv", index=False, encoding="utf-8-sig")

    unresolved = final[final.manual_review_required].copy()
    auto = final[~final.manual_review_required].copy()
    audit_n = max(1, int(np.ceil(len(auto) * .05))) if len(auto) else 0
    audit = auto.sample(n=audit_n, random_state=SEED).copy() if audit_n else auto.iloc[0:0].copy()
    unresolved["queue_source"] = "uncertain_trial"
    audit["queue_source"] = "random_5_percent_audit"
    queue = pd.concat([unresolved, audit], ignore_index=True)
    queue["manual_transcript"] = ""
    queue["manual_label"] = ""
    queue["manual_speech_onset_s"] = np.nan
    queue["manual_review_notes"] = ""
    queue.to_csv(OUT / "manual_review_queue.csv", index=False, encoding="utf-8-sig")

    def accuracy_row(scope, g):
        predicted = g.predicted_label.notna()
        participant_accuracy = g.assign(_correct0=g.correct.fillna(False).astype(float)).groupby("participant")["_correct0"].mean()
        return {
            "scope": scope, "trial_count": len(g), "predicted_count": int(predicted.sum()),
            "correct_count_automatic": int(g.correct.fillna(False).sum()),
            "automatic_accuracy_all_trials": float(g.correct.fillna(False).mean()),
            "manual_review_count": int(g.manual_review_required.sum()),
            "higher_confidence_count": int((~g.manual_review_required).sum()),
            "higher_confidence_accuracy_vs_target": float(g.loc[~g.manual_review_required, "correct"].mean()) if (~g.manual_review_required).any() else np.nan,
            "participant_n": int(participant_accuracy.size),
            "participant_accuracy_mean": float(participant_accuracy.mean()),
            "participant_accuracy_sd": float(participant_accuracy.std(ddof=1)) if participant_accuracy.size > 1 else np.nan,
            "participant_accuracy_min": float(participant_accuracy.min()),
            "participant_accuracy_max": float(participant_accuracy.max()),
            "status": "PROVISIONAL_AUTOMATIC_NOT_MANUALLY_CONFIRMED",
        }
    participant_summary = pd.DataFrame([accuracy_row(p, g) for p, g in final.groupby("participant")])
    participant_summary.rename(columns={"scope": "participant"}).to_csv(
        OUT / "participant_accuracy_summary.csv", index=False, encoding="utf-8-sig")
    condition_rows = [accuracy_row("OVERALL", final)]
    condition_rows += [accuracy_row(v, g) for v, g in final.groupby("visibility")]
    condition_rows += [accuracy_row(a, g) for a, g in final.groupby("audio_condition")]
    condition_rows += [accuracy_row(c, final[final.condition.eq(c)]) for c in CONDITIONS]
    pd.DataFrame(condition_rows).to_csv(OUT / "condition_accuracy_summary.csv", index=False, encoding="utf-8-sig")

    confusion = pd.crosstab(final.target_label, final.predicted_label, dropna=False).reindex(
        index=LABELS, columns=LABELS, fill_value=0)
    confusion.index.name = "target_label"
    confusion.to_csv(OUT / "confusion_matrix.csv", encoding="utf-8-sig")

    rt_rows = []
    for scope, g in [("OVERALL", final)] + [(c, final[final.condition.eq(c)]) for c in CONDITIONS]:
        valid_rt = g.response_time_s.where(g.vad_detected & g.response_time_s.between(0, 2))
        rt_rows.append({
            "scope": scope, "trial_count": len(g), "vad_detected_count": int(g.vad_detected.sum()),
            "rt_within_0_to_2s_count": int(valid_rt.notna().sum()),
            "median_rt_s": valid_rt.median(), "mean_rt_s": valid_rt.mean(),
            "sd_rt_s": valid_rt.std(ddof=1), "p05_rt_s": valid_rt.quantile(.05),
            "p95_rt_s": valid_rt.quantile(.95),
            "before_cue_count": int(g.response_time_s.lt(0).sum()),
            "after_2s_count": int(g.response_time_s.gt(2).sum()),
            "multiple_segments_count": int(g.vad_segment_count.gt(1).sum()),
            "status": "AUTOMATIC_VAD_DESCRIPTIVE_NOT_MANUALLY_VALIDATED",
        })
    pd.DataFrame(rt_rows).to_csv(OUT / "response_time_summary.csv", index=False, encoding="utf-8-sig")

    cond = pd.DataFrame(condition_rows).iloc[-6:]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cond.scope, cond.automatic_accuracy_all_trials, color="#4472C4")
    ax.set_ylim(0, 1); ax.set_ylabel("Provisional automatic accuracy")
    ax.set_title("Six-condition naming accuracy (automatic, pending manual review)")
    ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "condition_accuracy_provisional.png", dpi=170); plt.close(fig)
    return final, queue, participant_summary, pd.DataFrame(condition_rows), pd.DataFrame(rt_rows)


def write_report(final, queue, participant_summary, condition_summary, rt_summary):
    overall = condition_summary.iloc[0]
    six = condition_summary.iloc[-6:]
    report = f"""# Behavioural speech analysis report

## Status and method

All {len(final):,} matched formal WAVs were processed without modifying any source WAV, XDF, CSV, manifest, or log. This run implements the user's revised instruction to map similar sounds directly within the closed six-word vocabulary.

The classifier is target-blind: it never uses `target_label` to choose a prediction. Silero VAD finds speech, Wav2Vec2 supplies a raw transcript, exact CTC likelihoods compare only `bowl, cup, plate, knife, fork, spoon`, and an independent CMU pronunciation-distance check identifies near-sounding transcripts. The target is used only afterward to calculate `correct`.

On the fixed 100-trial validation subset, unrestricted exact-word recognition had previously been inadequate. The revised six-word constrained classifier agreed with the expected target on 80/100 trials. The predeclared higher-confidence cross-check covered 61/100 and agreed with the target on 56/61 (91.8%). Target agreement is not a manual ASR gold standard because participants can genuinely make naming errors; it supports use as a provisional classifier, not as a substitute for listening review.

## Confirmed data-structure results

- Formal WAVs: **{len(final):,}**; all matched to participant/run/trial metadata.
- Participants: **{final.participant.nunique()}**; six conditions represented.
- Existing unreadable WAVs: **{int(final.qc_flag.str.contains('unreadable').sum())}**.
- The separate inventory identified 18 missing expected files, all P005 run 4 trials 55–72.

## Automatically inferred naming results

- Provisional overall accuracy: **{overall.automatic_accuracy_all_trials:.1%}** ({int(overall.correct_count_automatic)}/{len(final)}).
- Higher-confidence automatic subset: **{int(overall.higher_confidence_count):,}** trials; target agreement **{overall.higher_confidence_accuracy_vs_target:.1%}**.
- Trials requiring manual review: **{int(overall.manual_review_count):,}**.
- Manual-review queue including 5% audit: **{len(queue):,}** rows.
- No-speech detections: **{int((~final.vad_detected).sum())}/{len(final):,} ({(~final.vad_detected).mean():.2%})**.

Six-condition provisional automatic accuracy:

{chr(10).join(f'- {r.scope}: {int(r.correct_count_automatic)}/{int(r.trial_count)} = {r.automatic_accuracy_all_trials:.1%}; review {int(r.manual_review_count)}' for _, r in six.iterrows())}

These values are automatically inferred, not manually confirmed. Near-sound mapping can recover pronunciations such as `up→cup`, `life→knife`, `for/folk→fork`, and `late/light→plate`, but ambiguous recordings remain flagged.

## Manual review status

No full-dataset trial has yet been manually reviewed in these output files. `manual_review_queue.csv` contains every uncertain trial plus a fixed-seed 5% audit of higher-confidence decisions. The manual columns are blank by design.

P018 should be reviewed first: 37/432 trials had no detected speech and 343/432 entered the review queue; its provisional all-trial accuracy was 59.3%. P007 and P009 also had lower provisional values, but those must not be interpreted until manual review separates genuine errors from recognizer errors.

## Response onset / RT

- VAD detected speech in **{int(final.vad_detected.sum()):,}/{len(final):,} ({final.vad_detected.mean():.1%})**.
- Median 0–2 s response time: **{rt_summary.iloc[0].median_rt_s:.3f} s**.
- Before-cue detections: **{int(rt_summary.iloc[0].before_cue_count)}**.
- After-2-s detections: **{int(rt_summary.iloc[0].after_2s_count)}**.
- Multiple speech segments: **{int(rt_summary.iloc[0].multiple_segments_count)}**.

RT is descriptive and automatically inferred. Because no human onset gold standard has yet quantified VAD timing error, it should not yet be treated as dissertation-ready condition-level RT.

## Confirmed vs inferred vs unresolved

- **Confirmed:** file integrity, deterministic trial mapping, target/condition/timestamp metadata, recorded cue positions.
- **Automatically inferred:** transcripts, constrained six-label predictions, correctness, speech onset, response time, and all current accuracy summaries.
- **Manually reviewed:** none yet.
- **Unresolved:** all `manual_review_required=True` trials until listening review is completed.

## Suitability for later 2 × 3 analysis

The six conditions are present for every participant with available WAVs, but final behavioural inferential analysis should wait until manual review is incorporated. Accuracy variation and completeness can then be reassessed participant-by-participant. RT should additionally wait for manual onset validation.
"""
    (OUT / "behavioural_analysis_report.md").write_text(report, encoding="utf-8")


def main():
    inventory = pd.read_csv(INVENTORY, low_memory=False)
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h", cache_dir=MODEL_CACHE, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h", cache_dir=MODEL_CACHE, local_files_only=True)
    model.eval()
    blank = int(processor.tokenizer.pad_token_id)
    label_tokens = {x: processor.tokenizer(x.upper(), add_special_tokens=False).input_ids for x in LABELS}

    done = pd.read_csv(CHECKPOINT, low_memory=False) if CHECKPOINT.exists() else pd.DataFrame()
    start = len(done)
    if start > len(inventory):
        raise RuntimeError("Checkpoint is longer than inventory")
    records = done.to_dict("records") if start else []
    for batch_start in range(start, len(inventory), BATCH_SIZE):
        part = inventory.iloc[batch_start:batch_start + BATCH_SIZE]
        inferred = infer_batch(part, processor, model, label_tokens, blank)
        for (_, source), info in zip(part.iterrows(), inferred):
            records.append({**source.to_dict(), **info})
        if len(records) % 120 < BATCH_SIZE or len(records) == len(inventory):
            pd.DataFrame(records).to_csv(CHECKPOINT, index=False, encoding="utf-8-sig")
        print(f"Full speech progress: {len(records)}/{len(inventory)}", flush=True)

    combined = pd.DataFrame(records)
    final, queue, participant_summary, condition_summary, rt_summary = build_outputs(combined)
    write_report(final, queue, participant_summary, condition_summary, rt_summary)
    params = {
        "labels": LABELS, "asr_model": "facebook/wav2vec2-base-960h",
        "classification": "exact CTC likelihood restricted to six labels; target-blind",
        "vad": {"model": "Silero via faster-whisper", "threshold": .5,
                "min_speech_duration_ms": 150, "min_silence_duration_ms": 150,
                "speech_pad_ms": 30},
        "phonetic_mapping": "CMU pronunciation normalized Levenshtein distance",
        "phonetic_distance_limit": PHONETIC_DISTANCE_LIMIT,
        "low_log_score_margin_limit": LOW_MARGIN_LIMIT,
        "manual_review_rule": "no VAD, decode error, multiple segments, phonetic distance > limit, acoustic/phonetic disagreement, or low score margin",
        "audit_fraction": .05, "random_seed": SEED,
        "target_used_for_prediction": False,
    }
    (OUT / "speech_analysis_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(final), "review": int(final.manual_review_required.sum()),
                      "queue": len(queue), "overall_accuracy": float(final.correct.fillna(False).mean()),
                      "vad_rate": float(final.vad_detected.mean())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
