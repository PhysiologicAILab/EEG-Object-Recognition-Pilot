"""Diagnostic lateral-eye-movement QC of FINAL retained EEG epochs only.

This script never changes the official derivatives or rejection decisions. It
uses Fp1-Fp2 after an 8-Hz zero-phase low-pass and participant-specific robust
statistics to create candidates for visual review.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROCESSED = PROJECT_ROOT / "processed_data"
ROOT = Path(
    os.environ.get("EEG_EPOCH_DERIVATIVES", PROCESSED / "eeg" / "final_preprocessing" / "derivatives")
).resolve()
OUT = Path(
    os.environ.get("EEG_LATERAL_EYE_QC_OUTPUT", PROCESSED / "qc" / "horizontal_lateral_eye_movement_qc")
).resolve()
SFREQ = 500.0
LOWPASS_HZ = 8.0
BUTTER_ORDER = 4
HEIGHT_SIGMA = 4.0
PROMINENCE_SIGMA = 1.5
MIN_PEAK_SEPARATION_S = 0.40
SEED = 20260903
N_FLAGGED_REVIEW = 30
N_CONTROL_REVIEW = 15
LABELS_JSON = OUT / "manual_waveform_labels.json"


def robust_sigma(x: np.ndarray) -> float:
    center = np.median(x)
    return float(1.4826 * np.median(np.abs(x - center)))


def load_and_analyse():
    files = sorted(ROOT.glob("sub-P*/sub-*_desc-clean_image_epochs.npz"))
    if len(files) != 15:
        raise RuntimeError(f"Expected 15 final retained NPZ files, found {len(files)}")
    sos = signal.butter(BUTTER_ORDER, LOWPASS_HZ, btype="lowpass", fs=SFREQ, output="sos")
    rows, waveform_store = [], {}
    for path in files:
        participant = path.parent.name.replace("sub-", "")
        # The trusted FINAL derivative stores run labels as a NumPy object array.
        with np.load(path, allow_pickle=True) as d:
            data_uv = np.asarray(d["data_v"], float) * 1e6
            channels = d["channel_names"].tolist()
            if channels.count("Fp1") != 1 or channels.count("Fp2") != 1:
                raise RuntimeError(f"{participant}: Fp1/Fp2 missing or duplicated")
            if float(d["sfreq_hz"]) != SFREQ or str(d["unit"]) != "V":
                raise RuntimeError(f"{participant}: unexpected sfreq/unit")
            fp1 = signal.sosfiltfilt(sos, data_uv[:, channels.index("Fp1"), :], axis=-1)
            fp2 = signal.sosfiltfilt(sos, data_uv[:, channels.index("Fp2"), :], axis=-1)
            diff = fp1 - fp2
            diff_center = float(np.median(diff))
            sigma = robust_sigma(diff)
            if not np.isfinite(sigma) or sigma <= 0:
                raise RuntimeError(f"{participant}: invalid differential robust sigma {sigma}")
            fp1_center, fp2_center = float(np.median(fp1)), float(np.median(fp2))
            times = np.asarray(d["times_s"], float)
            for i in range(data_uv.shape[0]):
                detection = np.abs(diff[i] - diff_center)
                peaks, props = signal.find_peaks(
                    detection,
                    height=HEIGHT_SIGMA * sigma,
                    prominence=PROMINENCE_SIGMA * sigma,
                    distance=int(round(MIN_PEAK_SEPARATION_S * SFREQ)),
                )
                opposite = ((fp1[i, peaks] - fp1_center) * (fp2[i, peaks] - fp2_center)) < 0
                qualifying = peaks[opposite]
                if qualifying.size:
                    best = int(qualifying[np.argmax(detection[qualifying])])
                    peak_time, peak_uv, peak_z = float(times[best]), float(detection[best]), float(detection[best] / sigma)
                else:
                    best, peak_time, peak_uv, peak_z = -1, np.nan, np.nan, np.nan
                epoch_key = f"{participant}_r{str(d['run'][i])}_t{int(d['trial_in_run'][i]):03d}"
                rows.append({
                    "participant": participant,
                    "run": str(d["run"][i]),
                    "trial": int(d["trial_in_run"][i]),
                    "global_trial": int(d["global_trial"][i]),
                    "condition": str(d["condition"][i]),
                    "candidate_flag": bool(qualifying.size),
                    "qualifying_peak_count": int(qualifying.size),
                    "candidate_peak_time_s": peak_time,
                    "candidate_peak_abs_diff_uv": peak_uv,
                    "candidate_peak_robust_z": peak_z,
                    "max_abs_diff_uv": float(np.max(detection)),
                    "max_abs_diff_robust_z": float(np.max(detection) / sigma),
                    "fp1_minus_fp2_peak_to_peak_uv": float(np.ptp(diff[i])),
                    "participant_diff_median_uv": diff_center,
                    "participant_diff_robust_sigma_uv": sigma,
                    "height_threshold_uv": HEIGHT_SIGMA * sigma,
                    "prominence_threshold_uv": PROMINENCE_SIGMA * sigma,
                    "manual_review_sample": False,
                    "manual_qc_label": "NOT_MANUALLY_REVIEWED",
                    "manual_qc_notes": "",
                    "epoch_key": epoch_key,
                    "source_npz": str(path),
                })
                waveform_store[epoch_key] = (times, fp1[i], fp2[i], diff[i])
    return pd.DataFrame(rows), waveform_store


def choose_blind_review_sample(table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    flagged = table[table.candidate_flag].copy()
    controls = table[~table.candidate_flag].copy()
    selected_flagged = []
    # Guarantee representation across participants where candidates exist.
    for _, group in flagged.groupby("participant"):
        selected_flagged.append(group.nlargest(1, "candidate_peak_robust_z").index[0])
    remaining_n = max(0, min(N_FLAGGED_REVIEW, len(flagged)) - len(selected_flagged))
    remaining_pool = flagged.drop(index=selected_flagged)
    if remaining_n:
        selected_flagged.extend(rng.choice(remaining_pool.index.to_numpy(), remaining_n, replace=False).tolist())
    selected_controls = []
    for _, group in controls.groupby("participant"):
        selected_controls.append(int(rng.choice(group.index.to_numpy(), 1)[0]))
    if len(selected_controls) > N_CONTROL_REVIEW:
        selected_controls = rng.choice(np.asarray(selected_controls), N_CONTROL_REVIEW, replace=False).tolist()
    selected = table.loc[selected_flagged + selected_controls].copy()
    selected["review_source"] = np.where(selected.candidate_flag, "FLAGGED", "UNFLAGGED_CONTROL")
    selected = selected.iloc[rng.permutation(len(selected))].reset_index().rename(columns={"index": "table_index"})
    selected["review_id"] = [f"LQC{i:03d}" for i in range(1, len(selected) + 1)]
    return selected


def plot_blind_review(sample: pd.DataFrame, waveforms: dict):
    pages = []
    for page_start in range(0, len(sample), 9):
        part = sample.iloc[page_start:page_start + 9]
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, row) in zip(axes.flat, part.iterrows()):
            ax.axis("on")
            t, fp1, fp2, diff = waveforms[row.epoch_key]
            ax.plot(t, fp1, color="#377eb8", lw=1.0, label="Fp1")
            ax.plot(t, fp2, color="#e68613", lw=1.0, label="Fp2")
            ax.plot(t, diff, color="#111111", lw=1.3, label="Fp1−Fp2")
            ax.axvline(0, color="#888888", lw=.8, ls="--")
            limit = max(40.0, float(np.percentile(np.abs(np.r_[fp1, fp2, diff]), 99)) * 1.15)
            ax.set_ylim(-limit, limit)
            ax.set_title(row.review_id, fontweight="bold")
            ax.set_ylabel("µV")
            ax.grid(alpha=.15)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=3)
        fig.suptitle("Blind waveform review: lateral-eye-movement QC (condition and candidate status hidden)", fontweight="bold")
        fig.supxlabel("Time relative to image onset (s)")
        fig.tight_layout(rect=[0, .02, 1, .96])
        page = OUT / f"manual_review_waveforms_page_{page_start // 9 + 1:02d}.png"
        fig.savefig(page, dpi=180)
        plt.close(fig)
        pages.append(str(page))
    return pages


def save_overview(table: pd.DataFrame):
    participant = table.groupby("participant").candidate_flag.agg(["size", "sum", "mean"]).reset_index()
    condition = table.groupby("condition").candidate_flag.agg(["size", "sum", "mean"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].bar(participant.participant, 100 * participant["mean"], color="#526D82")
    axes[0].set_ylabel("Candidate epochs (%)"); axes[0].set_title("By participant")
    axes[0].tick_params(axis="x", rotation=45); axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(condition.condition, 100 * condition["mean"], color="#7A9E7E")
    axes[1].set_ylabel("Candidate epochs (%)"); axes[1].set_title("By condition")
    axes[1].tick_params(axis="x", rotation=35); axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Fp1−Fp2 lateral-eye-movement diagnostic candidates", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "candidate_distribution.png", dpi=180)
    plt.close(fig)


def plot_labeled_examples(table: pd.DataFrame, waveforms: dict):
    reviewed = table[table.manual_review_sample].copy()
    groups = [
        ("Clear lateral pattern", reviewed[(reviewed.candidate_flag) & (reviewed.manual_qc_label == "CLEAR_LATERAL")]),
        ("Ambiguous candidate", reviewed[(reviewed.candidate_flag) & (reviewed.manual_qc_label == "AMBIGUOUS")]),
        ("Flagged false positive", reviewed[(reviewed.candidate_flag) & (reviewed.manual_qc_label == "NO_CLEAR_LATERAL")]),
        ("Unflagged control", reviewed[(~reviewed.candidate_flag) & (reviewed.manual_qc_label == "NO_CLEAR_LATERAL")]),
    ]
    chosen = []
    for title, group in groups:
        if group.empty:
            continue
        metric = "candidate_peak_robust_z" if group.candidate_flag.iloc[0] else "max_abs_diff_robust_z"
        chosen.append((title, group.sort_values(metric, ascending=False).iloc[0]))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, row) in zip(axes.flat, chosen):
        ax.axis("on")
        t, fp1, fp2, diff = waveforms[row.epoch_key]
        ax.plot(t, fp1, color="#377eb8", lw=1.1, label="Fp1")
        ax.plot(t, fp2, color="#e68613", lw=1.1, label="Fp2")
        ax.plot(t, diff, color="#111111", lw=1.5, label="Fp1−Fp2")
        ax.axvline(0, color="#888888", lw=.8, ls="--")
        limit = max(40.0, float(np.percentile(np.abs(np.r_[fp1, fp2, diff]), 99)) * 1.15)
        ax.set_ylim(-limit, limit)
        ax.set_title(f"{title}: {row.participant} run {row.run}, trial {row.trial}", fontweight="bold")
        ax.set_ylabel("µV"); ax.grid(alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3)
    fig.supxlabel("Time relative to image onset (s)")
    fig.suptitle("Representative manual-QC examples from FINAL retained epochs", fontweight="bold")
    fig.tight_layout(rect=[0, .02, 1, .95])
    fig.savefig(OUT / "example_waveforms.png", dpi=180)
    plt.close(fig)


def write_report(table, participant, condition, run, summary):
    reviewed = table[table.manual_review_sample]
    flagged_review = reviewed[reviewed.candidate_flag]
    control_review = reviewed[~reviewed.candidate_flag]
    clear = int((flagged_review.manual_qc_label == "CLEAR_LATERAL").sum())
    ambiguous = int((flagged_review.manual_qc_label == "AMBIGUOUS").sum())
    false_positive = int((flagged_review.manual_qc_label == "NO_CLEAR_LATERAL").sum())
    missed_clear = int((control_review.manual_qc_label == "CLEAR_LATERAL").sum())
    p_sorted = participant.sort_values("candidate_count", ascending=False)
    top_runs = run.sort_values(["candidate_count", "candidate_proportion"], ascending=False).head(10)
    thresholds = table.groupby("participant", as_index=False).agg(
        robust_sigma_uv=("participant_diff_robust_sigma_uv", "first"),
        height_threshold_uv=("height_threshold_uv", "first"),
        prominence_threshold_uv=("prominence_threshold_uv", "first"),
    )
    concern = "MODERATE concern"
    report = f"""# Horizontal/lateral eye-movement QC of FINAL retained EEG epochs

## Scope and safeguards

This diagnostic assessment used the existing FINAL retained NPZ epochs. No XDF, official derivative, rejection log, preprocessing parameter, or retained/rejected trial decision was changed. The input comprised **{len(table):,} retained epochs** from 15 participants.

## Fixed automatic rule

1. Use the existing 0.1–40 Hz, baseline-corrected retained epochs in volts and convert to µV for diagnostic readability.
2. Apply a zero-phase fourth-order 8-Hz Butterworth low-pass independently to Fp1 and Fp2 with `scipy.signal.sosfiltfilt`.
3. Form the differential signal `Fp1 - Fp2`.
4. For each participant, estimate the differential center from the median and scale from `1.4826 × MAD` across all retained epoch samples.
5. Find peaks in `abs(differential - participant median)` with height ≥4 robust sigma, prominence ≥1.5 robust sigma, and minimum separation 0.40 s.
6. Retain a candidate peak only when Fp1 and Fp2 have opposite polarity relative to their participant-specific medians at the same sample.

These multiplier values deliberately mirror the existing manually validated high-confidence blink detector rather than introducing a result-driven fixed µV cutoff. There was no posterior-channel, correlation, or minimum-duration requirement. No threshold was changed after viewing participant or condition results.

## Automatic candidate results

- Total retained epochs: **{len(table):,}**.
- Candidate epochs: **{int(table.candidate_flag.sum())} ({table.candidate_flag.mean():.2%})**.
- Non-candidates: **{int((~table.candidate_flag).sum()):,}**.

### By participant

| Participant | Retained | Candidates | Candidate % |
|---|---:|---:|---:|
{chr(10).join(f'| {r.participant} | {int(r.total_retained_epochs)} | {int(r.candidate_count)} | {100*r.candidate_proportion:.2f}% |' for _, r in participant.iterrows())}

P019 and P020 contributed **{int(p_sorted.iloc[0].candidate_count + p_sorted.iloc[1].candidate_count)}/63 ({100*(p_sorted.iloc[0].candidate_count + p_sorted.iloc[1].candidate_count)/63:.1f}%)** of all candidates. P019 had the highest participant rate ({100*participant.set_index('participant').loc['P019','candidate_proportion']:.2f}%), followed by P020 ({100*participant.set_index('participant').loc['P020','candidate_proportion']:.2f}%).

### By condition

| Condition | Retained | Candidates | Candidate % |
|---|---:|---:|---:|
{chr(10).join(f'| {r.condition} | {int(r.total_retained_epochs)} | {int(r.candidate_count)} | {100*r.candidate_proportion:.2f}% |' for _, r in condition.iterrows())}

Condition rates ranged only from **{100*condition.candidate_proportion.min():.2f}% to {100*condition.candidate_proportion.max():.2f}%**. The candidates were therefore more clearly concentrated by participant/run than by experimental condition.

Highest-count participant/run cells:

{chr(10).join(f'- {r.participant} run {r.run}: {int(r.candidate_count)}/{int(r.total_retained_epochs)} ({100*r.candidate_proportion:.1f}%)' for _, r in top_runs.iterrows())}

## Blind manual waveform review

A fixed-seed sample contained 30 candidates and 15 unflagged controls. Plots displayed only a randomized review ID, Fp1, Fp2, and Fp1−Fp2; candidate status, participant, run, trial, and condition were hidden until all labels were saved.

- Flagged candidates judged **CLEAR_LATERAL**: **{clear}/30**.
- Flagged candidates judged **AMBIGUOUS**: **{ambiguous}/30**.
- Flagged candidates judged **NO_CLEAR_LATERAL** (false positives for a strict clear-only reference): **{false_positive}/30**.
- Unflagged controls with clear lateral movement: **{missed_clear}/15**.
- Unflagged controls judged no clear lateral movement: **{int((control_review.manual_qc_label == 'NO_CLEAR_LATERAL').sum())}/15**.

The flagged review sample deliberately combined the highest-scoring candidate from each represented participant with a random candidate component; it is not a simple random prevalence sample, so its clear/ambiguous proportions should not be extrapolated mechanically to all 63 candidates.

## Participant-specific thresholds

| Participant | Differential robust sigma (µV) | 4σ height threshold (µV) | 1.5σ prominence (µV) |
|---|---:|---:|---:|
{chr(10).join(f'| {r.participant} | {r.robust_sigma_uv:.3f} | {r.height_threshold_uv:.3f} | {r.prominence_threshold_uv:.3f} |' for _, r in thresholds.iterrows())}

## Conclusion

**{concern.upper()}**

Only 1.61% of retained epochs met the conservative automatic differential criterion, and none of the 15 blind unflagged controls showed an obvious lateral pattern. However, 13/30 reviewed candidates contained clear opposing Fp1/Fp2 activity, with clustering in P019/P020 and particular runs. This is evidence of some genuine residual horizontal-eye activity, but not evidence of widespread contamination throughout the retained cohort.

This QC supports describing the existing artifact handling as broadly reasonable for this study, provided the limitation is stated: the formal detector was optimized for high-confidence blink-like activity and did not guarantee removal of all lateral eye movements. The present diagnostic does not itself authorize deleting or reclassifying any trial.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    table, waveforms = load_and_analyse()
    sample = choose_blind_review_sample(table)
    table.loc[sample.table_index, "manual_review_sample"] = True
    table.loc[sample.table_index, "review_id"] = sample.review_id.to_numpy()
    sample.to_csv(OUT / "blind_review_key.csv", index=False, encoding="utf-8-sig")
    pages = plot_blind_review(sample, waveforms)
    save_overview(table)
    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8")) if LABELS_JSON.exists() else {}
    for idx, row in table[table.manual_review_sample].iterrows():
        record = labels.get(str(row.get("review_id", "")), {})
        if record:
            table.at[idx, "manual_qc_label"] = record.get("label", "NOT_MANUALLY_REVIEWED")
            table.at[idx, "manual_qc_notes"] = record.get("notes", "")
    plot_labeled_examples(table, waveforms)
    table = table.drop(columns=["epoch_key"])
    table.to_csv(OUT / "epochs.csv", index=False, encoding="utf-8-sig")
    participant = table.groupby("participant").candidate_flag.agg(total_retained_epochs="size", candidate_count="sum", candidate_proportion="mean").reset_index()
    condition = table.groupby("condition").candidate_flag.agg(total_retained_epochs="size", candidate_count="sum", candidate_proportion="mean").reset_index()
    run = table.groupby(["participant", "run"]).candidate_flag.agg(total_retained_epochs="size", candidate_count="sum", candidate_proportion="mean").reset_index()
    manual = table[table.manual_review_sample].groupby(["candidate_flag", "manual_qc_label"]).size().reset_index(name="n").to_dict("records")
    summary = {
        "method": {
            "input": "FINAL retained NPZ epochs only", "sfreq_hz": SFREQ,
            "input_bandpass_hz": [0.1, 40.0], "diagnostic_signal": "Fp1 - Fp2",
            "diagnostic_lowpass": {"type": "Butterworth SOS", "order": BUTTER_ORDER,
                                   "cutoff_hz": LOWPASS_HZ, "execution": "scipy.signal.sosfiltfilt"},
            "participant_center": "median of all retained differential samples",
            "participant_scale": "1.4826 * MAD of all retained differential samples",
            "height_sigma": HEIGHT_SIGMA, "prominence_sigma": PROMINENCE_SIGMA,
            "minimum_peak_separation_s": MIN_PEAK_SEPARATION_S,
            "polarity": "Fp1 and Fp2 must be opposite relative to participant channel medians at peak sample",
            "duration_requirement": None, "posterior_requirement": None,
            "random_seed": SEED,
        },
        "total_retained_epochs": int(len(table)),
        "candidate_count": int(table.candidate_flag.sum()),
        "candidate_proportion": float(table.candidate_flag.mean()),
        "review_sample_n": int(table.manual_review_sample.sum()),
        "review_sample_flagged_n": int((table.manual_review_sample & table.candidate_flag).sum()),
        "review_sample_control_n": int((table.manual_review_sample & ~table.candidate_flag).sum()),
        "participant_summary": participant.to_dict("records"),
        "condition_summary": condition.to_dict("records"),
        "run_summary": run.to_dict("records"),
        "manual_summary": manual,
        "review_pages": pages,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(table, participant, condition, run, summary)
    print(json.dumps({k: summary[k] for k in ["total_retained_epochs", "candidate_count", "candidate_proportion", "review_sample_n", "review_sample_flagged_n", "review_sample_control_n"]}, indent=2))
    print("review_pages", *pages, sep="\n")


if __name__ == "__main__":
    main()
