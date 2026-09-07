"""Supplementary ICA diagnostic for P007, P008, and P013 only.

This reproduces the original P007 ICA diagnostic. It reads source XDF files
without modifying them, applies ICA only in memory, and never writes a cleaned
EEG dataset.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Numba is not used by this diagnostic, and importing it is pathologically slow
# in this Windows environment. This changes startup time, not ICA/filter results.
os.environ.setdefault("MNE_USE_NUMBA", "false")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", tempfile.mkdtemp(prefix="supp_ica_mne_"))
import mne
import numpy as np
import pandas as pd
import pyxdf
from scipy import signal


PARTICIPANTS = ["P007", "P008", "P013"]
CHANNELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
FRONTAL = ["Fp1", "Fp2"]
POSTERIOR = ["P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
MICROVOLT_UNITS = {"µv", "uv", "µvolt", "microvolt", "microvolts"}
SFREQ = 500.0


def one(value, default=""):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return default if value is None else value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def channel_metadata(eeg):
    count = int(float(one(eeg["info"].get("channel_count", [0]))))
    try:
        channels = eeg["info"]["desc"][0]["channels"][0]["channel"]
    except Exception:
        channels = []
    rows = []
    for index in range(count):
        item = channels[index] if index < len(channels) else {}
        rows.append({
            "index": index,
            "name": str(one(item.get("label", [f"Ch{index + 1}"]))),
            "type": str(one(item.get("type", [""]))),
            "unit": str(one(item.get("unit", [""]))),
        })
    return rows


def load_xdf_as_raw(path: Path):
    """Reproduce the verified import state in memory."""
    before_hash = sha256(path)
    streams, _ = pyxdf.load_xdf(str(path), verbose=False)
    eegs = [
        stream for stream in streams
        if str(one(stream["info"].get("type", [""]))).lower() == "eeg"
        and "saga" in str(one(stream["info"].get("name", [""]))).lower()
    ]
    if len(eegs) != 1:
        raise RuntimeError(f"{path}: expected one SAGA EEG stream, found {len(eegs)}")
    eeg = eegs[0]
    metadata = channel_metadata(eeg)
    selected = [
        row for row in metadata
        if row["type"].strip().lower() == "eeg"
        and row["name"].upper() not in {"TRIGGERS", "STATUS", "COUNTER"}
    ]
    original_names = [row["name"] for row in selected]
    final_names = ["Fp1" if name == "Fpz" else name for name in original_names]
    if final_names != CHANNELS:
        raise RuntimeError(f"{path}: unexpected EEG channels {final_names}")
    units = {row["unit"].strip().lower() for row in selected}
    if not units or not units.issubset(MICROVOLT_UNITS):
        raise RuntimeError(f"{path}: EEG units not explicitly microvolts: {units}")
    sfreq = float(one(eeg["info"].get("nominal_srate", [0])))
    if sfreq != SFREQ:
        raise RuntimeError(f"{path}: expected {SFREQ} Hz, got {sfreq}")
    data_v = np.asarray(
        eeg["time_series"][:, [row["index"] for row in selected]].T,
        dtype=np.float64,
    ) * 1e-6
    info = mne.create_info(CHANNELS, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_v, info, verbose=False)
    raw.set_montage("standard_1020", on_missing="raise", verbose=False)
    del streams, eeg, data_v
    gc.collect()
    after_hash = sha256(path)
    if before_hash != after_hash:
        raise RuntimeError(f"Source XDF changed during read-only processing: {path}")
    return raw, {
        "xdf_path": str(path),
        "xdf_sha256": before_hash,
        "source_hash_unchanged": True,
        "original_channels": original_names,
        "final_channels": final_names,
        "xdf_units": sorted(units),
        "conversion": "microvolts multiplied by 1e-6 once to create in-memory volts",
        "sfreq_hz": sfreq,
    }


def robust_sigma(x):
    median = np.median(x)
    return float(1.4826 * np.median(np.abs(x - median)))


def detect_blinks(raw):
    """Synchronized frontal-event detector used in the P007 diagnostic."""
    sfreq = float(raw.info["sfreq"])
    front = raw.get_data(picks=FRONTAL) * 1e6
    common = np.mean(front, axis=0)
    sos = signal.butter(4, 8.0, btype="lowpass", fs=sfreq, output="sos")
    smooth = signal.sosfiltfilt(sos, common)
    center = float(np.median(smooth))
    scale = robust_sigma(smooth)
    detection = np.abs(smooth - center)
    peaks, _ = signal.find_peaks(
        detection,
        height=4.0 * scale,
        prominence=1.5 * scale,
        distance=int(round(0.40 * sfreq)),
    )
    edge = int(round(20 * sfreq))
    peaks = peaks[(peaks >= edge) & (peaks < detection.size - edge)]
    fp1_smooth = signal.sosfiltfilt(sos, front[0])
    fp2_smooth = signal.sosfiltfilt(sos, front[1])
    fp1_center, fp2_center = np.median(fp1_smooth), np.median(fp2_smooth)
    keep = (fp1_smooth[peaks] - fp1_center) * (fp2_smooth[peaks] - fp2_center) > 0
    return peaks[keep], {"robust_sigma_uv": scale, "threshold_uv": 4.0 * scale}


def make_event_epochs(data, event_samples, sfreq, tmin=-0.5, tmax=0.8, decim=1):
    before = int(round(-tmin * sfreq))
    after = int(round(tmax * sfreq))
    offsets = np.arange(-before, after + 1, decim, dtype=int)
    valid = event_samples[(event_samples - before >= 0) & (event_samples + after < data.shape[1])]
    epochs = np.stack([data[:, event + offsets] for event in valid], axis=0)
    times = offsets / sfreq
    baseline = (times >= -0.5) & (times <= -0.2)
    epochs = epochs - np.mean(epochs[:, :, baseline], axis=2, keepdims=True)
    return epochs, times


def component_metrics(ica, fit, event_samples):
    """Multimetric component classification used in the P007 diagnostic."""
    sfreq = float(fit.info["sfreq"])
    decim = 5
    sources = ica.get_sources(fit).get_data()[:, ::decim]
    sensor_front = fit.get_data(picks=FRONTAL)[:, ::decim]
    source_z = sources / np.std(sources, axis=1, keepdims=True)
    frontal_corrs = np.array([
        [np.corrcoef(source_z[i], sensor_front[j])[0, 1] for j in range(2)]
        for i in range(sources.shape[0])
    ])
    event_decim = np.rint(event_samples / decim).astype(int)
    source_epochs, source_times = make_event_epochs(
        source_z, event_decim, sfreq / decim, tmin=-0.5, tmax=0.8, decim=1
    )
    source_average = np.mean(source_epochs, axis=0)
    response = (source_times >= -0.1) & (source_times <= 0.6)
    event_peak_z = np.max(np.abs(source_average[:, response]), axis=1)

    freq, psd = signal.welch(
        sources, fs=sfreq / decim, window="hann",
        nperseg=int(20 * sfreq / decim), noverlap=int(10 * sfreq / decim), axis=1,
    )

    def band(power, low, high):
        mask = (freq >= low) & (freq <= high)
        return np.trapezoid(power[:, mask], freq[mask], axis=1)

    slow_ratio = band(psd, 1, 4) / band(psd, 1, 40)
    topographies = ica.get_components()
    index = {name: fit.ch_names.index(name) for name in fit.ch_names}
    frontal_idx = [index[name] for name in FRONTAL]
    posterior_idx = [index[name] for name in POSTERIOR]
    topo_ratio = (
        np.mean(np.abs(topographies[frontal_idx]), axis=0)
        / (np.mean(np.abs(topographies[posterior_idx]), axis=0) + 1e-30)
    )

    rows = []
    for component in range(sources.shape[0]):
        max_corr = float(np.max(np.abs(frontal_corrs[component])))
        evidence = {
            "frontal_correlation": max_corr >= 0.45,
            "blink_locked": event_peak_z[component] >= 0.35,
            "frontal_topography": topo_ratio[component] >= 1.5,
            "slow_spectrum": slow_ratio[component] >= 0.30,
        }
        strong = all(evidence.values())
        possible = sum(evidence.values()) >= 2 or max_corr >= 0.60
        classification = "STRONG OCULAR CANDIDATE" if strong else ("POSSIBLE OCULAR" if possible else "NOT OCULAR")
        rows.append({
            "component": component,
            "classification": classification,
            "correlation_with_Fp1": float(frontal_corrs[component, 0]),
            "correlation_with_Fp2": float(frontal_corrs[component, 1]),
            "maximum_absolute_frontal_correlation": max_corr,
            "blink_locked_average_peak_z": float(event_peak_z[component]),
            "frontal_to_posterior_topography_ratio": float(topo_ratio[component]),
            "slow_1_4_to_1_40_power_ratio": float(slow_ratio[component]),
            "evidence_count": int(sum(evidence.values())),
            "evidence": "; ".join(key for key, value in evidence.items() if value) or "none",
        })
    table = pd.DataFrame(rows)
    strong = table[table.classification == "STRONG OCULAR CANDIDATE"].copy()
    if len(strong) > 2:
        score = (
            strong.maximum_absolute_frontal_correlation
            * strong.blink_locked_average_peak_z
            * strong.frontal_to_posterior_topography_ratio
        )
        keep_components = list(strong.loc[score.nlargest(2).index, "component"])
        downgrade = (table.classification == "STRONG OCULAR CANDIDATE") & ~table.component.isin(keep_components)
        table.loc[downgrade, "classification"] = "POSSIBLE OCULAR"
    return table


def posterior_psd_and_power(data_uv, sfreq):
    """PSD settings and posterior-median definition used in the P007 diagnostic."""
    nperseg = int(round(30 * sfreq))
    freq, psd = signal.welch(
        data_uv, fs=sfreq, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, detrend="constant", axis=1,
    )
    posterior_idx = [CHANNELS.index(name) for name in POSTERIOR]
    posterior_curve = np.median(psd[posterior_idx], axis=0)
    mask = (freq >= 1.0) & (freq <= 40.0)
    power = float(np.trapezoid(posterior_curve[mask], freq[mask]))
    return freq, posterior_curve, power


def process_participant(participant, path):
    print(f"[{participant}] load XDF", flush=True)
    raw0, provenance = load_xdf_as_raw(path)
    if raw0.ch_names != CHANNELS or raw0.info["bads"] or raw0.info["projs"]:
        raise RuntimeError(f"{participant}: unexpected initial Raw state")
    print(f"[{participant}] filter main 0.1-40 Hz and ICA-fit 1-40 Hz", flush=True)
    main_data = raw0.copy().filter(
        l_freq=0.1, h_freq=40.0, method="fir", phase="zero",
        fir_window="hamming", fir_design="firwin", verbose=False,
    )
    ica_fit = raw0.copy().filter(
        l_freq=1.0, h_freq=40.0, method="fir", phase="zero",
        fir_window="hamming", fir_design="firwin", verbose=False,
    )
    del raw0
    gc.collect()

    blink_samples, blink_info = detect_blinks(main_data)
    if len(blink_samples) == 0:
        raise RuntimeError(f"{participant}: no frontal events for the established P007 classifier")
    rank = int(mne.compute_rank(ica_fit, rank=None, tol="auto", verbose=False)["eeg"])
    n_components = min(rank, len(CHANNELS) - 1)
    ica = mne.preprocessing.ICA(
        n_components=n_components, method="fastica", random_state=97, max_iter=1000,
        fit_params={"algorithm": "parallel", "fun": "logcosh"},
    )
    print(f"[{participant}] fit FastICA ({n_components} components)", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ica.fit(ica_fit, picks="eeg", decim=5, reject_by_annotation=True, verbose=False)
    warning_messages = [str(item.message) for item in caught]
    converged = int(ica.n_iter_) < 1000 and not any("did not converge" in item.lower() for item in warning_messages)
    components = component_metrics(ica, ica_fit, blink_samples)
    strong_components = components.loc[
        components.classification == "STRONG OCULAR CANDIDATE", "component"
    ].astype(int).tolist()
    if not strong_components:
        raise RuntimeError(f"{participant}: unchanged P007 rule found no strong ocular component")

    before_uv = main_data.get_data() * 1e6
    cleaned = main_data.copy()
    ica.apply(cleaned, exclude=strong_components, verbose=False)
    after_uv = cleaned.get_data() * 1e6
    freq, before_psd, before_power = posterior_psd_and_power(before_uv, SFREQ)
    freq_after, after_psd, after_power = posterior_psd_and_power(after_uv, SFREQ)
    if not np.array_equal(freq, freq_after):
        raise RuntimeError(f"{participant}: before/after frequency vectors differ")
    result = {
        "participant": participant,
        "posterior_power_before_uv2": before_power,
        "posterior_power_after_uv2": after_power,
        "change_percent": 100.0 * (after_power - before_power) / before_power,
        "removed_components": ";".join(str(item) for item in strong_components),
        "n_removed_components": len(strong_components),
        "blink_events": int(len(blink_samples)),
        "ica_iterations": int(ica.n_iter_),
        "ica_converged": bool(converged),
    }
    audit = {
        **provenance,
        "blink_detection": blink_info,
        "mne_rank": rank,
        "ica_n_components": int(ica.n_components_),
        "ica_iterations": int(ica.n_iter_),
        "ica_converged": bool(converged),
        "fit_warnings": warning_messages,
        "strong_components": strong_components,
        "component_metrics": components.to_dict(orient="records"),
    }
    del main_data, ica_fit, cleaned, before_uv, after_uv, ica
    gc.collect()
    return result, audit, freq, before_psd, after_psd


def save_figure(curves, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, participant in zip(axes, PARTICIPANTS):
        freq, before, after = curves[participant]
        mask = (freq >= 1.0) & (freq <= 40.0)
        ax.plot(freq[mask], 10 * np.log10(before[mask] + 1e-30), label="Before ICA", color="#245C9F", lw=1.6)
        ax.plot(freq[mask], 10 * np.log10(after[mask] + 1e-30), label="After ocular IC removal", color="#C44E52", lw=1.6)
        ax.set_title(participant, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Posterior median PSD (dB µV²/Hz)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Supplementary ICA diagnostic: posterior PSD before and after", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "posterior_psd.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        participant: next((args.root / f"sub-{participant}").rglob("*.xdf"))
        for participant in PARTICIPANTS
    }
    results, audits, curves = [], {}, {}
    for participant in PARTICIPANTS:
        result, audit, freq, before, after = process_participant(participant, paths[participant])
        results.append(result)
        audits[participant] = audit
        curves[participant] = (freq, before, after)
        (args.output / f"{participant}_ica_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)

    table = pd.DataFrame(results)
    table.to_csv(args.output / "posterior_power.csv", index=False, encoding="utf-8-sig")
    save_figure(curves, args.output)
    parameters = {
        "scope": PARTICIPANTS,
        "method_source": "P007 ICA diagnostic reproduced without redesign",
        "main_comparison_filter_hz": [0.1, 40.0],
        "ica_fit_filter_hz": [1.0, 40.0],
        "filter": {"method": "FIR", "phase": "zero", "window": "hamming", "design": "firwin"},
        "notch": None,
        "reference": "SAGA Average Reference retained; no additional rereference",
        "ica": {"method": "fastica", "algorithm": "parallel", "fun": "logcosh", "random_state": 97, "max_iter": 1000, "fit_decim": 5, "n_components": 15},
        "ocular_selection": {
            "automated_not_manual": True,
            "strong_requires_all": {
                "maximum_absolute_Fp1_or_Fp2_correlation": ">= 0.45",
                "blink_locked_average_peak_z": ">= 0.35",
                "frontal_to_posterior_topography_ratio": ">= 1.5",
                "slow_1_4_to_1_40_power_ratio": ">= 0.30",
            },
            "maximum_strong_components": 2,
            "selection_did_not_use_posterior_power_change": True,
        },
        "posterior_channels": POSTERIOR,
        "psd": {"method": "scipy.signal.welch", "window": "hann", "segment_seconds": 30, "overlap_seconds": 15, "detrend": "constant", "power_definition": "integral of posterior-channel median PSD from 1 through 40 Hz"},
        "formal_outputs_modified": False,
        "cleaned_eeg_saved": False,
        "software": {"mne": mne.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "pyxdf": getattr(pyxdf, "__version__", "unknown")},
    }
    (args.output / "parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), args.output / "ica_diagnostic.py")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
