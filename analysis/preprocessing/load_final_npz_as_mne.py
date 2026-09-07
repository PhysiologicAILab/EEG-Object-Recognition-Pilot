"""Load one final cohort NPZ derivative as an MNE EpochsArray.

This does not preprocess, reject, filter, rereference, or alter any epoch.
Use --save-fif only when the local MNE import is working normally.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd


def load_epochs(path: Path):
    # These trusted project derivatives contain object arrays for condition/run labels.
    with np.load(path, allow_pickle=True) as item:
        data_v = item["data_v"]
        times = item["times_s"]
        channels = item["channel_names"].tolist()
        xyz = item["channel_xyz_m"]
        sfreq = float(item["sfreq_hz"])
        conditions = item["condition"].tolist()
        event_codes = item["event_code"].astype(int)
        metadata = pd.DataFrame({
            "run": item["run"].tolist(),
            "trial_in_run": item["trial_in_run"].astype(int),
            "global_trial": item["global_trial"].astype(int),
            "condition": conditions,
        })
    info = mne.create_info(channels, sfreq, ch_types="eeg")
    info["highpass"], info["lowpass"], info["line_freq"] = 0.1, 40.0, 50.0
    info["description"] = "Final cohort: 0.1-40 Hz; SAGA Average Reference retained; ocular then 250 µV PTP rejection"
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(channels, xyz)), coord_frame="head")
    info.set_montage(montage, on_missing="raise")
    event_id = {condition: code for condition, code in sorted(set(zip(conditions, event_codes)), key=lambda x: x[1])}
    events = np.column_stack([np.arange(len(data_v)), np.zeros(len(data_v), int), event_codes])
    return mne.EpochsArray(
        data_v, info, events=events, event_id=event_id,
        tmin=float(times[0]), baseline=None, metadata=metadata, verbose="ERROR",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--save-fif", type=Path)
    args = parser.parse_args()
    epochs = load_epochs(args.npz)
    print(epochs)
    print("Channels:", epochs.ch_names)
    print("Montage attached:", epochs.get_montage() is not None)
    if args.save_fif:
        epochs.save(args.save_fif, overwrite=False)
        print("Saved:", args.save_fif)


if __name__ == "__main__":
    main()
