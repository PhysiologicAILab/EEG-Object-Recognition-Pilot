# PsychoPy experiment

The formal experiment entry point is `main_experiment.py`.

## Contents

- `main_experiment.py`: experiment entry point.
- `config.py`, `data_manager.py`, `lsl_markers.py`, `microphone_recorder.py`, `stimulus_manager.py`, `trial_generator.py`: required local modules.
- `data/`: object definitions, trigger codebook, and six formal run templates.
- `stimuli_final_candidates/`: 24 visible and 24 corresponding occluded images.
- `assets/audio/spoken/`: the six spoken-word stimuli.

## Running the experiment

Open `main_experiment.py` in PsychoPy Coder and select Run.

Alternatively, `run_experiment.bat` can be used. It uses `python` by default. If PsychoPy has a separate Python executable, set the `PSYCHOPY_PYTHON` environment variable to that executable before running the batch file.

## Formal design

- Objects: bowl, cup, plate, knife, fork, spoon.
- Conditions: Visible/Occluded × None/Congruent/Incongruent.
- Formal session: six runs of 72 trials (432 trials in total).
- Practice: 12 trials.

The program records trial, event, behavioural, audio-manifest, configuration, protocol, and session-summary files. Trial WAV files are stored in the session audio directory.
