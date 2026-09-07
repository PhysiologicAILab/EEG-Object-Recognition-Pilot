# A Pilot EEG Dataset for Object Recognition under Visual Occlusion and Auditory Semantic Congruency

This repository contains the experiment and analysis code developed for a pilot multimodal EEG study of object naming under visual occlusion and auditory semantic congruency. The workflow combines continuous EEG, synchronised event markers, and spoken responses.

## Experimental design

The experiment uses a 2 × 3 within-subject design:

- Visibility: Visible, Occluded
- Audio: None, Congruent, Incongruent

The six object categories are bowl, cup, plate, knife, fork, and spoon. The formal PsychoPy entry point is `experiment/main_experiment.py`.

## Analysis workflow

The included scripts cover:

1. XDF import, EEG channel configuration, filtering, epoching, and final cohort preprocessing.
2. Acquisition, channel, ocular-artifact, retained-epoch, and supplementary ICA quality control.
3. Welch PSD estimation and channel-level absolute and relative spectral features.
4. Occipital/parietal ROI aggregation, repeated-measures statistics, post-hoc comparisons, and sensitivity analysis.
5. Trial-level speech inventory, constrained recognition, manual-label integration, naming accuracy, and response-time measures.
6. Low-level visual-feature analysis for the formal visible/occluded stimulus pairs.

Scientific parameters and participant-selection rules are defined in the relevant scripts. Input and output locations are project-relative or configurable through command-line arguments and documented environment variables.

## Repository structure

```text
EEG_Object_Recognition_Pilot_GitHub/
├── README.md
├── requirements.txt
├── .gitignore
├── experiment/
│   ├── main_experiment.py
│   ├── required modules and configuration
│   ├── formal run templates
│   └── formal visual and auditory stimuli
└── analysis/
    ├── preprocessing/
    ├── qc/
    ├── spectral_analysis/
    ├── statistics/
    ├── behavioural_speech_analysis/
    └── stimulus_analysis/
```

## Project contribution

This project developed:

- the PsychoPy experimental task;
- the synchronised EEG, event-marker, and speech-recording workflow;
- EEG preprocessing and quality-control scripts;
- spectral and ROI analysis scripts;
- the behavioural speech-analysis workflow;
- low-level stimulus-feature analysis; and
- a pilot multimodal EEG dataset for future research.

## Installation

Create a Python environment appropriate for PsychoPy and install the packages listed in `requirements.txt`. PsychoPy and hardware-related packages may require platform-specific installation steps. Large speech-recognition models are downloaded or configured separately and are not stored in this repository.


