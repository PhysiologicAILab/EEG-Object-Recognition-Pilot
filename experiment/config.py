"""Fixed researcher configuration for the PsychoPy experiment."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VALIDATION_OUTPUTS_DIR = OUTPUTS_DIR / "_run_start_validation"
OBJECTS_FILE = DATA_DIR / "objects.json"
TRIGGER_CODEBOOK_FILE = DATA_DIR / "trigger_codebook.csv"

PROTOCOL_VERSION = "v3_formal_six_objects_70occ"
CANONICAL_MASTER_SEED = 27062026
RUN_TEMPLATE_DIR = DATA_DIR / "run_templates" / PROTOCOL_VERSION
OCCLUSION_QA_FILE = PROJECT_ROOT / "stimuli_final_candidates" / "metadata" / "occlusion_report.csv"

EXPERIMENT_TITLE = (
    "EEG-Assisted Everyday Object Target Selection under Visual Occlusion: "
    "A Pilot Study of Auditory Object Cues"
)

OBJECTS = ("bowl", "cup", "plate", "knife", "fork", "spoon")
VISIBILITIES = ("visible", "occluded")
AUDIO_CONDITIONS = ("none", "congruent", "incongruent")
CONDITION_CODES = {
    "visible_none": 11,
    "visible_congruent": 12,
    "visible_incongruent": 13,
    "occluded_none": 21,
    "occluded_congruent": 22,
    "occluded_incongruent": 23,
}

FIXATION_MS = 800
AUDIO_LEAD_MS = 0
IMAGE_DURATION_MS = 1000
SILENT_DELAY_MS = 600
RESPONSE_WINDOW_MS = 2000
RECORDING_GUARD_MS = 100
DESIRED_PRE_ROLL_MS = 500
MIN_CLEAN_PRE_ROLL_MS = 400
RECORDING_TAIL_MS = 700
MICROPHONE_PREFLIGHT_MS = 4000
MICROPHONE_PEAK_THRESHOLD = 0.001
WAV_DURATION_TOLERANCE_MS = 300
ITI_MIN_MS = 800
ITI_MAX_MS = 1200
FEEDBACK_MS = 600
DELAYED_NAMING = True
DISPLAY_TIMING_TOLERANCE_MS = 50.0
TIMING_WARNING_CONSECUTIVE_TRIALS = 3
TIMING_WARNING_RUN_PERCENT = 5.0
TIMING_WARNING_MIN_VIOLATIONS = 2
RUN_WARMUP_BLANK_FRAMES = 60
RUN_REFRESH_SAMPLE_FRAMES = 120
RUN_MICROPHONE_REWARM_SECONDS = 0.10
RUN_START_VALIDATION_TRIALS_PER_RUN = 5
MIN_PLAUSIBLE_REFRESH_HZ = 20.0
MAX_PLAUSIBLE_REFRESH_HZ = 360.0
REFRESH_CALIBRATION_MAX_RELATIVE_MAD = 0.05
SESSION_REFRESH_DRIFT_TOLERANCE_PERCENT = 15.0

OCCLUSION_LEVEL = 70
OCCLUSION_STYLE = "diagnostic_irregular_patch"
OCCLUSION_PATTERNS = 1
EXEMPLARS_PER_OBJECT = 4

FULL_REPETITIONS_PER_OBJECT_CONDITION = 12
FULL_RUNS = 6
FULL_TRIALS_PER_RUN = 72
FULL_MAIN_TRIALS = 432
PILOT_RUNS = 1
PILOT_MAIN_TRIALS = 36
PRACTICE_TRIALS = 12
MAX_RANDOMISATION_ATTEMPTS = 1000

AUDIO_BACKEND = "ptb"
LSL_STREAM_NAME = "PsychoPyMarkers"
LSL_STREAM_TYPE = "Markers"
LSL_SOURCE_PREFIX = "ucl-eeg-object-naming"

WINDOW_SIZE = (1280, 800)
BACKGROUND_COLOUR = (0.0, 0.0, 0.0)
TEXT_COLOUR = "white"
IMAGE_SIZE_HEIGHT = 0.72
MONITOR_NAME = "testMonitor"

MANIFEST_ID_FIELDS = (
    "participant_id", "participant_number", "latin_square_row", "source_list_id",
    "presented_run_number", "canonical_list_seed", "protocol_version",
)

BEHAVIOUR_FIELDS = (
    "participant_id", "participant_number", "session_id", "protocol_version",
    "latin_square_row", "source_list_id", "presented_run_number", "canonical_list_seed",
    "run_id", "trial_index_global", "trial_index_run", "visibility", "audio_condition",
    "condition_code", "image_object", "image_label", "exemplar_id", "image_path",
    "occlusion_level", "occlusion_pattern_id", "audio_object", "audio_label", "audio_path",
    "is_congruent", "correct_response", "accepted_responses", "fixation_ms_planned",
    "fixation_ms_actual", "fixation_frame_count", "fixation_first_flip", "fixation_last_flip",
    "image_duration_ms_planned", "image_duration_ms_actual", "image_frame_count",
    "image_first_flip", "image_last_flip", "planned_audio_onset_ptb",
    "condition_audio_onset_time", "condition_audio_offset_time", "audio_marker_timestamp",
    "image_offset_time", "silent_delay_ms_planned", "silent_delay_ms_actual",
    "silent_delay_frame_count", "response_window_ms", "response_window_ms_actual",
    "response_window_end_time", "response_frame_count", "response_cue_flip",
    "recording_segment_start_time", "response_cue_time", "recording_segment_stop_time",
    "clean_pre_roll_ms", "recording_duration_ms", "tail_ms", "wav_duration_ms",
    "sample_count", "peak_amplitude", "response_wav_path", "timeout", "iti_ms",
    "iti_ms_actual", "iti_frame_count", "is_practice", "random_seed", "trial_start_iso",
    "recording_status", "microphone_start_block_ms",
    "session", "run", "trial_in_run", "global_trial", "object",
    "image_filename", "audio_filename", "spoken_condition_word",
    "response_cue_timestamp", "microphone_segment_start", "microphone_segment_stop",
    "response_wav_filename", "audio_status",
    "planned_fixation_ms", "actual_fixation_ms", "planned_image_ms", "actual_image_ms",
    "planned_silent_delay_ms", "actual_silent_delay_ms", "planned_response_ms",
    "actual_response_ms", "timing_violation", "timing_violation_phase", "timing_error_ms",
    "timing_violation_reason",
)

EVENT_FIELDS = (
    "participant_id", "participant_number", "session_id", "protocol_version",
    "latin_square_row", "source_list_id", "presented_run_number", "run_id",
    "trial_index_global", "trial_index_run", "condition", "condition_code", "object_label",
    "event_name", "event_code", "psychopy_timestamp", "lsl_timestamp",
    "browser_or_system_datetime_iso", "marker_sent", "notes",
)

AUDIO_MANIFEST_FIELDS = (
    "participant_id", "participant_number", "session_id", "protocol_version",
    "latin_square_row", "source_list_id", "presented_run_number", "run_id",
    "trial_index_global", "trial_index_run", "condition", "microphone_device_name",
    "microphone_device_index", "microphone_sample_rate", "microphone_stream_initialised_time",
    "condition_audio_onset_time", "condition_audio_offset_time", "image_offset_time",
    "recording_segment_start_time", "response_cue_time", "response_window_end_time",
    "recording_segment_stop_time", "clean_pre_roll_ms", "tail_ms", "recording_duration_ms",
    "recording_started_iso", "recording_stopped_iso", "wav_duration_ms", "sample_count",
    "peak_amplitude", "wav_sample_rate", "wav_channels", "recording_status", "final_wav_path",
)

TRIGGER_CODES = {
    "lsl_self_test": 0,
    "fixation_onset": 1,
    "visible_none_image_onset": 11,
    "visible_congruent_image_onset": 12,
    "visible_incongruent_image_onset": 13,
    "occluded_none_image_onset": 21,
    "occluded_congruent_image_onset": 22,
    "occluded_incongruent_image_onset": 23,
    "congruent_audio_onset": 31,
    "incongruent_audio_onset": 32,
    "image_offset": 40,
    "silent_delay_onset": 41,
    "response_cue_onset": 42,
    "microphone_recording_start": 43,
    "microphone_recording_stop": 44,
    "trial_start": 50,
    "trial_end": 51,
    "run_start": 60,
    "run_end": 61,
    "break_start": 70,
    "break_end": 71,
    "practice_start": 80,
    "practice_end": 81,
    "experiment_start": 90,
    "experiment_end": 91,
    "experiment_abort": 98,
}

INSTRUCTION_PAGES = {
    "en": (
        ("Task overview", "You will see pictures of everyday objects.\n\nYour task is to identify the object shown in the picture."),
        ("The image is the correct source", "Sometimes you will also hear an English object word.\n\nAlways answer according to the picture. Do not answer according to the word that you hear."),
        ("Delayed naming", "Identify the picture silently.\n\nWait until the microphone icon appears.\n\nSay the object name immediately after the icon appears."),
        ("Standard English responses", "Use only the six standard English object names introduced in the next section."),
        ("EEG instructions", "Keep your head and body still. Avoid blinking during picture presentation when possible.\n\nBlink normally during breaks. Ask the experimenter if anything is unclear."),
    ),
    "zh": (
        ("任务概述", "屏幕上会出现日常物品图片。\n\n你的任务是辨认图片中的物品。"),
        ("以图片为准", "有些试次会同时播放一个英文物品词。\n\n请始终根据图片作答，不要根据听到的单词作答。"),
        ("延迟命名", "先在心中辨认图片。\n\n等待麦克风图标出现。\n\n图标出现后立即说出物品名称。"),
        ("标准英文回答", "请只使用下一阶段介绍的六个标准英文物品名称。"),
        ("EEG 注意事项", "请保持头部和身体不动；图片呈现时尽量避免眨眼。\n\n休息时可以正常眨眼。如有疑问，请询问实验人员。"),
    ),
}

TEXT = {
    "en": {
        "continue": "Press SPACE or ENTER to continue",
        "stage_familiarisation": "Stage 1: Object and Word Familiarisation",
        "stage_practice": "Stage 2: Practice Trials",
        "stage_formal": "Formal Experiment",
        "practice": "There will be 12 practice trials.",
        "practice_done": "Practice is complete.\n\nAnswer according to the image. Wait for the microphone icon, then speak immediately.\n\nAsk the experimenter now if anything remains unclear.",
        "speak_now": "Speak now",
        "recorded": "Recorded",
        "mic_test_title": "Compulsory microphone test",
        "mic_test_prompt": "Say a short word immediately when the microphone icon appears.",
        "mic_test_controls": "R = record again    P = play recording    ENTER = continue after PASS",
        "familiarisation_hint": "Use this exact English name. R = replay; SPACE or ENTER = continue.",
        "familiarisation_summary": "Use these exact English names during the experiment.",
        "break": "Break",
        "break_body": "You may take as long as you need. Press SPACE or ENTER when ready.",
        "complete": "Experiment complete\n\nThank you. Please tell the researcher.",
        "asset_ready": "Asset check passed\nMarker outlet ready\nMicrophone stream initialised and warm",
        "start_hint": "Press ENTER to continue or A to run the asset check again",
        "recording_warning": "Recording validation failed. Please call the experimenter.",
    },
    "zh": {
        "continue": "按空格键或回车键继续",
        "stage_familiarisation": "阶段 1：物品与英文词熟悉",
        "stage_practice": "阶段 2：练习试次",
        "stage_formal": "正式实验",
        "practice": "接下来有 12 个练习试次。",
        "practice_done": "练习已完成。\n\n请根据图片作答；等待麦克风图标出现后立即说出答案。\n\n如仍有疑问，请现在询问实验人员。",
        "speak_now": "请作答",
        "recorded": "已录音",
        "mic_test_title": "必须完成的麦克风测试",
        "mic_test_prompt": "麦克风图标出现后，请立即说一个简短单词。",
        "mic_test_controls": "R = 重新录音    P = 播放录音    ENTER = 通过后继续",
        "familiarisation_hint": "实验中请使用这个英文名称。R = 重播；空格或回车 = 继续。",
        "familiarisation_summary": "实验中请使用以下标准英文名称。",
        "break": "休息",
        "break_body": "你可以按需要休息。准备好后按空格键或回车键继续。",
        "complete": "实验完成\n\n谢谢，请告知实验人员。",
        "asset_ready": "资源检查通过\nMarker outlet 已就绪\n麦克风数据流已初始化并预热",
        "start_hint": "按回车键继续，或按 A 再次检查资源",
        "recording_warning": "录音验证失败，请通知实验人员。",
    },
}
