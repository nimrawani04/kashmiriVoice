"""
Configuration module for Kashmiri Text-to-Speech (TTS) Engine.
Optimized for NVIDIA GeForce RTX 4050 (6GB VRAM) and Windows environment.
"""

from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

# Ensure essential directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Audio Parameters (Standard for high-fidelity Neural TTS / HiFi-GAN)
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0

# Audio Filtering Thresholds for TTS
MIN_DURATION_SEC = 0.6    # Allows single-word vocabulary and conversational clips
MAX_DURATION_SEC = 11.0   # Longer clips cause VRAM spikes / alignment failure
TARGET_LUFS = -23.0       # ITU-R BS.1770 broadcast standard
MIN_SNR_DB = 10.0         # Signal-to-noise ratio cutoff

# Hardware & Training Parameters (Tuned for RTX 4050 6GB VRAM)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16_MIXED = True if torch.cuda.is_available() else False
BATCH_SIZE = 8            # Conservative for 6GB VRAM to avoid OOM
GRADIENT_ACCUMULATION = 4  # Effective batch size = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-6
WARMUP_STEPS = 1000
MAX_EPOCHS = 100
SAVE_EVERY_N_EPOCHS = 5

# Dataset Identifiers & Repositories
HF_DATASET_ID = "programindz/kashmiri-audio-corpus"
KAGGLE_DATASET_ID = "umar1103/converteddatanew"
ACCENT_REPO_URL = "https://github.com/shehzensidiq/kashmiri-accent-classification"
BENCHMARK_MODEL_ID = "GAASH-Lab/Matcha-TTS-Kashmiri"

# Dataset Working Directories
HF_RAW_DIR = RAW_DATA_DIR / "hf_corpus"
KAGGLE_RAW_DIR = RAW_DATA_DIR / "kaggle_corpus"
ACCENT_REPO_DIR = RAW_DATA_DIR / "kashmiri-accent-classification"
ACCENT_RAW_DIR = RAW_DATA_DIR / "accent_corpus"

# Ensure dataset raw directories exist
for path in [HF_RAW_DIR, KAGGLE_RAW_DIR, ACCENT_RAW_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Kashmiri Regional Dialects and Accent Labels (from shehzensidiq repo)
KASHMIRI_DIALECTS = {
    "kupwara": {"name_en": "Kupwara", "dialect_zone": "Kamraz (Northern Kashmiri)", "speaker_id": "spk_kupwara"},
    "bandipora": {"name_en": "Bandipora", "dialect_zone": "Kamraz / Wular (Northern Kashmiri)", "speaker_id": "spk_bandipora"},
    "shopian": {"name_en": "Shopian", "dialect_zone": "Maraz (Southern Kashmiri)", "speaker_id": "spk_shopian"},
    "islamabad": {"name_en": "Islamabad (Anantnag)", "dialect_zone": "Maraz (Southern Kashmiri)", "speaker_id": "spk_islamabad"}
}

# Speaker Mapping for Multi-Speaker Conditioning
SPEAKER_ID_MAP = {
    "hf_speaker": 0,
    "kaggle_speaker": 1,
    "spk_kupwara": 2,
    "spk_bandipora": 3,
    "spk_shopian": 4,
    "spk_islamabad": 5,
    "kashmiri_default": 0
}

# Kashmiri Language Information
LANG_CODE = "ks"
SCRIPT_NAME = "Perso-Arabic (کٲشُر)"
