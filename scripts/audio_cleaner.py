"""
Audio Cleaning and Standardization Pipeline for Kashmiri Speech Data.
Prepares audio for high-fidelity Neural TTS training (HiFi-GAN/VITS/Flow Matching).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Tuple, Dict, Any, Optional
import numpy as np
import soundfile as sf
import librosa
from config import (
    SAMPLE_RATE,
    MIN_DURATION_SEC,
    MAX_DURATION_SEC,
    TARGET_LUFS,
    MIN_SNR_DB
)


def compute_snr(audio: np.ndarray) -> float:
    """Estimates the Signal-to-Noise Ratio (SNR) in decibels."""
    if len(audio) == 0:
        return 0.0
    signal_power = float(np.mean(audio ** 2))
    if signal_power < 1e-8:
        return 0.0

    frame_length = 512
    num_frames = len(audio) // frame_length
    if num_frames < 2:
        return 30.0

    frames = audio[:num_frames * frame_length].reshape((num_frames, frame_length))
    frame_powers = np.mean(frames ** 2, axis=1)
    
    # Sort frame powers: lowest 10% represent silent/background floor
    sorted_powers = np.sort(frame_powers)
    noise_power = float(np.mean(sorted_powers[:max(1, int(len(sorted_powers) * 0.10))]))
    
    if noise_power < 1e-9:
        return 35.0  # Clean recording with digital silence floor
    
    snr = 10.0 * np.log10(max(signal_power / noise_power, 1.0))
    return float(snr)


def normalize_loudness(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    """Normalizes audio to a consistent RMS energy level without digital clipping."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio
    gain = target_rms / rms
    normalized = audio * gain
    # Soft peak limiting if dynamic peaks exceed 0.95
    peak = np.max(np.abs(normalized))
    if peak > 0.95:
        normalized = normalized / peak * 0.95
    return normalized


def clean_audio_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    target_sr: int = SAMPLE_RATE,
    top_db: int = 28,
    min_duration: float = MIN_DURATION_SEC,
    max_duration: float = MAX_DURATION_SEC
) -> Dict[str, Any]:
    """
    Loads, cleans, resamples, trims silence, and standardizes an audio recording.
    Returns status metadata and cleaned audio.
    """
    result = {
        "valid": False,
        "input_path": str(input_path),
        "duration": 0.0,
        "snr_db": 0.0,
        "reason": ""
    }

    try:
        # Load audio (mono)
        audio, orig_sr = librosa.load(str(input_path), sr=None, mono=True)
    except Exception as e:
        result["reason"] = f"Failed to load audio: {str(e)}"
        return result

    # Check for empty or flatline audio
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-4:
        result["reason"] = "Audio is silent or empty"
        return result

    # Resample if needed
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    # Trim leading and trailing silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=512, hop_length=128)
    duration = len(audio_trimmed) / target_sr

    # Check duration bounds
    if duration < min_duration:
        result["reason"] = f"Duration {duration:.2f}s is below minimum {min_duration}s"
        return result
    if duration > max_duration:
        result["reason"] = f"Duration {duration:.2f}s exceeds maximum {max_duration}s"
        return result

    # Compute Signal-to-Noise Ratio
    snr = compute_snr(audio_trimmed)
    if snr < MIN_SNR_DB:
        result["reason"] = f"SNR {snr:.1f} dB is below threshold {MIN_SNR_DB} dB (too noisy)"
        return result

    # Normalize audio energy
    audio_normalized = normalize_loudness(audio_trimmed)

    # Export if output path is provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_normalized, target_sr, subtype="PCM_16")

    result["valid"] = True
    result["duration"] = round(duration, 2)
    result["snr_db"] = round(snr, 1)
    result["output_path"] = str(output_path) if output_path else None
    return result
