"""
Dataset Inspection and Linguistic Audit Tool for Kashmiri Speech Data.
Analyzes audio duration, sampling rate, SNR quality, dialect distribution,
and Kashmiri Perso-Arabic text character coverage across all data sources.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import soundfile as sf
from text_normalizer import normalize_kashmiri_text, KASHMIRI_ALPHABET, CHAR_TO_ID
from audio_cleaner import compute_snr
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    HF_RAW_DIR,
    KAGGLE_RAW_DIR,
    ACCENT_RAW_DIR,
    MIN_DURATION_SEC,
    MAX_DURATION_SEC,
    MIN_SNR_DB
)


def audit_dataset(metadata_csv: Path, wavs_dir: Path, dataset_name: str = "Corpus") -> Dict[str, Any]:
    """Runs a complete linguistic and acoustic audit on a dataset directory."""
    print(f"\n{'='*70}")
    print(f" AUDITING KASHMIRI SPEECH CORPUS: {dataset_name.upper()} ({metadata_csv.parent.name})")
    print(f"{'='*70}")

    if not metadata_csv.exists():
        print(f"[-] Metadata CSV not found: {metadata_csv}")
        return {}

    df = pd.read_csv(metadata_csv)
    print(f"[+] Total samples indexed: {len(df)}")

    durations = []
    snrs = []
    sample_rates = set()
    all_chars = set()
    kashmiri_specific_counts = {"ٲ": 0, "ۄ": 0, "ۆ": 0, "ؠ": 0, "ۍ": 0}
    discarded_short = 0
    discarded_long = 0
    discarded_noisy = 0
    dialect_counts = {}

    for idx, row in df.iterrows():
        # Resolve audio file
        audio_name = str(row.get("audio_file", row.get("audio_path", "")))
        audio_path = wavs_dir / audio_name if not Path(audio_name).is_absolute() else Path(audio_name)
        
        # Dialect tracking if present
        dialect = str(row.get("dialect", row.get("speaker_id", "default")))
        dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1

        text = str(row.get("text", ""))
        normalized_text = normalize_kashmiri_text(text)
        all_chars.update(set(normalized_text))

        for k_char in kashmiri_specific_counts:
            kashmiri_specific_counts[k_char] += normalized_text.count(k_char)

        if audio_path.exists():
            try:
                info = sf.info(str(audio_path))
                sample_rates.add(info.samplerate)
                durations.append(info.duration)

                if info.duration < MIN_DURATION_SEC:
                    discarded_short += 1
                elif info.duration > MAX_DURATION_SEC:
                    discarded_long += 1

                # Sample SNR on a subset to keep audit fast
                if idx < 100:
                    data, _ = sf.read(str(audio_path))
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    snr = compute_snr(data)
                    snrs.append(snr)
                    if snr < MIN_SNR_DB:
                        discarded_noisy += 1
            except Exception:
                continue

    durations = np.array(durations) if durations else np.array([0.0])
    snrs = np.array(snrs) if snrs else np.array([0.0])

    print(f"\n--- Acoustic Quality Metrics ---")
    print(f" * Sample rates detected    : {list(sample_rates)}")
    print(f" * Total Audio Duration     : {np.sum(durations)/60:.2f} minutes ({np.sum(durations)/3600:.2f} hours)")
    print(f" * Duration Mean / Median   : {np.mean(durations):.2f}s / {np.median(durations):.2f}s")
    print(f" * Duration Min / Max       : {np.min(durations):.2f}s / {np.max(durations):.2f}s")
    print(f" * Estimated Mean SNR       : {np.mean(snrs):.1f} dB")
    print(f" * Short clips (<{MIN_DURATION_SEC}s)      : {discarded_short}")
    print(f" * Long clips (>{MAX_DURATION_SEC}s)     : {discarded_long}")

    if dialect_counts:
        print(f"\n--- Dialect / Speaker Breakdown ---")
        for dia, count in dialect_counts.items():
            print(f" * {dia}: {count} samples")

    print(f"\n--- Kashmiri Phonological & Script Coverage ---")
    print(f" * Unique Characters Found  : {len(all_chars)}")
    print(f" * Native Kashmiri Vowel 'ٲ' : {kashmiri_specific_counts['ٲ']} occurrences")
    print(f" * Native Kashmiri Vowel 'ۄ' : {kashmiri_specific_counts['ۄ']} occurrences")
    print(f" * Native Kashmiri Vowel 'ۆ' : {kashmiri_specific_counts['ۆ']} occurrences")
    print(f" * Palatalization Marker 'ؠ': {kashmiri_specific_counts['ؠ']} occurrences")

    unmapped = [c for c in all_chars if c not in CHAR_TO_ID]
    if unmapped:
        print(f" * Warning: {len(unmapped)} unmapped characters detected: {unmapped[:10]}")
    else:
        print(f" * Vocabulary Alignment: 100% of characters mapped to Kashmiri Perso-Arabic alphabet tokens!")

    print(f"{'='*70}\n")
    return {
        "dataset": dataset_name,
        "total_samples": len(df),
        "total_minutes": round(float(np.sum(durations)/60), 2),
        "mean_snr": round(float(np.mean(snrs)), 1),
        "kashmiri_vowels": kashmiri_specific_counts
    }


def audit_all_corpora():
    """Audits all available Kashmiri speech datasets."""
    targets = [
        {"name": "Hugging Face Audio Corpus", "meta": HF_RAW_DIR / "metadata.csv", "wavs": HF_RAW_DIR / "wavs"},
        {"name": "Kaggle Voice Corpus", "meta": KAGGLE_RAW_DIR / "metadata.csv", "wavs": KAGGLE_RAW_DIR},
        {"name": "Kashmiri Accent Classification Repo", "meta": ACCENT_RAW_DIR / "metadata.csv", "wavs": ACCENT_RAW_DIR / "wavs"},
        {"name": "Unified Processed Training Dataset", "meta": PROCESSED_DATA_DIR / "metadata.csv", "wavs": PROCESSED_DATA_DIR / "wavs"}
    ]

    for t in targets:
        if t["meta"].exists():
            audit_dataset(t["meta"], t["wavs"], dataset_name=t["name"])
        else:
            print(f"[!] Notice: {t['name']} metadata not yet generated ({t['meta'].name}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and inspect Kashmiri speech corpora.")
    parser.add_argument(
        "--target",
        choices=["all", "hf", "kaggle", "accent", "processed"],
        default="all",
        help="Target dataset to audit (all, hf, kaggle, accent, processed)"
    )

    args = parser.parse_args()
    if args.target == "all":
        audit_all_corpora()
    elif args.target == "hf":
        audit_dataset(HF_RAW_DIR / "metadata.csv", HF_RAW_DIR / "wavs", "Hugging Face")
    elif args.target == "kaggle":
        audit_dataset(KAGGLE_RAW_DIR / "metadata.csv", KAGGLE_RAW_DIR, "Kaggle")
    elif args.target == "accent":
        audit_dataset(ACCENT_RAW_DIR / "metadata.csv", ACCENT_RAW_DIR / "wavs", "Accent Classification")
    elif args.target == "processed":
        audit_dataset(PROCESSED_DATA_DIR / "metadata.csv", PROCESSED_DATA_DIR / "wavs", "Processed Dataset")
