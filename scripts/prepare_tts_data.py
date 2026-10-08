"""
Data Preparation and Curation Pipeline for Kashmiri TTS.
Transforms raw speech datasets (Hugging Face, Kaggle, and Accent Classification)
into clean, standardized, 22,050 Hz normalized audio and canonical Kashmiri transcripts ready for training.
"""

import sys
import random
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

from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from text_normalizer import normalize_kashmiri_text
from audio_cleaner import clean_audio_file
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    HF_RAW_DIR,
    KAGGLE_RAW_DIR,
    ACCENT_RAW_DIR,
    SAMPLE_RATE
)


def load_raw_dataset_manifests(sources: List[str]) -> List[Dict]:
    """Loads and aggregates metadata rows from specified raw corpora."""
    aggregated = []

    source_configs = {
        "hf": {
            "meta_csv": HF_RAW_DIR / "metadata.csv",
            "wavs_dir": HF_RAW_DIR / "wavs",
            "default_speaker": "hf_speaker"
        },
        "kaggle": {
            "meta_csv": KAGGLE_RAW_DIR / "metadata.csv",
            "wavs_dir": KAGGLE_RAW_DIR,
            "default_speaker": "kaggle_speaker"
        },
        "accent": {
            "meta_csv": ACCENT_RAW_DIR / "metadata.csv",
            "wavs_dir": ACCENT_RAW_DIR / "wavs",
            "default_speaker": "spk_kashmiri_accent"
        }
    }

    for src_name in sources:
        if src_name not in source_configs:
            continue
        cfg = source_configs[src_name]
        meta_path = cfg["meta_csv"]
        wavs_dir = cfg["wavs_dir"]

        if not meta_path.exists():
            print(f"[!] Notice: {src_name.upper()} metadata not found at {meta_path}. Skipping.")
            continue

        try:
            df = pd.read_csv(meta_path)
            print(f"[+] Loaded {len(df)} records from {src_name.upper()} corpus manifest.")
            for _, row in df.iterrows():
                audio_name = str(row.get("audio_file", "")).strip()
                audio_path = Path(audio_name) if Path(audio_name).is_absolute() else (wavs_dir / audio_name)
                text = str(row.get("text", "")).strip()
                speaker_id = str(row.get("speaker_id", cfg["default_speaker"])).strip()
                dialect = str(row.get("dialect", "general"))

                aggregated.append({
                    "audio_path": audio_path,
                    "text": text,
                    "speaker_id": speaker_id,
                    "source": src_name,
                    "dialect": dialect
                })
        except Exception as e:
            print(f"[-] Error loading {src_name} metadata: {e}")

    return aggregated


def prepare_tts_dataset(
    sources: Optional[List[str]] = None,
    output_dir: Path = PROCESSED_DATA_DIR,
    val_split_ratio: float = 0.1,
    random_seed: int = 42,
    max_total_samples: Optional[int] = None
):
    """Processes aggregated raw speech corpora into standardized 22,050 Hz Kashmiri TTS data."""
    if sources is None:
        sources = ["hf", "kaggle", "accent"]

    random.seed(random_seed)
    print(f"\n{'='*70}")
    print(f" STARTING MULTI-DATASET KASHMIRI TTS CURATION PIPELINE")
    print(f" • Selected Sources : {', '.join(sources)}")
    print(f" • Output Directory : {output_dir}")
    print(f" • Target Sampling  : {SAMPLE_RATE} Hz")
    print(f"{'='*70}\n")

    raw_items = load_raw_dataset_manifests(sources)
    if not raw_items:
        print("[-] No raw samples found across selected sources. Run download_dataset.py first.")
        return

    print(f"\n[+] Total raw items aggregated across datasets: {len(raw_items)}")
    if max_total_samples and len(raw_items) > max_total_samples:
        random.shuffle(raw_items)
        raw_items = raw_items[:max_total_samples]
        print(f"[+] Subsampled to {max_total_samples} items for processing.")

    output_wavs_dir = output_dir / "wavs"
    output_wavs_dir.mkdir(parents=True, exist_ok=True)

    valid_entries = []
    discarded_stats = {"audio_invalid": 0, "text_empty": 0, "file_missing": 0}
    speaker_counts = {}

    for idx, item in enumerate(tqdm(raw_items, desc="Cleaning & Standardizing Audio")):
        raw_audio_path = item["audio_path"]
        raw_text = item["text"]
        speaker_id = item["speaker_id"]
        source_name = item["source"]
        dialect = item["dialect"]

        if not raw_audio_path.exists():
            discarded_stats["file_missing"] += 1
            continue

        # 1. Normalize Kashmiri Perso-Arabic text
        clean_text = normalize_kashmiri_text(raw_text)
        if len(clean_text) < 2:
            discarded_stats["text_empty"] += 1
            continue

        # 2. Clean, resample, trim, and normalize audio
        out_wav_name = f"ks_synth_{source_name}_{idx:05d}.wav"
        out_wav_path = output_wavs_dir / out_wav_name

        audio_res = clean_audio_file(
            input_path=raw_audio_path,
            output_path=out_wav_path,
            target_sr=SAMPLE_RATE
        )

        if not audio_res["valid"]:
            discarded_stats["audio_invalid"] += 1
            continue

        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

        valid_entries.append({
            "audio_path": str(out_wav_path.resolve()),
            "relative_wav": f"wavs/{out_wav_name}",
            "speaker_id": speaker_id,
            "text": clean_text,
            "duration": audio_res["duration"],
            "snr_db": audio_res["snr_db"],
            "source": source_name,
            "dialect": dialect
        })

    print(f"\n[+] Curation Summary:")
    print(f" * Validated & Standardized Samples : {len(valid_entries)}")
    print(f" * Discarded (Audio Quality/Duration): {discarded_stats['audio_invalid']}")
    print(f" * Discarded (Text Empty/Invalid)    : {discarded_stats['text_empty']}")
    print(f" * Discarded (File Missing)          : {discarded_stats['file_missing']}")
    print(f"\n • Distribution across Speaker / Dialect Embeddings:")
    for spk, cnt in speaker_counts.items():
        print(f"   - {spk}: {cnt} utterances")

    if not valid_entries:
        print("[-] No valid samples were retained. Check raw files and SNR thresholds.")
        return

    # Shuffle and split into Train and Validation sets
    random.shuffle(valid_entries)
    n_val = max(1, int(len(valid_entries) * val_split_ratio))
    val_set = valid_entries[:n_val]
    train_set = valid_entries[n_val:]

    def _write_filelist(items: List[Dict], filepath: Path):
        with open(filepath, "w", encoding="utf-8") as f:
            for it in items:
                f.write(f"{it['audio_path']}|{it['speaker_id']}|{it['text']}\n")

    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"
    metadata_csv = output_dir / "metadata.csv"

    _write_filelist(train_set, train_file)
    _write_filelist(val_set, val_file)
    pd.DataFrame(valid_entries).to_csv(metadata_csv, index=False, encoding="utf-8")

    print(f"\n[OK] Generated Training split ({len(train_set)} samples): {train_file}")
    print(f"[OK] Generated Validation split ({len(val_set)} samples): {val_file}")
    print(f"[OK] Generated Clean Multi-Dataset Metadata: {metadata_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and curate multi-source Kashmiri speech datasets.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["hf", "kaggle", "accent"],
        help="Sources to include: hf, kaggle, accent"
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to process")

    args = parser.parse_args()
    prepare_tts_dataset(
        sources=args.sources,
        val_split_ratio=args.val_ratio,
        max_total_samples=args.max_samples
    )
