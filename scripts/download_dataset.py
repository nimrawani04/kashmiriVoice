"""
Dataset Downloader and Ingestion Manager for Kashmiri Speech Datasets.
Supports:
1. Hugging Face: programindz/kashmiri-audio-corpus (Continuous speech with Perso-Arabic text)
2. Kaggle: umar1103/converteddatanew (Isolated spoken Kashmiri vocabulary)
3. GitHub: shehzensidiq/kashmiri-accent-classification (Regional accents: Kupwara, Bandipora, Shopian, Islamabad)
"""

import os
import io
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf
import pandas as pd
from tqdm import tqdm

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
from config import (
    RAW_DATA_DIR,
    HF_DATASET_ID,
    KAGGLE_DATASET_ID,
    ACCENT_REPO_URL,
    ACCENT_REPO_DIR,
    ACCENT_RAW_DIR,
    KAGGLE_RAW_DIR,
    HF_RAW_DIR,
    KASHMIRI_DIALECTS
)

# Standard Persian-Arabic Kashmiri script mapping for Kaggle vocabulary
KAGGLE_WORD_PERSO_ARABIC = {
    "aah": "آہ",
    "adsa": "ادسا",
    "bandd": "بَند",
    "be": "بیٚیہِ",
    "chea": "چھیا",
    "khabri": "خَبری",
    "kya": "کیاہ",
    "mae": "مےٚ",
    "nah": "نہٕ",
    "novve": "نۆو",
    "theek": "ٹھیک",
    "vaarai": "واریاہ",
    "wuchh": "وُچھ",
    "yealle": "یلہِ"
}

# Regional Kashmiri conversational dialect prompts for accent recordings
DIALECT_DEFAULT_TRANSCRIPTS = {
    "kupwara": "کَمراز کٲشُر لَہجہِ بولُن",
    "bandipora": "وُلَر علاقُک کٲشُر لَہجہِ بولُن",
    "shopian": "مَراز کٲشُر لَہجہِ بولُن",
    "islamabad": "اِسلام آباد مَراز کٲشُر لَہجہِ بولُن"
}


def download_hf_kashmiri_corpus(max_samples: int = 200, output_dir: Path = HF_RAW_DIR):
    """
    Downloads samples from programindz/kashmiri-audio-corpus on Hugging Face.
    Extracts audio waveforms to WAV files and records transcripts to metadata.csv.
    """
    print(f"\n{'='*70}")
    print(f"[+] 1. Fetching Kashmiri Audio Corpus from Hugging Face: {HF_DATASET_ID}")
    print(f"{'='*70}")
    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset, Audio
        ds = load_dataset(HF_DATASET_ID, split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"[-] Error initializing Hugging Face datasets stream: {e}")
        return

    records = []
    print(f"[+] Streaming up to {max_samples} audio samples...")

    count = 0
    for idx, item in enumerate(tqdm(ds, total=max_samples, desc="Ingesting HF samples")):
        if count >= max_samples:
            break

        text = item.get("text", "")
        audio_info = item.get("audio", {})
        
        if not text or not audio_info:
            continue

        raw_bytes = audio_info.get("bytes")
        if not raw_bytes:
            continue

        wav_filename = f"hf_sample_{count:05d}.wav"
        wav_filepath = wavs_dir / wav_filename

        try:
            audio_array, sample_rate = sf.read(io.BytesIO(raw_bytes))
            sf.write(str(wav_filepath), audio_array, sample_rate)
            records.append({
                "audio_file": wav_filename,
                "text": text,
                "original_duration": round(len(audio_array) / sample_rate, 3),
                "speaker_id": "hf_speaker",
                "source": "huggingface"
            })
            count += 1
        except Exception:
            continue

    df = pd.DataFrame(records)
    metadata_csv = output_dir / "metadata.csv"
    df.to_csv(metadata_csv, index=False, encoding="utf-8")
    print(f"[OK] Successfully downloaded {len(records)} HF samples to: {output_dir}")
    print(f"[OK] HF metadata saved to: {metadata_csv}\n")


def ingest_kaggle_dataset(kaggle_dir: Path = KAGGLE_RAW_DIR, output_dir: Path = KAGGLE_RAW_DIR):
    """
    Downloads and indexes Kaggle Kashmiri Voice dataset (umar1103/converteddatanew).
    Maps folder-based word classes to canonical Kashmiri Perso-Arabic text.
    """
    print(f"\n{'='*70}")
    print(f"[+] 2. Ingesting Kaggle Kashmiri Dataset: {KAGGLE_DATASET_ID}")
    print(f"{'='*70}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already present or download via kagglehub
    existing_wavs = list(kaggle_dir.glob("**/*.wav")) if kaggle_dir.exists() else []
    if len(existing_wavs) < 50:
        try:
            import kagglehub
            print(f"[+] Downloading Kaggle dataset via kagglehub...")
            cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
            print(f"[+] Extracting files to: {kaggle_dir}")
            for item in cache_path.iterdir():
                target = kaggle_dir / item.name
                if item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
            existing_wavs = list(kaggle_dir.glob("**/*.wav"))
        except Exception as e:
            print(f"[!] Warning: Could not automatically download Kaggle dataset: {e}")
            print(f"    Please place extracted folders in '{kaggle_dir}'")

    if not existing_wavs:
        print(f"[-] No WAV files found in Kaggle directory: {kaggle_dir}")
        return

    print(f"[+] Found {len(existing_wavs)} audio files in Kaggle dataset.")

    records = []
    for wav in existing_wavs:
        word_class = wav.parent.name.lower()
        perso_arabic_text = KAGGLE_WORD_PERSO_ARABIC.get(word_class, word_class)
        
        try:
            info = sf.info(str(wav))
            dur = info.duration
        except Exception:
            dur = 0.0

        records.append({
            "audio_file": str(wav.resolve()),
            "text": perso_arabic_text,
            "word_label": wav.parent.name,
            "original_duration": round(dur, 3),
            "speaker_id": "kaggle_speaker",
            "source": "kaggle"
        })

    df = pd.DataFrame(records)
    metadata_csv = output_dir / "metadata.csv"
    df.to_csv(metadata_csv, index=False, encoding="utf-8")
    print(f"[OK] Indexed {len(records)} Kaggle samples across {len(KAGGLE_WORD_PERSO_ARABIC)} vocabulary words")
    print(f"[OK] Kaggle metadata saved to: {metadata_csv}\n")


def ingest_accent_classification_dataset(
    repo_dir: Path = ACCENT_REPO_DIR,
    output_dir: Path = ACCENT_RAW_DIR,
    include_augmented: bool = False
):
    """
    Ingests Kashmiri Accent Classification Dataset (shehzensidiq/kashmiri-accent-classification).
    Extracts district dialect categories (Kupwara, Bandipora, Shopian, Islamabad)
    and maps them to distinct multi-speaker regional dialect embeddings.
    """
    print(f"\n{'='*70}")
    print(f"[+] 3. Ingesting Kashmiri Accent Classification Dataset: {ACCENT_REPO_URL}")
    print(f"{'='*70}")

    # Clone repo if not present
    if not repo_dir.exists() or not (repo_dir / "Dataset").exists():
        print(f"[+] Cloning repository {ACCENT_REPO_URL}...")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", ACCENT_REPO_URL, str(repo_dir)],
            check=True
        )

    dataset_dir = repo_dir / "Dataset"
    final_audio_dir = dataset_dir / "Audio" / "Final Data"
    augmented_audio_dir = dataset_dir / "Audio" / "Augmented"
    audio_meta_file = dataset_dir / "audio_meta.csv"

    if not audio_meta_file.exists():
        print(f"[-] Accent metadata file not found at: {audio_meta_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_wavs_dir = output_dir / "wavs"
    out_wavs_dir.mkdir(parents=True, exist_ok=True)

    # Read audio_meta.csv (format: filename, district)
    df_labels = pd.read_csv(audio_meta_file, header=None, names=["filename", "district"])
    label_dict = dict(zip(df_labels["filename"].str.strip(), df_labels["district"].str.strip()))

    audio_files = list(final_audio_dir.glob("*.wav"))
    if include_augmented and augmented_audio_dir.exists():
        audio_files.extend(list(augmented_audio_dir.glob("*.wav")))

    print(f"[+] Processing {len(audio_files)} accent classification audio recordings...")

    records = []
    district_counts = {}

    for wav in tqdm(audio_files, desc="Indexing Accent Recordings"):
        name = wav.name
        district = label_dict.get(name)
        
        # Infer district from filename prefix if not in dictionary
        if not district:
            prefix = name.split("_")[0]
            district = prefix

        dist_lower = district.lower()
        dialect_info = KASHMIRI_DIALECTS.get(dist_lower, {
            "name_en": district,
            "dialect_zone": "General Kashmiri",
            "speaker_id": f"spk_{dist_lower}"
        })

        speaker_id = dialect_info["speaker_id"]
        transcript = DIALECT_DEFAULT_TRANSCRIPTS.get(dist_lower, "کٲشُر لَہجہِ")

        # Copy or link WAV to output wavs directory
        dest_wav = out_wavs_dir / name
        if not dest_wav.exists():
            shutil.copy2(wav, dest_wav)

        try:
            info = sf.info(str(wav))
            dur = info.duration
            sr = info.samplerate
        except Exception:
            dur = 0.0
            sr = 48000

        district_counts[district] = district_counts.get(district, 0) + 1

        records.append({
            "audio_file": name,
            "text": transcript,
            "dialect": district,
            "dialect_zone": dialect_info.get("dialect_zone", "Kashmiri"),
            "speaker_id": speaker_id,
            "original_duration": round(dur, 3),
            "sample_rate": sr,
            "source": "accent_classification_repo"
        })

    df = pd.DataFrame(records)
    metadata_csv = output_dir / "metadata.csv"
    df.to_csv(metadata_csv, index=False, encoding="utf-8")

    print(f"[OK] Indexed {len(records)} accent speech samples to: {metadata_csv}")
    print(f" • Regional Breakdown:")
    for dist, cnt in district_counts.items():
        print(f"   - {dist}: {cnt} recordings")
    print(f"{'='*70}\n")


def ingest_all_datasets(max_hf_samples: int = 150):
    """Runs end-to-end ingestion across all 3 speech corpora."""
    print("\n" + "="*70)
    print(" KASHMIRI VOICE: MULTI-DATASET UNIFIED INGESTION SUITE")
    print(" 1. Hugging Face Audio Corpus (Continuous speech)")
    print(" 2. Kaggle Voice Dataset (Discrete spoken vocabulary)")
    print(" 3. Kashmiri Accent Classification Dataset (Kamraz & Maraz accents)")
    print("="*70)

    download_hf_kashmiri_corpus(max_samples=max_hf_samples)
    ingest_kaggle_dataset()
    ingest_accent_classification_dataset()

    print("[OK] ALL 3 DATASETS SUCCESSFULLY INGESTED AND INDEXED!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and ingest Kashmiri speech datasets.")
    parser.add_argument(
        "--source",
        choices=["hf", "kaggle", "accent", "all"],
        default="all",
        help="Dataset source to process (hf, kaggle, accent, all)"
    )
    parser.add_argument("--max_samples", type=int, default=150, help="Number of samples to stream from HF")
    parser.add_argument("--include_augmented", action="store_true", help="Include augmented audio in accent repo")

    args = parser.parse_args()

    if args.source == "all":
        ingest_all_datasets(max_hf_samples=args.max_samples)
    elif args.source == "hf":
        download_hf_kashmiri_corpus(max_samples=args.max_samples)
    elif args.source == "kaggle":
        ingest_kaggle_dataset()
    elif args.source == "accent":
        ingest_accent_classification_dataset(include_augmented=args.include_augmented)
