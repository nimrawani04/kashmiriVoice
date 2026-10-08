# 🎙️ Kashmiri Voice (کٲشُر آواز)

A Neural Speech Synthesis (TTS) & Regional Accent Classification Engine for the Kashmiri language (**کٲشُر**), built for Perso-Arabic script and multi-dialect voice conditioning across Kashmir valley.

Optimized for **Windows** and **NVIDIA RTX GPUs** (RTX 4050 6GB VRAM) with FP16 mixed precision and gradient accumulation.

---

## 🌟 Key Features

1. **Multi-Source Speech Corpora Ingestion**:
   - **Hugging Face** ([`programindz/kashmiri-audio-corpus`](https://huggingface.co/datasets/programindz/kashmiri-audio-corpus)): Sentence-level continuous Kashmiri speech with native Perso-Arabic transcripts.
   - **Kaggle** ([`umar1103/converteddatanew`](https://www.kaggle.com/datasets/umar1103/converteddatanew)): 375 spoken Kashmiri vocabulary recordings mapped to canonical Perso-Arabic script (*آہ, ادسا, بَند, بیٚیہِ, چھیا, خَبری, کیاہ, مےٚ, نہٕ, نۆو, ٹھیک, واریاہ, وُچھ, یلہِ*).
   - **Regional Accent Classification Repo** ([`shehzensidiq/kashmiri-accent-classification`](https://github.com/shehzensidiq/kashmiri-accent-classification)): Field speech recordings across 4 major Kashmiri dialect regions.

2. **Kashmiri Regional Dialect Conditioning**:
   - **Kupwara** (*Kamraz* / Northern Kashmiri)
   - **Bandipora** (*Wular* / Northern Kashmiri)
   - **Shopian** (*Maraz* / Southern Kashmiri)
   - **Islamabad** (*Anantnag* / *Maraz* / Southern Kashmiri)

3. **Linguistic Normalization**:
   - Full preservation and normalization of Kashmiri-specific vowels (`ٲ`, `ۄ`, `ۆ`, `ؠ`, `ۍ`) and palatalization glides.
   - Kashmiri numerical expansion (e.g. `1` -> `اکھ`, `2` -> `زٕ`, `10` -> `دَہ`).

4. **Regional Accent Classifier**:
   - Neural classification module predicting regional Kashmiri dialect from acoustic prosody and MFCC features with 90%+ confidence.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/kashmiri-voice.git
cd kashmiri-voice
pip install -r requirements.txt
```

### 2. Download and Ingest Datasets
Fetch and structure all three speech corpora:
```bash
python scripts/download_dataset.py --source all
```
*(Options: `--source all`, `--source hf`, `--source kaggle`, `--source accent`)*

### 3. Audit Dataset Quality & Phonology
Inspect audio sample rates, duration, SNR, dialect distribution, and vowel coverage:
```bash
python scripts/inspect_dataset.py --target all
```

### 4. Prepare and Normalize TTS Training Data
Resample to 22,050 Hz, normalize loudness to ITU-R -23 LUFS, and generate `train.txt` and `val.txt`:
```bash
python scripts/prepare_tts_data.py --sources hf kaggle accent
```

### 5. Train / Fine-Tune Acoustic Model
Train the multi-speaker neural TTS model on your local GPU:
```bash
python scripts/train.py --epochs 25
```

### 6. Synthesize Kashmiri Speech
Generate speech in any regional Kashmiri dialect:
```bash
python scripts/inference.py --text "سلام! تُہؠ چھِوا ٹھیک؟ کٲشُر بولُن چھُ مۆدُر" --dialect kupwara
```

### 7. Classify Kashmiri Voice Accent
Predict whether a Kashmiri audio recording is from Kupwara, Bandipora, Shopian, or Islamabad:
```bash
python scripts/accent_classifier.py --predict "path/to/kashmiri_audio.wav"
```

### 8. Run Self-Test Suite
Verify all components with 9 unit and integration tests:
```bash
python scripts/test_pipeline.py
```

---

## 📁 Repository Structure

```
kashmiri voice/
├── config.py                 # Core hyperparameters, directories & dialect configs
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes large audio datasets and model binaries
├── scripts/
│   ├── download_dataset.py   # Multi-corpus ingestion (HF, Kaggle, Accent repo)
│   ├── inspect_dataset.py    # Acoustic & phonological audit tool
│   ├── prepare_tts_data.py   # Audio standardization, -23 LUFS & Perso-Arabic normalizer
│   ├── audio_cleaner.py      # Resampling, silence trimming & SNR computation
│   ├── text_normalizer.py    # Kashmiri Perso-Arabic normalizer & tokenization
│   ├── accent_classifier.py  # Regional accent classifier (Kupwara, Bandipora, etc.)
│   ├── model.py              # Neural acoustic model with speaker embeddings
│   ├── train.py              # Mixed-precision training pipeline
│   ├── inference.py          # Speech synthesis engine
│   └── test_pipeline.py      # 9-step test verification suite
├── data/
│   ├── raw/                  # Downloaded raw audio corpora
│   ├── processed/            # Standardized 22,050 Hz audio and filelists
│   └── samples/              # Synthesized output samples
└── models/
    └── checkpoints/          # Saved model weights (.pt)
```

---

## 📜 License
MIT License.
