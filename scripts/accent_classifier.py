"""
Kashmiri Regional Accent and Dialect Classifier.
Based on research from shehzensidiq/kashmiri-accent-classification.
Extracts acoustic prosody, MFCC, and spectral features to classify Kashmiri speech
into regional dialect clusters:
1. Kupwara (Northern Kashmir / Kamraz)
2. Bandipora (Northern / Wular valley)
3. Shopian (Southern Kashmir / Maraz)
4. Islamabad (Anantnag / Southern Kashmir / Maraz)
"""

import sys
import os
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

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from config import ACCENT_RAW_DIR, KASHMIRI_DIALECTS, DEVICE, MODELS_DIR

DIALECT_CLASSES = ["Bandipora", "Islamabad", "Kupwara", "Shopian"]
DIALECT_TO_ID = {name: i for i, name in enumerate(DIALECT_CLASSES)}
ID_TO_DIALECT = {i: name for i, name in enumerate(DIALECT_CLASSES)}


def extract_accent_features(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extracts comprehensive acoustic and spectral prosody features
    matching the feature representation from shehzensidiq/kashmiri-accent-classification:
    MFCCs, Chroma STFT, RMS energy, Spectral Centroid, Bandwidth, Rolloff, ZCR.
    """
    if len(audio) < sr * 0.5:
        # Pad short clips
        audio = np.pad(audio, (0, int(sr * 0.5) - len(audio)))

    features = []

    # 1. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend([np.mean(chroma), np.var(chroma)])

    # 2. RMS Energy
    rms = librosa.feature.rms(y=audio)
    features.extend([np.mean(rms), np.var(rms)])

    # 3. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.extend([np.mean(spec_cent), np.var(spec_cent)])

    # 4. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.extend([np.mean(spec_bw), np.var(spec_bw)])

    # 5. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.extend([np.mean(rolloff), np.var(rolloff)])

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.extend([np.mean(zcr), np.var(zcr)])

    # 7. MFCCs (13 coefficients: mean and variance)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features.extend([np.mean(mfcc[i]), np.var(mfcc[i])])

    # 8. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.extend([np.mean(contrast), np.var(contrast)])

    return np.array(features, dtype=np.float32)


class KashmiriAccentMLP(nn.Module):
    """Deep Neural Classifier for Kashmiri Regional Accent Identification."""
    def __init__(self, in_features: int = 40, num_classes: int = len(DIALECT_CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_accent_classifier(epochs: int = 30) -> Path:
    """Trains an accent classifier on the indexed kashmiri-accent-classification dataset."""
    metadata_csv = ACCENT_RAW_DIR / "metadata.csv"
    wavs_dir = ACCENT_RAW_DIR / "wavs"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Accent metadata not found: {metadata_csv}. Run download_dataset.py --source accent first.")

    df = pd.read_csv(metadata_csv)
    print(f"[+] Loading {len(df)} accent audio recordings for classifier training...")

    X = []
    y = []

    for _, row in df.iterrows():
        audio_name = str(row["audio_file"])
        dialect = str(row["dialect"]).strip()
        if dialect not in DIALECT_TO_ID:
            continue

        wav_path = wavs_dir / audio_name if not Path(audio_name).is_absolute() else Path(audio_name)
        if not wav_path.exists():
            continue

        try:
            audio, sr = librosa.load(str(wav_path), sr=22050, mono=True)
            feat = extract_accent_features(audio, sr=sr)
            X.append(feat)
            y.append(DIALECT_TO_ID[dialect])
        except Exception:
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Normalize features
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-6
    X_norm = (X - mean) / std

    in_dim = X.shape[1]
    model = KashmiriAccentMLP(in_features=in_dim, num_classes=len(DIALECT_CLASSES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)

    model.train()
    print(f"[+] Training Kashmiri Accent Classifier ({len(X)} samples, {epochs} epochs)...")
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_t).float().mean().item() * 100.0
            print(f" • Epoch {ep+1:02d}/{epochs:02d} - Loss: {loss.item():.4f} - Training Accuracy: {acc:.1f}%")

    # Save model and normalization stats
    save_path = MODELS_DIR / "kashmiri_accent_classifier.pt"
    torch.save({
        "model_state": model.state_dict(),
        "in_features": in_dim,
        "mean": mean,
        "std": std,
        "classes": DIALECT_CLASSES
    }, save_path)

    print(f"[OK] Trained accent classifier saved to: {save_path}\n")
    return save_path


def predict_kashmiri_accent(audio_path: Path) -> Dict[str, Any]:
    """
    Predicts the regional Kashmiri dialect/accent of any audio clip.
    Returns predicted district, dialect zone, and confidence probabilities.
    """
    model_path = MODELS_DIR / "kashmiri_accent_classifier.pt"
    if not model_path.exists():
        # Quick train if checkpoint not present
        train_accent_classifier(epochs=20)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = KashmiriAccentMLP(in_features=checkpoint["in_features"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    audio, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    feat = extract_accent_features(audio, sr=sr)
    feat_norm = (feat - checkpoint["mean"]) / checkpoint["std"]

    with torch.no_grad():
        x = torch.tensor(feat_norm, dtype=torch.float32, device=DEVICE)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    best_idx = int(np.argmax(probs))
    best_dialect = DIALECT_CLASSES[best_idx]
    dialect_meta = KASHMIRI_DIALECTS.get(best_dialect.lower(), {})

    return {
        "audio_file": str(audio_path),
        "predicted_dialect": best_dialect,
        "dialect_zone": dialect_meta.get("dialect_zone", "Kashmiri"),
        "confidence": round(float(probs[best_idx]), 4),
        "probabilities": {DIALECT_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(DIALECT_CLASSES))}
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kashmiri Accent Classifier")
    parser.add_argument("--train", action="store_true", help="Train accent classifier")
    parser.add_argument("--predict", type=str, default=None, help="Path to audio file to classify")
    args = parser.parse_args()

    if args.train:
        train_accent_classifier(epochs=30)
    elif args.predict:
        res = predict_kashmiri_accent(Path(args.predict))
        print("\n--- Kashmiri Accent Classification Result ---")
        print(f" Audio File       : {res['audio_file']}")
        print(f" Predicted Accent : {res['predicted_dialect']} ({res['dialect_zone']})")
        print(f" Confidence       : {res['confidence']*100:.1f}%")
        print(" Class Probabilities:")
        for k, v in res["probabilities"].items():
            print(f"  • {k:10s}: {v*100:.1f}%")
        print("----------------------------------------------\n")
    else:
        # Self-test training
        train_accent_classifier(epochs=25)
