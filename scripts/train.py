"""
Training and Fine-Tuning Pipeline for Kashmiri TTS.
Specifically optimized for NVIDIA GeForce RTX 4050 (6GB VRAM):
- FP16 Mixed Precision
- Gradient Accumulation (effective batch size 32)
- Dynamic padding & memory-efficient CUDA cache clearing
"""

import sys
import time
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

from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from tqdm import tqdm
from text_normalizer import text_to_sequence
from model import KashmiriTTSModel
from config import (
    PROCESSED_DATA_DIR,
    CHECKPOINTS_DIR,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    DEVICE,
    FP16_MIXED,
    MAX_EPOCHS,
    SPEAKER_ID_MAP
)


class KashmiriTTSDataset(Dataset):
    """Loads standardized Kashmiri audio-transcript pairs."""
    def __init__(self, filelist_path: Path):
        self.samples = []
        if filelist_path.exists():
            with open(filelist_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 3:
                        audio_path, spk_id, text = parts[0], parts[1], parts[2]
                        if Path(audio_path).exists():
                            self.samples.append((audio_path, spk_id, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        audio_path, spk_id, text = self.samples[idx]

        # 1. Text to token sequence
        seq = text_to_sequence(text)
        text_tensor = torch.tensor(seq, dtype=torch.long)

        # 2. Extract Mel Spectrogram
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            fmin=0.0,
            fmax=8000.0,
            power=1.0
        )
        # Log compression
        log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        mel_tensor = torch.tensor(log_mel, dtype=torch.float32)

        # Speaker ID index from configuration
        spk_idx = SPEAKER_ID_MAP.get(spk_id, 0)

        return text_tensor, mel_tensor, spk_idx


def collate_fn(batch):
    """Pads variable-length text sequences and mel spectrograms."""
    text_seqs, mels, spk_ids = zip(*batch)

    # Pad text sequences
    text_lengths = torch.tensor([len(t) for t in text_seqs], dtype=torch.long)
    max_text_len = max(1, text_lengths.max().item())
    padded_texts = torch.zeros(len(text_seqs), max_text_len, dtype=torch.long)
    for i, t in enumerate(text_seqs):
        padded_texts[i, :len(t)] = t

    # Pad Mel Spectrograms
    mel_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    max_mel_len = max(1, mel_lengths.max().item())
    padded_mels = torch.zeros(len(mels), N_MELS, max_mel_len, dtype=torch.float32)
    for i, m in enumerate(mels):
        padded_mels[i, :, :m.shape[1]] = m

    spk_tensor = torch.tensor(spk_ids, dtype=torch.long)
    return padded_texts, padded_mels, spk_tensor, text_lengths, mel_lengths


def train_kashmiri_tts(
    epochs: int = 15,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE
):
    """Runs fine-tuning loop on Kashmiri speech dataset."""
    print(f"\n{'='*70}")
    print(f" TRAINING KASHMIRI TTS MODEL")
    print(f" Target Device: {DEVICE.upper()} (NVIDIA RTX 4050 6GB Optimized)")
    print(f" Mixed Precision: {FP16_MIXED} | Grad Accumulation: {GRADIENT_ACCUMULATION}")
    print(f"{'='*70}")

    train_file = PROCESSED_DATA_DIR / "train.txt"
    val_file = PROCESSED_DATA_DIR / "val.txt"

    if not train_file.exists():
        print(f"[-] Training split not found: {train_file}")
        print("    Please run `python scripts/prepare_tts_data.py` first.")
        return

    train_dataset = KashmiriTTSDataset(train_file)
    val_dataset = KashmiriTTSDataset(val_file)

    if len(train_dataset) == 0:
        print("[-] Training dataset is empty. Check data paths.")
        return

    print(f"[+] Loaded {len(train_dataset)} training utterances, {len(val_dataset)} validation utterances.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows-friendly
        pin_memory=(DEVICE == "cuda")
    )

    # Initialize model
    model = KashmiriTTSModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=FP16_MIXED)
    mel_criterion = nn.L1Loss()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, (texts, mels, spks, text_lens, mel_lens) in enumerate(pbar):
            texts = texts.to(DEVICE)
            mels = mels.to(DEVICE)
            spks = spks.to(DEVICE)
            mel_lens = mel_lens.to(DEVICE)

            with torch.amp.autocast('cuda', enabled=FP16_MIXED):
                pred_mel, _ = model(texts, spks, target_lengths=mel_lens)
                # Align lengths for loss computation
                min_len = min(pred_mel.size(-1), mels.size(-1))
                loss = mel_criterion(pred_mel[..., :min_len], mels[..., :min_len])
                scaled_loss = loss / GRADIENT_ACCUMULATION

            scaler.scale(scaled_loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix({"L1 Mel Loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"[✓] Epoch {epoch} Complete - Average Mel Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = CHECKPOINTS_DIR / f"kashmiri_tts_epoch_{epoch:03d}.pt"
        best_ckpt_path = CHECKPOINTS_DIR / "kashmiri_tts_best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss
            }, best_ckpt_path)
            print(f"    ⭐ Saved new best checkpoint to: {best_ckpt_path.name}")

    print(f"\n{'='*70}")
    print(f"[✓] Training completed successfully!")
    print(f"    Checkpoints stored in: {CHECKPOINTS_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Kashmiri TTS Acoustic Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args, unknown = parser.parse_known_args()
    if unknown and unknown[0].isdigit():
        epochs = int(unknown[0])
    else:
        epochs = args.epochs
    train_kashmiri_tts(epochs=epochs, learning_rate=args.lr)
