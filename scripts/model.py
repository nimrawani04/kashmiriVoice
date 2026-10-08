"""
Kashmiri Neural TTS Acoustic Architecture.
Character-level encoder for Perso-Arabic script paired with a non-autoregressive
Mel-Spectrogram generator and HiFi-GAN/Griffin-Lim vocoder synthesis pipeline.
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from text_normalizer import VOCAB_SIZE, CHAR_TO_ID
from config import (
    N_MELS,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    DEVICE
)


class ConvBlock(nn.Module):
    """Gated 1D Convolutional Residual Block with Layer Normalization."""
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        out = self.conv(x)
        # Gated activation
        a, b = torch.chunk(out, 2, dim=1)
        out = a * torch.sigmoid(b)
        out = out + residual
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(out)
        return out


class KashmiriTextEncoder(nn.Module):
    """
    Character-level Encoder designed specifically for Kashmiri Perso-Arabic script.
    Encodes unique Kashmiri diacritics, palatalization glides, and central vowels.
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 192,
        hidden_dim: int = 192,
        num_layers: int = 4
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pre_conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ConvBlock(hidden_dim, kernel_size=5, dropout=0.1) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_seq: torch.Tensor) -> torch.Tensor:
        # text_seq: [B, T_text]
        x = self.embedding(text_seq)  # [B, T_text, embed_dim]
        x = x.transpose(1, 2)         # [B, embed_dim, T_text]
        x = self.pre_conv(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)         # [B, T_text, hidden_dim]
        x = self.proj(x)
        return x


class DurationPredictor(nn.Module):
    """Predicts phoneme/character durations for non-autoregressive speech synthesis."""
    def __init__(self, in_dim: int = 192, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]
        h = F.relu(self.conv1(x.transpose(1, 2)))
        h = F.relu(self.conv2(h))
        durations = self.proj(h.transpose(1, 2)).squeeze(-1)
        return F.relu(durations)  # Non-negative durations


class KashmiriTTSModel(nn.Module):
    """
    Complete Kashmiri TTS Model.
    Supports multi-speaker conditioning and generates 80-band mel spectrograms.
    """
    def __init__(
        self,
        num_speakers: int = 16,
        hidden_dim: int = 192,
        speaker_dim: int = 64,
        n_mels: int = N_MELS
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim
        self.n_mels = n_mels

        # Linguistic character encoder
        self.encoder = KashmiriTextEncoder(vocab_size=VOCAB_SIZE, hidden_dim=hidden_dim)

        # Multi-speaker conditioning
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_dim)

        # Duration predictor
        self.duration_predictor = DurationPredictor(in_dim=hidden_dim)

        # Mel Decoder
        total_hidden = hidden_dim + speaker_dim
        self.decoder_blocks = nn.ModuleList([
            ConvBlock(total_hidden, kernel_size=5, dropout=0.1) for _ in range(4)
        ])
        self.mel_proj = nn.Conv1d(total_hidden, n_mels, kernel_size=1)

    def forward(
        self,
        text_seq: torch.Tensor,
        speaker_id: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = text_seq.size(0)

        # Encode Kashmiri script
        enc_out = self.encoder(text_seq)  # [B, T_text, hidden_dim]

        # Speaker conditioning
        if speaker_id is None:
            spk_emb = torch.zeros(batch_size, self.speaker_dim, device=text_seq.device)
        else:
            spk_emb = self.speaker_embedding(speaker_id)  # [B, speaker_dim]

        # Duration prediction
        pred_durations = self.duration_predictor(enc_out)  # [B, T_text]

        # Expand character representations over time
        if target_lengths is not None:
            # Interpolate to match target mel frame length during training
            target_len = target_lengths.max().item()
            expanded = F.interpolate(
                enc_out.transpose(1, 2),
                size=target_len,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)
        else:
            # Inference: expand using predicted durations
            dur_scale = 3.5  # average frames per token at 22050Hz/256 hop
            total_frames = max(16, int(pred_durations.sum().item() * dur_scale))
            expanded = F.interpolate(
                enc_out.transpose(1, 2),
                size=total_frames,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

        # Concat speaker embedding across all time frames
        spk_expanded = spk_emb.unsqueeze(1).expand(-1, expanded.size(1), -1)
        combined = torch.cat([expanded, spk_expanded], dim=-1).transpose(1, 2)

        # Decode into Mel Spectrogram
        h = combined
        for block in self.decoder_blocks:
            h = block(h)

        mel_pred = self.mel_proj(h)  # [B, n_mels, T_mel]
        return mel_pred, pred_durations

    @torch.no_grad()
    def synthesize_mel(self, text_seq: torch.Tensor, speaker_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Runs model inference to generate a mel-spectrogram."""
        self.eval()
        mel_pred, _ = self.forward(text_seq, speaker_id=speaker_id)
        return mel_pred


def mel_to_audio_griffin_lim(mel: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fast, reliable audio synthesis from predicted Mel-spectrogram via Griffin-Lim."""
    # Invert mel to linear STFT
    linear_stft = librosa.feature.inverse.mel_to_stft(
        mel,
        sr=sr,
        n_fft=N_FFT,
        fmin=0.0,
        fmax=8000.0,
        power=1.0
    )
    # Reconstruct phase
    audio = librosa.griffinlim(linear_stft, n_iter=32, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    # Normalize peak amplitude
    peak = np.max(np.abs(audio))
    if peak > 0.01:
        audio = audio / peak * 0.90
    return audio
