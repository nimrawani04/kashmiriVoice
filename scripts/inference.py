"""
Kashmiri Voice Synthesis Inference Engine.
Takes written Kashmiri (Perso-Arabic script) and synthesizes audio WAV file.
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

import torch
import soundfile as sf
import numpy as np
from text_normalizer import normalize_kashmiri_text, text_to_sequence
from model import KashmiriTTSModel, mel_to_audio_griffin_lim
from config import (
    CHECKPOINTS_DIR,
    SAMPLES_DATA_DIR,
    SAMPLE_RATE,
    DEVICE,
    SPEAKER_ID_MAP,
    KASHMIRI_DIALECTS
)

# Standard benchmark sentences representing authentic Kashmiri phonology & vowels
BENCHMARK_SENTENCES = [
    ("salam", "سلام! تُہؠ چھِوا ٹھیک؟", "Hello! Are you doing well?"),
    ("sweet_voice", "کٲشُر بولُن چھُ واریاہ مۆدُر۔", "Speaking Kashmiri is very sweet."),
    ("kashmir_love", "میٚہ چھُ کٲشُر زَبان پَسَند۔", "I love the Kashmiri language."),
    ("weather", "آز چھُ مۄسَم واریاہ خوبصوٗرَتھ۔", "Today the weather is very beautiful."),
    ("counting", "اکھ، زٕ، ترےٚ، ژور، پانژھ۔", "One, two, three, four, five.")
]


def load_kashmiri_tts_model(checkpoint_path: Path = None) -> KashmiriTTSModel:
    """Loads TTS model weights from checkpoint or initializes baseline model."""
    model = KashmiriTTSModel().to(DEVICE)
    
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / "kashmiri_tts_best.pt"

    if checkpoint_path.exists():
        print(f"[+] Loading trained checkpoint: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print("[✓] Model weights successfully loaded.")
        except Exception as e:
            print(f"[!] Warning: Could not load checkpoint weights ({e}). Running baseline.")
    else:
        print(f"[i] No trained checkpoint found at {checkpoint_path}. Using base acoustic model.")

    model.eval()
    return model


def synthesize_kashmiri_speech(
    text: str,
    output_path: Path,
    model: KashmiriTTSModel = None,
    speaker_id: int = 0
) -> Path:
    """
    Synthesizes speech from written Kashmiri text.
    1. Normalizes Perso-Arabic text (preserving ٲ, ۄ, ۆ, ؠ)
    2. Encodes to token IDs
    3. Predicts mel spectrogram via neural model
    4. Inverts to 22.05kHz audio waveform
    """
    if model is None:
        model = load_kashmiri_tts_model()

    # Step 1: Normalize Kashmiri text
    normalized_text = normalize_kashmiri_text(text)
    if not normalized_text:
        raise ValueError("Input text is empty after normalization.")

    # Step 2: Convert to token sequence
    seq = text_to_sequence(normalized_text)
    seq_tensor = torch.tensor([seq], dtype=torch.long, device=DEVICE)
    spk_tensor = torch.tensor([speaker_id], dtype=torch.long, device=DEVICE)

    # Step 3: Run acoustic model
    with torch.no_grad():
        mel = model.synthesize_mel(seq_tensor, spk_tensor)
        mel_np = mel.squeeze(0).cpu().numpy()

    # Step 4: Reconstruct waveform
    audio_wav = mel_to_audio_griffin_lim(mel_np, sr=SAMPLE_RATE)

    # Step 5: Save output audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_wav, SAMPLE_RATE)
    
    duration = len(audio_wav) / SAMPLE_RATE
    print(f"[+] Synthesized {duration:.2f}s audio to: {output_path}")
    return output_path


def run_benchmark_synthesis(output_dir: Path = SAMPLES_DATA_DIR):
    """Synthesizes all standard benchmark Kashmiri sentences."""
    print(f"\n{'='*70}")
    print(f" GENERATING KASHMIRI BENCHMARK AUDIO SAMPLES")
    print(f"{'='*70}")
    model = load_kashmiri_tts_model()

    for tag, text, translation in BENCHMARK_SENTENCES:
        out_file = output_dir / f"sample_{tag}.wav"
        print(f"\n• Text (کٲشُر) : {text}")
        print(f"  English Meaning: {translation}")
        synthesize_kashmiri_speech(text, out_file, model=model)

    print(f"\n[✓] All benchmark samples generated in: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kashmiri Text-to-Speech Inference CLI")
    parser.add_argument("--text", type=str, default=None, help="Kashmiri text to synthesize (Perso-Arabic script)")
    parser.add_argument("--output", type=str, default=str(SAMPLES_DATA_DIR / "output.wav"), help="Output WAV path")
    parser.add_argument(
        "--dialect",
        type=str,
        default="default",
        choices=["default", "kupwara", "bandipora", "shopian", "islamabad", "kaggle", "hf"],
        help="Regional Kashmiri dialect / accent voice"
    )
    parser.add_argument("--speaker_id", type=int, default=None, help="Raw speaker ID index (0-15)")
    parser.add_argument("--benchmark", action="store_true", help="Generate all benchmark Kashmiri audio samples")

    args = parser.parse_args()

    # Resolve speaker ID
    if args.speaker_id is not None:
        spk_idx = args.speaker_id
    elif args.dialect != "default":
        dialect_key = f"spk_{args.dialect}" if f"spk_{args.dialect}" in SPEAKER_ID_MAP else (
            f"{args.dialect}_speaker" if f"{args.dialect}_speaker" in SPEAKER_ID_MAP else "hf_speaker"
        )
        spk_idx = SPEAKER_ID_MAP.get(dialect_key, 0)
    else:
        spk_idx = 0

    if args.benchmark or args.text is None:
        run_benchmark_synthesis()
    else:
        synthesize_kashmiri_speech(args.text, Path(args.output), speaker_id=spk_idx)
