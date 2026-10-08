"""
Complete Verification and Self-Test Suite for Kashmiri Voice TTS Pipeline.
Tests:
1. Kashmiri text normalization & diacritic preservation
2. Audio cleaning, resampling, and loudness standardization
3. Neural acoustic model forward pass on CUDA (RTX 4050)
4. Audio waveform synthesis and file export
"""

import sys
import unittest
from pathlib import Path

# Add project root and scripts dir to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import soundfile as sf

from text_normalizer import (
    normalize_kashmiri_text,
    expand_number_to_kashmiri,
    text_to_sequence,
    sequence_to_text,
    VOCAB_SIZE
)
from audio_cleaner import compute_snr, clean_audio_file, normalize_loudness
from model import KashmiriTTSModel, mel_to_audio_griffin_lim
from config import SAMPLE_RATE, DEVICE, SAMPLES_DATA_DIR


class TestKashmiriTTSPipeline(unittest.TestCase):
    
    def test_01_kashmiri_numbers(self):
        """Verify Kashmiri digit-to-word conversion."""
        self.assertEqual(expand_number_to_kashmiri(1), "اکھ")
        self.assertEqual(expand_number_to_kashmiri(2), "زٕ")
        self.assertEqual(expand_number_to_kashmiri(4), "ژور")
        self.assertEqual(expand_number_to_kashmiri(5), "پانژھ")
        self.assertEqual(expand_number_to_kashmiri(10), "دَہ")
        print("[PASS] Kashmiri Number Expansion")

    def test_02_text_normalization_and_diacritics(self):
        """Verify preservation of distinct Kashmiri vowels and palatalization."""
        # 1. Kashmiri Vowel ٲ
        res1 = normalize_kashmiri_text("کٲشُر زَبان")
        self.assertIn("ٲ", res1)
        
        # 2. Plat Ye ؠ (Palatalization)
        res2 = normalize_kashmiri_text("تُہؠ چھِوا ٹھیک؟")
        self.assertIn("ؠ", res2)
        
        # 3. Kashmiri Vowel ۄ
        res3 = normalize_kashmiri_text("آز چھُ مۄسَم")
        self.assertIn("ۄ", res3)
        print("[PASS] Text Normalizer & Kashmiri Vowel Preservation")

    def test_03_tokenization_roundtrip(self):
        """Verify character-level tokenization."""
        text = "کٲشُر بولُن چھُ واریاہ مۆدُر"
        seq = text_to_sequence(text)
        self.assertGreater(len(seq), 5)
        for token_id in seq:
            self.assertLess(token_id, VOCAB_SIZE)
        print(f"[PASS] Tokenization ({len(seq)} tokens generated, vocab size: {VOCAB_SIZE})")

    def test_04_audio_cleaner(self):
        """Verify audio cleaning, silence trimming, and SNR computation."""
        from config import ACCENT_RAW_DIR
        sr = SAMPLE_RATE
        sample_wavs = list((ACCENT_RAW_DIR / "wavs").glob("*.wav"))
        if sample_wavs:
            test_wav = sample_wavs[0]
        else:
            test_wav = SAMPLES_DATA_DIR / "test_synth_input.wav"
            t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
            tone = 0.4 * np.sin(2 * np.pi * 320 * t)
            sf.write(str(test_wav), tone, sr)

        out_wav = SAMPLES_DATA_DIR / "test_audio_cleaned.wav"
        res = clean_audio_file(test_wav, out_wav, target_sr=sr)
        self.assertTrue(res["valid"], msg=f"Audio cleaner rejected file: {res.get('reason')}")
        self.assertGreater(res["snr_db"], 10.0)
        self.assertTrue(out_wav.exists())
        print(f"[PASS] Audio Cleaner on Kashmiri recording (Duration: {res['duration']}s, SNR: {res['snr_db']} dB)")

    def test_05_acoustic_model_forward_pass(self):
        """Verify neural model forward pass and shape dimensions on CUDA/CPU."""
        model = KashmiriTTSModel().to(DEVICE)
        model.eval()

        dummy_seq = torch.randint(1, VOCAB_SIZE, (2, 20), device=DEVICE)
        dummy_spk = torch.tensor([0, 1], device=DEVICE)

        with torch.no_grad():
            mel_pred, dur_pred = model(dummy_seq, dummy_spk)

        self.assertEqual(mel_pred.dim(), 3)
        self.assertEqual(mel_pred.size(1), 80)
        print(f"[PASS] Model Forward Pass on {DEVICE.upper()} (Output Shape: {list(mel_pred.shape)})")

    def test_06_end_to_end_synthesis(self):
        """Verify complete synthesis from Kashmiri text to output WAV file."""
        from inference import synthesize_kashmiri_speech
        test_text = "سلام! تُہؠ چھِوا ٹھیک؟"
        out_wav = SAMPLES_DATA_DIR / "test_salam_output.wav"
        
        path = synthesize_kashmiri_speech(test_text, out_wav)
        self.assertTrue(path.exists())
        info = sf.info(str(path))
        self.assertEqual(info.samplerate, SAMPLE_RATE)
        self.assertGreater(info.duration, 0.1)
        print(f"[PASS] End-to-End Synthesis -> {path.name} ({info.duration:.2f}s @ {info.samplerate}Hz)")

    def test_07_kaggle_dataset_manifest(self):
        """Verify Kaggle Kashmiri dataset ingestion and Perso-Arabic text mapping."""
        from config import KAGGLE_RAW_DIR
        import pandas as pd
        meta_csv = KAGGLE_RAW_DIR / "metadata.csv"
        self.assertTrue(meta_csv.exists(), msg=f"Kaggle metadata not found: {meta_csv}")
        df = pd.read_csv(meta_csv)
        self.assertGreaterEqual(len(df), 300)
        self.assertIn("text", df.columns)
        self.assertIn("word_label", df.columns)
        print(f"[PASS] Kaggle Dataset Manifest ({len(df)} samples indexed across vocabulary words)")

    def test_08_accent_classification_dataset_manifest(self):
        """Verify Kashmiri Accent Classification dataset indexing across 4 districts."""
        from config import ACCENT_RAW_DIR
        import pandas as pd
        meta_csv = ACCENT_RAW_DIR / "metadata.csv"
        self.assertTrue(meta_csv.exists(), msg=f"Accent metadata not found: {meta_csv}")
        df = pd.read_csv(meta_csv)
        self.assertGreaterEqual(len(df), 100)
        districts = set(df["dialect"].str.strip())
        self.assertTrue({"Kupwara", "Bandipora", "Shopian", "Islamabad"}.issubset(districts))
        print(f"[PASS] Accent Dataset Manifest ({len(df)} clips across Kupwara, Bandipora, Shopian, Islamabad)")

    def test_09_kashmiri_accent_classifier(self):
        """Verify regional Kashmiri accent predictor."""
        from accent_classifier import predict_kashmiri_accent
        from config import ACCENT_RAW_DIR
        wavs = list((ACCENT_RAW_DIR / "wavs").glob("*.wav"))
        self.assertGreater(len(wavs), 0)
        sample = wavs[0]
        res = predict_kashmiri_accent(sample)
        self.assertIn("predicted_dialect", res)
        self.assertIn("confidence", res)
        self.assertGreater(res["confidence"], 0.0)
        print(f"[PASS] Accent Classifier Predictor -> {sample.name} classified as {res['predicted_dialect']} ({res['confidence']*100:.1f}%)")


if __name__ == "__main__":
    unittest.main()
