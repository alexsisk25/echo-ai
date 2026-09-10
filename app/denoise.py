"""Optional speech enhancement before diarization.

Restaurant and street recordings can defeat diarization: background
noise blurs the acoustic difference between voices and the clustering
collapses two people into one. Running a speech-enhancement pass first
sometimes fixes that. It can also hurt, by smearing the very timbre
differences the diarizer relies on, so this is off by default and only
ever runs when a recording is reprocessed with the toggle on.

The model is MetricGAN+ from SpeechBrain, which is already a dependency
(the voice fingerprinter uses its ECAPA encoder), so no new package is
needed. Weights download from HuggingFace on first use like the other
models. Nothing here touches the original file: enhancement writes a
temporary wav that the caller passes to diarization and then deletes.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from app import config

log = logging.getLogger("otter.denoise")

MODEL = "speechbrain/metricgan-plus-voicebank"
SAMPLE_RATE = 16000

_enhancer = None


def _get_enhancer():
    global _enhancer
    if _enhancer is None:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        _enhancer = SpectralMaskEnhancement.from_hparams(
            source=MODEL,
            savedir=str(config.DATA_DIR / "models" / "metricgan"),
        )
    return _enhancer


def enhance(audio_path: str | Path) -> Path | None:
    """Write a denoised 16 kHz mono copy and return its path.

    Returns None when enhancement is unavailable or fails, so callers
    fall back to the original audio instead of losing the run.
    """
    src = Path(audio_path)
    wav = None
    out = None
    try:
        import torch
        import torchaudio

        # Decode to the mono 16 kHz the model expects.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            wav = t.name
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-ar", str(SAMPLE_RATE), "-ac", "1", wav],
            check=True, capture_output=True, text=True,
        )

        enhancer = _get_enhancer()
        noisy = enhancer.load_audio(wav).unsqueeze(0)
        cleaned = enhancer.enhance_batch(
            noisy, lengths=torch.tensor([1.0]))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            out = t.name
        torchaudio.save(out, cleaned.cpu(), SAMPLE_RATE)
        log.info("Denoised %s for diarization", src.name)
        return Path(out)
    except Exception:
        log.exception("Speech enhancement failed for %s; using the "
                      "original audio", src.name)
        if out:
            try:
                os.unlink(out)
            except OSError:
                pass
        return None
    finally:
        if wav:
            try:
                os.unlink(wav)
            except OSError:
                pass
