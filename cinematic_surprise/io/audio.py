"""
io/audio.py

Audio extractor for cinematic_surprise.

Extracts the audio track from a video file using ffmpeg, loads it into
memory with librosa, and provides 1-second segment access by index.

IN:  path to video file
OUT: get_segment(second_idx) → numpy array of ~22050 float32 samples

ffmpeg command used:
    ffmpeg -i <video> -vn -acodec pcm_s16le -ar 22050 -ac 1 <tmp.wav>
    -vn      : skip video stream
    -acodec  : uncompressed 16-bit PCM
    -ar      : resample to target sample rate
    -ac 1    : downmix to mono
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    log.warning("librosa not installed. Audio modality will be unavailable.")


class AudioExtractor:
    """
    Extracts and caches the full audio track from a video file.

    Requires ffmpeg on PATH. If ffmpeg or librosa are unavailable,
    get_segment() returns None and the pipeline skips audio gracefully.

    Args:
        video_path : Path to the video file
        sr         : Target sample rate in Hz (default 22050)
    """

    def __init__(self, video_path: str | Path, sr: int = 22050):
        self.video_path = Path(video_path)
        self.sr         = sr
        self._audio: Optional[np.ndarray] = None

        if _LIBROSA_OK:
            self._load()

    def _load(self) -> None:
        """Extract audio via ffmpeg → temporary WAV → librosa load."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(self.sr),
            "-ac", "1",
            tmp_path,
        ]

        try:
            subprocess.run(cmd, check=True)
            self._audio, _ = librosa.load(tmp_path, sr=self.sr, mono=True)
            log.info(
                f"Audio loaded: {len(self._audio) / self.sr:.1f}s "
                f"at {self.sr} Hz from '{self.video_path.name}'"
            )
        except subprocess.CalledProcessError as e:
            log.warning(f"ffmpeg failed ({e}). Audio modality disabled.")
        except Exception as e:
            log.warning(f"Audio load error ({e}). Audio modality disabled.")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @property
    def available(self) -> bool:
        return self._audio is not None

    @property
    def duration_s(self) -> float:
        return len(self._audio) / self.sr if self.available else 0.0

    def get_segment(self, second_idx: int) -> Optional[np.ndarray]:
        """
        Return the 1-second audio segment at second_idx.

        Args:
            second_idx : 0-based integer second index

        Returns:
            float32 array of length ~sr, or None if unavailable.
        """
        if not self.available:
            return None
        start = second_idx * self.sr
        end   = start + self.sr
        if start >= len(self._audio):
            return None
        segment = self._audio[start:end]
        # Pad if last second is shorter than sr samples
        if len(segment) < self.sr:
            segment = np.pad(segment, (0, self.sr - len(segment)))
        return segment.astype(np.float32)
