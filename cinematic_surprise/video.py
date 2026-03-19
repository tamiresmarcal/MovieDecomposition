"""
io/video.py

Frame-accurate video reader.

Design:
    - Opens one video file at a time (OpenCV VideoCapture)
    - Yields 1-second batches of frames as numpy arrays
    - Runs scene-cut detection on every frame via SceneCutDetector
    - Frames are held in memory for one second only, then discarded
    - Never writes frames to disk

IN:  path to video file (MP4, MKV, AVI, ...)
OUT: (second_idx, frames, has_cut) via iter_seconds()

    second_idx : int         — 0-based second index
    frames     : list[ndarray] — BGR frames in this second (~fps frames)
    has_cut    : bool        — True if any frame in this second triggered
                               the scene-cut threshold
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from cinematic_surprise.utils.scene_cut import SceneCutDetector

log = logging.getLogger(__name__)


class VideoReader:
    """
    Frame-accurate video reader that yields 1-second windows.

    Args:
        path            : Path to the video file
        max_seconds     : Stop after this many seconds (None = whole film)
        cut_threshold   : Chi-squared distance threshold for scene-cut detection
    """

    def __init__(
        self,
        path:          str | Path,
        max_seconds:   Optional[int] = None,
        cut_threshold: float = 0.15,
    ):
        self.path        = Path(path)
        self.max_seconds = max_seconds

        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open: {self.path}")

        self._detector = SceneCutDetector(threshold=cut_threshold)

        log.info(
            f"Opened '{self.path.name}' | "
            f"{self.fps:.2f} fps | "
            f"{self.n_frames} frames | "
            f"{self.duration_s:.1f}s"
        )

    # ── Video metadata ─────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def n_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_s(self) -> float:
        return self.n_frames / max(self.fps, 1.0)

    @property
    def n_seconds(self) -> int:
        return int(self.duration_s)

    # ── Core iterator ──────────────────────────────────────────────────────────

    def iter_seconds(self) -> Iterator[Tuple[int, List[np.ndarray], bool]]:
        """
        Yield one 1-second window at a time.

        The window contains all frames whose index falls within
        [second_idx * fps, (second_idx+1) * fps).

        Memory: at most one second of frames (~23 frames) is held at once.
        Frames are discarded after yielding.

        Yields:
            (second_idx, frames, has_cut)
        """
        fps_int     = max(1, round(self.fps))
        total_read  = 0
        second_idx  = 0
        max_frames  = (
            int(self.max_seconds * self.fps) if self.max_seconds else None
        )

        while True:
            frames: List[np.ndarray] = []
            has_cut = False

            for _ in range(fps_int):
                if max_frames and total_read >= max_frames:
                    break
                ok, frame = self._cap.read()
                if not ok:
                    break
                total_read += 1

                is_cut, _ = self._detector.update(frame)
                if is_cut:
                    has_cut = True

                frames.append(frame)

            if not frames:
                break

            yield second_idx, frames, has_cut
            second_idx += 1

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def release(self) -> None:
        self._cap.release()

    def reset_detector(self) -> None:
        """Reset scene-cut detector state. Call between films."""
        self._detector.reset()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoReader('{self.path.name}', "
            f"{self.fps:.1f} fps, {self.duration_s:.0f}s)"
        )
