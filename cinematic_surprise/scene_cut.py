"""
utils/scene_cut.py

Chi-squared colour histogram scene-cut detector.

IN:  two consecutive BGR frames (numpy arrays)
OUT: bool — is this a scene cut?

Algorithm:
    Compute a 3D colour histogram (16 bins per BGR channel) for each frame.
    Compute the chi-squared distance between consecutive histograms.
    Return True if distance exceeds the threshold in config.

Chi-squared distance:
    χ²(H1, H2) = Σ (H1_i - H2_i)² / (H1_i + H2_i + ε)

This is a standard method for video shot detection (Smeaton et al. 2010).
A colour histogram is more robust to minor motion and illumination changes
than raw pixel difference.

Note: no threshold is stored in this file — the caller (io/video.py)
decides whether to flag a cut based on the returned distance value.
"""

from __future__ import annotations

import cv2
import numpy as np


# Histogram parameters
_BINS      = [16, 16, 16]    # bins per BGR channel
_RANGES    = [0, 256, 0, 256, 0, 256]
_CHANNELS  = [0, 1, 2]


def frame_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Compute a normalised 3D colour histogram for a single frame.

    Args:
        frame_bgr : (H, W, 3) uint8 BGR frame

    Returns:
        Normalised histogram as a flat float32 array, shape (16*16*16,) = (4096,)
    """
    hist = cv2.calcHist(
        [frame_bgr], _CHANNELS, None, _BINS, _RANGES
    ).ravel().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def chi_squared_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Chi-squared distance between two normalised histograms.

    χ²(H1, H2) = Σ (H1_i - H2_i)² / (H1_i + H2_i + ε)

    Returns:
        float ≥ 0. Higher = more different. 0 = identical histograms.
    """
    denom = h1 + h2 + 1e-9
    return float(np.sum((h1 - h2) ** 2 / denom))


class SceneCutDetector:
    """
    Stateful scene-cut detector.

    Stores the previous frame's histogram internally.
    Call update(frame) for each frame; returns the chi-squared distance
    to the previous frame. Returns 0.0 on the first call.

    Args:
        threshold : Chi-squared distance above which a cut is flagged.
                    Default 0.15 is empirically tuned for Hollywood films.
    """

    def __init__(self, threshold: float = 0.15):
        self.threshold   = threshold
        self._prev_hist: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Process one frame.

        Args:
            frame_bgr : (H, W, 3) uint8 BGR frame

        Returns:
            (is_cut, distance)
            is_cut   : True if chi-squared distance exceeds threshold
            distance : Raw chi-squared distance (stored in output for inspection)
        """
        hist = frame_histogram(frame_bgr)

        if self._prev_hist is None:
            self._prev_hist = hist
            return False, 0.0

        dist = chi_squared_distance(self._prev_hist, hist)
        self._prev_hist = hist
        is_cut = dist > self.threshold
        return is_cut, dist

    def reset(self) -> None:
        """Reset detector state. Call between films."""
        self._prev_hist = None
