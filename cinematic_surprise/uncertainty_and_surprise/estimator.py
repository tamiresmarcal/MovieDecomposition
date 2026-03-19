"""
uncertainty_and_surprise/estimator.py

Core computational primitive used by every modality channel.

Three operations, always in this order per observation:
    1. EMA   → produces prior and posterior Gaussians
    2. KL    → measures surprise (posterior vs prior)
    3. H     → measures uncertainty (entropy of prior, before update)

Theory
------
Surprise  = KL(posterior ∥ prior)           [Itti & Baldi 2009, Eq. 4]
Uncertainty = H(prior) ∝ mean_d[log(σ²_prior)]  [Cheung et al. 2019]
Interaction = z(surprise) × z(uncertainty)  [Cheung et al. 2019, Table 1]

The interaction is computed post-hoc by the aggregator after z-scoring,
not here. This class returns raw KL and raw H per observation.

References
----------
Itti, L., & Baldi, P. (2009). Bayesian surprise attracts human attention.
    Vision Research, 49(10), 1295-1306.

Cheung, V.K.M., et al. (2019). Uncertainty and Surprise Jointly Predict
    Musical Pleasure and Amygdala, Hippocampus, and Auditory Cortex Activity.
    Current Biology, 29, 4084-4092.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from cinematic_surprise.config import ALPHA, CHANNELS, EMA_EPSILON


class OnlineGaussianEstimator:
    """
    Per-channel online Bayesian estimator.

    Maintains one diagonal Gaussian belief (μ, σ²) per named channel.
    On each observation:
        1. Compute uncertainty H from the CURRENT prior (before update)
        2. Update prior → posterior via EMA
        3. Compute surprise KL(posterior ∥ prior)
        4. Set prior ← posterior

    Args:
        alpha   : Dict mapping channel name → EMA learning rate.
                  Defaults to config.ALPHA if not provided.
        epsilon : Variance floor for numerical stability.
    """

    def __init__(
        self,
        alpha:   Optional[Dict[str, float]] = None,
        epsilon: float = EMA_EPSILON,
    ):
        self.alpha   = alpha if alpha is not None else dict(ALPHA)
        self.epsilon = epsilon
        # channel → (mu, var) both shape (d,)
        self._beliefs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, features: np.ndarray, channel: str) -> Tuple[float, float]:
        """
        Process one observation for a channel.

        Args:
            features : Feature array, any shape. Flattened to 1-D internally.
            channel  : Channel name (must be a key in self.alpha).

        Returns:
            (surprise, uncertainty)
            surprise    : KL(posterior ∥ prior) in nats/dimension. 0.0 on first call.
            uncertainty : H(prior) ∝ mean[log(σ²_prior)] in nats/dimension.
                          0.0 on first call (prior not yet meaningful).
        """
        x = np.asarray(features, dtype=np.float64).ravel()
        a = self._get_alpha(channel)

        if channel not in self._beliefs:
            # First observation: bootstrap belief, no surprise or uncertainty yet.
            # Variance initialised to 1.0 (not epsilon) so that the second
            # observation with the same value produces near-zero surprise.
            # Initialising to epsilon would cause KL(2ε ∥ ε) ≈ 0.15 on the
            # second identical observation — a spurious warmup artefact.
            self._beliefs[channel] = (
                x.copy(),
                np.ones_like(x, dtype=np.float64),
            )
            return 0.0, 0.0

        mu_prior, var_prior = self._beliefs[channel]

        # ── Step 1: uncertainty from prior BEFORE update ───────────────────
        uncertainty = self._entropy(var_prior)

        # ── Step 2: EMA posterior update ──────────────────────────────────
        mu_post  = a * x + (1.0 - a) * mu_prior
        var_post = (
            a * (x - mu_post) ** 2
            + (1.0 - a) * var_prior
            + self.epsilon
        )

        # ── Step 3: surprise = KL(posterior ∥ prior) ──────────────────────
        surprise = self._kl_diagonal_gaussian(mu_post, var_post, mu_prior, var_prior)

        # ── Step 4: advance prior ──────────────────────────────────────────
        self._beliefs[channel] = (mu_post, var_post)

        return float(surprise), float(uncertainty)

    def reset(self, channel: Optional[str] = None) -> None:
        """
        Reset beliefs. Call before processing a new film.

        Args:
            channel : Name of channel to reset, or None to reset all.
        """
        if channel is None:
            self._beliefs.clear()
        elif channel in self._beliefs:
            del self._beliefs[channel]

    def half_life_frames(self, channel: str) -> float:
        """Return the EMA half-life in frames for a given channel."""
        a = self._get_alpha(channel)
        return math.log(0.5) / math.log(1.0 - a)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_alpha(self, channel: str) -> float:
        if channel not in self.alpha:
            raise KeyError(
                f"Channel '{channel}' not found in alpha config. "
                f"Available channels: {list(self.alpha.keys())}"
            )
        return self.alpha[channel]

    @staticmethod
    def _kl_diagonal_gaussian(
        mu_p: np.ndarray,
        var_p: np.ndarray,
        mu_q: np.ndarray,
        var_q: np.ndarray,
    ) -> float:
        """
        KL(P ∥ Q) between two diagonal multivariate Gaussians.

        Formula:
            KL = 0.5 * mean_d [
                log(var_q / var_p)
                + (var_p + (mu_p - mu_q)^2) / var_q
                - 1
            ]

        Mean over dimensions (not sum) makes values comparable across
        channels with different vector sizes (e.g. 7-d emotion vs 2048-d IT).

        Units: nats per dimension.
        """
        var_p = np.maximum(var_p, 1e-9)
        var_q = np.maximum(var_q, 1e-9)

        kl = 0.5 * (
            np.log(var_q / var_p)
            + (var_p + (mu_p - mu_q) ** 2) / var_q
            - 1.0
        )
        return float(np.mean(kl))

    @staticmethod
    def _entropy(var_prior: np.ndarray) -> float:
        """
        Differential entropy of a diagonal Gaussian prior.

        H(prior) = 0.5 * mean_d [ log(2πe σ²_d) ]
                 ∝ mean_d [ log(σ²_d) ]

        High H → diffuse prior → uncertain (many things could happen next)
        Low H  → peaked prior  → confident (clear expectation)

        Units: nats per dimension (proportional form used for consistency
        with the KL units; the additive constant 0.5*log(2πe) is omitted
        because only relative changes across time matter).
        """
        var = np.maximum(var_prior, 1e-9)
        return float(np.mean(np.log(var)))
