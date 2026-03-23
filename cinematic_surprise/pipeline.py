"""
pipeline.py — CinematicSurprisePipeline

Main entry point. Orchestrates Phase 2 of the pipeline:
    - Read video frame by frame (1-second windows)
    - Extract features per modality
    - Run EMA + KL + uncertainty per channel per frame/second
    - Aggregate to per-second scalars
    - Run post-hoc aggregation (interactions, combined columns)
    - Return TWO outputs:
        1. pandas DataFrame with 48 columns (surprise/uncertainty/interaction)
        2. numpy ndarray (T, 4900) of raw per-second feature vectors

Phase 1 (Whisper transcription) is handled separately:
    from cinematic_surprise.io.transcript import transcribe
    transcribe('film.mp4', output='film_transcript.csv')

Usage
-----
    from cinematic_surprise import CinematicSurprisePipeline

    pipe = CinematicSurprisePipeline()
    df, features = pipe.run('film.mp4', transcript='film_transcript.csv')

    # Output 1: surprise / uncertainty / interaction table
    df.to_parquet('film_surprise.parquet', index=False)

    # Output 2: raw feature vectors for encoding models
    import numpy as np
    np.save('film_features.npy', features)  # (T, 4900)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from cinematic_surprise import config as cfg
from cinematic_surprise.io.audio import AudioExtractor
from cinematic_surprise.io.transcript import TranscriptReader
from cinematic_surprise.io.video import VideoReader
from cinematic_surprise.modalities.audio import extract_audio_features
from cinematic_surprise.modalities.face import FaceExtractor
from cinematic_surprise.modalities.motion import MotionExtractor
from cinematic_surprise.modalities.narrative import NarrativeExtractor
from cinematic_surprise.modalities.semantic import SemanticExtractor
from cinematic_surprise.modalities.visual import VisualExtractor
from cinematic_surprise.uncertainty_and_surprise.aggregator import run_all
from cinematic_surprise.uncertainty_and_surprise.estimator import OnlineGaussianEstimator

log = logging.getLogger(__name__)

# Channels that produce frame-level observations (need within-second mean)
_FRAME_LEVEL_CHANNELS = {"L1", "L2", "L3", "L4", "semantic", "motion", "emotion", "faces"}

# Channels that produce one observation per second (no within-second aggregation)
_SECOND_LEVEL_CHANNELS = {"audio_mel", "audio_spec", "audio_onset", "narrative"}


class CinematicSurprisePipeline:
    """
    Per-second Bayesian surprise and uncertainty estimator for film stimuli.

    Produces two outputs per film:
        1. A 48-column DataFrame of surprise, uncertainty, interaction scalars
           (the Bayesian-compressed representation for GLM regressors).
        2. A (T, 4900) ndarray of raw feature vectors from all 12 channels,
           concatenated in config.FEATURE_CHANNEL_ORDER
           (the uncompressed representation for encoding models).

    Both outputs share the same feature extraction — no redundant compute.

    All parameters default to config.py values and can be overridden.

    Args:
        alpha       : Dict of per-channel EMA learning rates.
                      Defaults to config.ALPHA.
        output_fmt  : 'parquet' or 'csv'. Used by save().
        max_seconds : Process only the first N seconds (None = whole film).
        batch_size  : CNN frames per forward pass.
        cut_threshold: Chi-squared scene-cut threshold.
    """

    def __init__(
        self,
        alpha:          Optional[Dict[str, float]] = None,
        output_fmt:     str = "parquet",
        max_seconds:    Optional[int] = None,
        batch_size:     int = cfg.BATCH_SIZE,
        cut_threshold:  float = 0.15,
    ):
        self.output_fmt    = output_fmt
        self.max_seconds   = max_seconds
        self.batch_size    = batch_size
        self.cut_threshold = cut_threshold

        # Shared estimator: one EMA+KL belief per named channel
        self.estimator = OnlineGaussianEstimator(alpha=alpha)

        # Modality extractors (loaded once, reused across films)
        log.info("Loading visual extractor (ResNet-50)...")
        self.visual    = VisualExtractor(device=cfg.RESNET_DEVICE)

        log.info(f"Loading semantic extractor (CLIP {cfg.CLIP_MODEL})...")
        self.semantic  = SemanticExtractor(
            model_name=cfg.CLIP_MODEL,
            device=cfg.CLIP_DEVICE,
        )

        self.motion    = MotionExtractor()
        self.face      = FaceExtractor()
        self.narrative = NarrativeExtractor()

        log.info("Pipeline ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        video_path:  str | Path,
        transcript:  Optional[str | Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process one film and return per-second surprise table + feature matrix.

        Args:
            video_path : Path to video file (MP4, MKV, AVI, ...)
            transcript : Path to standardised transcript CSV.
                         If None, narrative channel is skipped (zero-filled).

        Returns:
            (surprise_df, features_df)

            surprise_df  : pandas DataFrame with 48 columns
                           (9 metadata + 12 surprise + 12 uncertainty +
                            12 interaction + 3 aggregates).

            features_df  : pandas DataFrame, shape (T, 4900).
                           Raw per-second feature vectors from all 12 channels,
                           concatenated in config.FEATURE_CHANNEL_ORDER.
                           Column names follow '{channel}_{dim:04d}' convention
                           (e.g. L1_0000, audio_mel_0127, narrative_0383).
                           Frame-level channels are averaged across frames
                           within each second. Missing channels are zero-filled.
        """
        video_path = Path(video_path)

        # Reset all beliefs and stateful extractors for fresh film
        self._reset(video_path)

        # Load audio and transcript
        log.info("Loading audio...")
        audio_ext = AudioExtractor(str(video_path), sr=cfg.AUDIO_SAMPLE_RATE)

        transcript_reader: Optional[TranscriptReader] = None
        if transcript is not None:
            log.info(f"Loading transcript from '{transcript}'...")
            transcript_reader = TranscriptReader(transcript)
        else:
            log.info("No transcript provided. Narrative channel will be skipped.")

        # Main loop
        rows = []
        with VideoReader(
            video_path,
            max_seconds=self.max_seconds,
            cut_threshold=self.cut_threshold,
        ) as vr:
            total_s = min(vr.n_seconds, self.max_seconds or vr.n_seconds)

            for second_idx, frames, has_cut in tqdm(
                vr.iter_seconds(),
                total=total_s,
                desc=f"Processing '{video_path.name}'",
                unit="s",
            ):
                row = self._process_second(
                    second_idx=second_idx,
                    frames=frames,
                    has_cut=has_cut,
                    audio_ext=audio_ext,
                    transcript_reader=transcript_reader,
                )
                rows.append(row)

        # ── Build raw feature matrix (T, 4900) from _feat_* keys ──────────
        features = self._build_feature_matrix(rows)

        # ── Build surprise DataFrame ───────────────────────────────────────
        # Strip internal _feat_* keys before DataFrame construction
        clean_rows = [
            {k: v for k, v in row.items() if not k.startswith("_feat_")}
            for row in rows
        ]
        df = pd.DataFrame(clean_rows)

        # Post-hoc: compute interaction columns and aggregate columns
        log.info("Computing interactions and aggregates...")
        df = run_all(df)

        log.info(
            f"Done. {len(df)} seconds processed. "
            f"Columns: {len(df.columns)}. "
            f"Feature matrix: {features.shape}. "
            f"Peak surprise_combined at t={df['surprise_combined'].idxmax()}s."
        )
        return df, features

    def save(
        self,
        df:       pd.DataFrame,
        features: pd.DataFrame,
        path:     str | Path,
    ) -> Tuple[Path, Path]:
        """
        Save both outputs to disk.

        Args:
            df       : Surprise DataFrame from run()
            features : Feature DataFrame from run()  (T, 4900)
            path     : Base output path (extensions added automatically)

        Returns:
            (surprise_path, features_path) — actual paths written.
        """
        path = Path(path)

        # Save surprise table
        if self.output_fmt == "parquet":
            out_df = path.with_suffix(".parquet")
            df.to_parquet(out_df, index=False)
        else:
            out_df = path.with_suffix(".csv")
            df.to_csv(out_df, index=False)

        # Save feature matrix (same format as surprise table)
        if self.output_fmt == "parquet":
            out_feat = path.with_name(path.stem + "_features").with_suffix(".parquet")
            features.to_parquet(out_feat, index=False)
        else:
            out_feat = path.with_name(path.stem + "_features").with_suffix(".csv")
            features.to_csv(out_feat, index=False)

        log.info(f"Saved surprise → '{out_df}', features → '{out_feat}'")
        return out_df, out_feat

    # ── Internal ───────────────────────────────────────────────────────────────

    def _reset(self, video_path: Path) -> None:
        """Reset all stateful components for a new film."""
        self.estimator.reset()
        self.motion.reset()
        self.narrative.reset()
        log.info(f"State reset for '{video_path.name}'")

    def _build_feature_matrix(self, rows: List[dict]) -> pd.DataFrame:
        """
        Assemble (T, 4900) feature DataFrame from the _feat_* keys in each row.

        Each row dict contains _feat_{channel} → numpy array for every channel
        that was successfully extracted. Missing channels are zero-filled using
        the expected dimension from config.FEATURE_DIMS.

        The concatenation order is config.FEATURE_CHANNEL_ORDER — this is the
        single source of truth and must not be changed without versioning.

        Column names follow '{channel}_{dim:04d}' convention, generated by
        config.feature_column_names().

        Returns:
            pandas DataFrame, shape (T, FEATURE_TOTAL_DIM), dtype float32.
        """
        feature_rows = []
        for row in rows:
            parts = []
            for ch in cfg.FEATURE_CHANNEL_ORDER:
                key = f"_feat_{ch}"
                if key in row and row[key] is not None:
                    vec = np.asarray(row[key], dtype=np.float32).ravel()
                    # Validate dimension matches expectation
                    expected_d = cfg.FEATURE_DIMS[ch]
                    if vec.shape[0] != expected_d:
                        log.warning(
                            f"Feature dim mismatch for '{ch}': "
                            f"got {vec.shape[0]}, expected {expected_d}. "
                            f"Zero-filling."
                        )
                        vec = np.zeros(expected_d, dtype=np.float32)
                    parts.append(vec)
                else:
                    # Zero-fill missing channels (e.g. narrative with no transcript,
                    # semantic if CLIP unavailable, audio if ffmpeg missing)
                    parts.append(np.zeros(cfg.FEATURE_DIMS[ch], dtype=np.float32))
            feature_rows.append(np.concatenate(parts))

        data = np.stack(feature_rows, axis=0)  # (T, 4900)
        assert data.shape[1] == cfg.FEATURE_TOTAL_DIM, (
            f"Feature matrix width {data.shape[1]} != "
            f"FEATURE_TOTAL_DIM {cfg.FEATURE_TOTAL_DIM}"
        )
        return pd.DataFrame(data, columns=cfg.feature_column_names())

    def _process_second(
        self,
        second_idx:        int,
        frames:            List[np.ndarray],
        has_cut:           bool,
        audio_ext:         AudioExtractor,
        transcript_reader: Optional[TranscriptReader],
    ) -> dict:
        """
        Process one 1-second window → one row dict.

        The row dict contains:
            - Metadata keys (time_s, n_frames, scene_cut, face metadata)
            - surprise_* and uncertainty_* scalars for each channel
            - _feat_* numpy arrays for the raw feature output (stripped later)

        Frame-level channels (L1-L4, semantic, motion, emotion, faces):
            EMA+KL per frame → collect frame-level surprise and uncertainty
            Per-second surprise/uncertainty = mean of frame-level values
            Per-second feature = mean of frame-level feature vectors

        Second-level channels (audio_mel, audio_spec, audio_onset, narrative):
            One feature vector per second → one EMA+KL update per second
            Per-second feature = the raw vector itself (no averaging needed)
        """
        frames_arr = np.stack(frames, axis=0)   # (B, H, W, 3)
        B = len(frames_arr)

        row: dict = {
            "time_s":    second_idx,
            "n_frames":  B,
            "scene_cut": has_cut,
        }

        # ── Frame-level: visual (L1–L4) ────────────────────────────────────
        # Process in sub-batches for GPU memory safety
        visual_feats:   Dict[str, List[np.ndarray]] = {ch: [] for ch in ["L1", "L2", "L3", "L4"]}
        semantic_feats: List[np.ndarray] = []

        for start in range(0, B, self.batch_size):
            batch = frames_arr[start: start + self.batch_size]

            # ResNet L1-L4
            vis = self.visual.extract(batch)
            for ch in ["L1", "L2", "L3", "L4"]:
                visual_feats[ch].append(vis[ch])

            # CLIP semantic
            if self.semantic.available:
                sem = self.semantic.extract(batch)
                if "semantic" in sem:
                    semantic_feats.append(sem["semantic"])

        # Concatenate sub-batches → (B, d) per channel, run EMA+KL, store features
        for ch in ["L1", "L2", "L3", "L4"]:
            feat_seq = np.concatenate(visual_feats[ch], axis=0)   # (B, d)
            s_vals, u_vals = zip(*[
                self.estimator.update(feat_seq[i], ch)
                for i in range(B)
            ])
            row[f"surprise_{ch}"]    = float(np.mean(s_vals))
            row[f"uncertainty_{ch}"] = float(np.mean(u_vals))
            # Raw feature: mean across frames within this second
            row[f"_feat_{ch}"]       = feat_seq.mean(axis=0)      # (d,)

        # Semantic
        if semantic_feats:
            sem_seq = np.concatenate(semantic_feats, axis=0)   # (B, d)
            s_vals, u_vals = zip(*[
                self.estimator.update(sem_seq[i], "semantic")
                for i in range(len(sem_seq))
            ])
            row["surprise_semantic"]    = float(np.mean(s_vals))
            row["uncertainty_semantic"] = float(np.mean(u_vals))
            row["_feat_semantic"]       = sem_seq.mean(axis=0)    # (d,)
        else:
            row["surprise_semantic"]    = np.nan
            row["uncertainty_semantic"] = np.nan
            row["_feat_semantic"]       = None  # will be zero-filled in _build_feature_matrix

        # ── Frame-level: motion ────────────────────────────────────────────
        motion_s, motion_u = [], []
        motion_vecs = []                          # collect frame-level vectors
        for frame in frames_arr:
            feat = self.motion.extract(frame)
            motion_vecs.append(feat)
            s, u = self.estimator.update(feat, "motion")
            motion_s.append(s)
            motion_u.append(u)
        row["surprise_motion"]    = float(np.mean(motion_s))
        row["uncertainty_motion"] = float(np.mean(motion_u))
        row["_feat_motion"]       = np.mean(motion_vecs, axis=0)  # (19,)

        # ── Frame-level: face / emotion ────────────────────────────────────
        face_result = self.face.extract(list(frames_arr))

        for face_ch in ["emotion", "faces"]:
            feat = face_result[face_ch]
            # Face features are already aggregated to per-second vectors
            # by FaceExtractor, so we do one EMA+KL update per second
            s, u = self.estimator.update(feat, face_ch)
            row[f"surprise_{face_ch}"]    = s
            row[f"uncertainty_{face_ch}"] = u
            row[f"_feat_{face_ch}"]       = np.asarray(feat, dtype=np.float32)

        # Face metadata columns (not fed into EMA+KL, not in feature vector)
        row["n_faces_mean"]          = face_result["n_faces_mean"]
        row["n_faces_std"]           = face_result["n_faces_std"]
        row["face_coverage_mean"]    = face_result["face_coverage_mean"]
        row["face_coverage_std"]     = face_result["face_coverage_std"]
        row["dominant_emotion_idx"]  = face_result["dominant_emotion_idx"]
        row["dominant_emotion"]      = face_result["dominant_emotion"]

        # ── Second-level: audio ────────────────────────────────────────────
        if audio_ext.available:
            segment = audio_ext.get_segment(second_idx)
            if segment is not None:
                audio_feats = extract_audio_features(segment, sr=audio_ext.sr)
                if audio_feats:
                    for audio_ch in ["audio_mel", "audio_spec", "audio_onset"]:
                        s, u = self.estimator.update(audio_feats[audio_ch], audio_ch)
                        row[f"surprise_{audio_ch}"]    = s
                        row[f"uncertainty_{audio_ch}"] = u
                        # Raw feature: already one vector per second
                        row[f"_feat_{audio_ch}"]       = np.asarray(
                            audio_feats[audio_ch], dtype=np.float32
                        )
                else:
                    for audio_ch in ["audio_mel", "audio_spec", "audio_onset"]:
                        row[f"surprise_{audio_ch}"]    = np.nan
                        row[f"uncertainty_{audio_ch}"] = np.nan
                        row[f"_feat_{audio_ch}"]       = None
            else:
                for audio_ch in ["audio_mel", "audio_spec", "audio_onset"]:
                    row[f"surprise_{audio_ch}"]    = np.nan
                    row[f"uncertainty_{audio_ch}"] = np.nan
                    row[f"_feat_{audio_ch}"]       = None
        else:
            for audio_ch in ["audio_mel", "audio_spec", "audio_onset"]:
                row[f"surprise_{audio_ch}"]    = np.nan
                row[f"uncertainty_{audio_ch}"] = np.nan
                row[f"_feat_{audio_ch}"]       = None

        # ── Second-level: narrative ────────────────────────────────────────
        if transcript_reader is not None:
            text = transcript_reader.get_words(second_idx)
            nar_feat = self.narrative.extract(text)
            if "narrative" in nar_feat:
                s, u = self.estimator.update(nar_feat["narrative"], "narrative")
                row["surprise_narrative"]    = s
                row["uncertainty_narrative"] = u
                row["_feat_narrative"]       = np.asarray(
                    nar_feat["narrative"], dtype=np.float32
                )
            else:
                row["surprise_narrative"]    = np.nan
                row["uncertainty_narrative"] = np.nan
                row["_feat_narrative"]       = None
        else:
            row["surprise_narrative"]    = np.nan
            row["uncertainty_narrative"] = np.nan
            row["_feat_narrative"]       = None

        return row
