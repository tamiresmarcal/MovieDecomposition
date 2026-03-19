"""
config.py — Single source of truth for all pipeline parameters.

Every value in this file corresponds to a reportable decision in the
methods section of the paper. Do not hardcode any of these values
elsewhere in the package.
"""

# ── Model selection ────────────────────────────────────────────────────────────

CLIP_MODEL        = "ViT-B/32"      # options: "ViT-B/32", "ViT-L/14"
                                    # ViT-B/32: 512-d, fast, good for scene-level shifts
                                    # ViT-L/14: 768-d, ~2x slower, better fine-grained semantics

CLIP_DEVICE       = "cuda"          # options: "cuda", "cpu"
                                    # falls back to cpu automatically if cuda unavailable

RESNET_DEVICE     = "cuda"          # options: "cuda", "cpu"

DEEPFACE_BACKEND  = "retinaface"    # options: "retinaface", "mtcnn", "opencv", "ssd"
                                    # retinaface: most accurate, slowest
                                    # mtcnn:      good balance (recommended for testing)
                                    # opencv:     fastest, least accurate

WHISPER_MODEL     = "base"          # options: "tiny", "base", "small", "medium", "large"
                                    # base:  fast, good for clean audio
                                    # small: better with accents/noise
                                    # large: best accuracy, slow

# ── CNN inference ──────────────────────────────────────────────────────────────

BATCH_SIZE        = 8               # frames per CNN forward pass
                                    # reduce to 4 if GPU OOM
                                    # increase to 16-32 on large GPUs

# ── Transcript ────────────────────────────────────────────────────────────────

# Word confidence types to include when building per-second text.
# From the confirmed transcript format (Pulp Fiction sample):
#   matched, partial, full, similar, continuous, partial_adjusted, continuous_adjusted
TRANSCRIPT_TYPES  = ["matched", "full", "similar"]

# Confirmed column names from the standardised transcript format
TRANSCRIPT_COL_WORD  = "subtitle"
TRANSCRIPT_COL_START = "start_time_new"
TRANSCRIPT_COL_END   = "end_time_new"
TRANSCRIPT_COL_TYPE  = "type"

# ── Audio ─────────────────────────────────────────────────────────────────────

AUDIO_SAMPLE_RATE = 22050           # Hz, librosa standard
AUDIO_N_MELS      = 128             # mel frequency bands (A1 tonotopic resolution)
AUDIO_N_FFT       = 2048            # FFT window in samples
AUDIO_HOP_LENGTH  = 512             # STFT hop in samples (~43 frames/second)

# ── Optical flow ──────────────────────────────────────────────────────────────

FLOW_RESIZE       = (320, 180)      # resolution for flow computation (speed/accuracy trade-off)
FLOW_N_BINS       = 16              # magnitude histogram bins
FLOW_MAX_MAG      = 20.0            # pixels/frame, clip above this

# ── Face detection ────────────────────────────────────────────────────────────

FACE_MIN_AREA_PCT = 0.0001          # ignore faces smaller than 0.01% of frame area
                                    # filters false positives in background noise

# ── Emotion labels ────────────────────────────────────────────────────────────

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_IDX    = {label: i for i, label in enumerate(EMOTION_LABELS)}
IDX_EMOTION    = {i: label for i, label in enumerate(EMOTION_LABELS)}

# ── EMA alpha per channel ─────────────────────────────────────────────────────
# Half-life = log(0.5) / log(1 - alpha) frames
# At 25fps:
#   alpha=0.10 → T_half ~6.6 frames / 0.26s   (fast: V1, motion, audio onset)
#   alpha=0.05 → T_half ~13.5 frames / 0.54s  (medium: LOC, IT, faces, audio)
#   alpha=0.03 → T_half ~22.8 frames / 0.91s  (slow: semantic, emotion)
#   alpha=0.01 → T_half ~69 frames / ~7s      (very slow: narrative)

ALPHA = {
    "L1":           0.10,   # V1: adapts rapidly to edge/orientation changes
    "L2":           0.10,   # V2/V4: texture responses similarly fast
    "L3":           0.05,   # LOC: object-part responses adapt over ~1 shot
    "L4":           0.05,   # IT: object identity adapts over ~1 shot
    "semantic":     0.03,   # PFC: semantic content changes over multiple shots
    "emotion":      0.03,   # FFA/amygdala: emotional arcs span multiple shots
    "faces":        0.05,   # OFA/STS: face count changes at shot boundaries
    "motion":       0.10,   # MT/V5: motion changes frame to frame
    "audio_mel":    0.05,   # A1: spectral texture adapts over seconds
    "audio_spec":   0.05,   # Belt: timbral character adapts over seconds
    "audio_onset":  0.10,   # Parabelt: onset events are inherently transient
    "narrative":    0.01,   # Language areas: context spans paragraphs/scenes
}

# ── Channel registry ──────────────────────────────────────────────────────────
# Ordered list of all 12 channels. Order determines column order in output.

CHANNELS = [
    "L1", "L2", "L3", "L4",
    "semantic",
    "emotion", "faces",
    "motion",
    "audio_mel", "audio_spec", "audio_onset",
    "narrative",
]

# ── EMA numerical stability ───────────────────────────────────────────────────

EMA_EPSILON = 1e-7              # variance floor to prevent zero-variance collapse

# ── ResNet layer map ──────────────────────────────────────────────────────────

RESNET_LAYER_MAP = {
    "L1": "layer1",   # (B, 256, 56, 56) → GAP → (256,)   V1 analog
    "L2": "layer2",   # (B, 512, 28, 28) → GAP → (512,)   V2/V4 analog
    "L3": "layer3",   # (B,1024, 14, 14) → GAP → (1024,)  LOC analog
    "L4": "layer4",   # (B,2048,  7,  7) → GAP → (2048,)  IT analog
}

CNN_INPUT_SIZE = 224            # ResNet/CLIP standard input resolution

# ImageNet normalisation constants (mean/std per RGB channel)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
