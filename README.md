# cinematic_surprise

Per-second hierarchical Bayesian surprise and uncertainty measurement in movies.

## Quick Start

```bash
pip install -e .
# CLIP requires separate install:
pip install git+https://github.com/openai/CLIP.git
```

**With Whisper transcription (Phase 1 + Phase 2):**
```python
from cinematic_surprise import CinematicSurprisePipeline
from cinematic_surprise.io.transcript import transcribe

# Phase 1: run once per film
transcribe('my_film.mp4', output='my_film_transcript.csv')

# Phase 2: main pipeline
pipe = CinematicSurprisePipeline()
df   = pipe.run('my_film.mp4', transcript='my_film_transcript.csv')
df.to_parquet('my_film_surprise.parquet', index=False)
```

**With pre-existing transcript:**
```python
pipe = CinematicSurprisePipeline()
df   = pipe.run('my_film.mp4', transcript='my_film_transcript.csv')
```

**Jupyter notebook / first test (CPU-friendly, 60 seconds):**
```python
import cinematic_surprise.config as cfg
cfg.CLIP_DEVICE      = "cpu"
cfg.RESNET_DEVICE    = "cpu"
cfg.DEEPFACE_BACKEND = "mtcnn"
cfg.WHISPER_MODEL    = "base"
cfg.BATCH_SIZE       = 4

pipe = CinematicSurprisePipeline(max_seconds=60)
df   = pipe.run("my_film.mp4", transcript="transcript.csv")
df.head()
```

## Output

48 columns per second:
- 9 metadata columns (time_s, scene_cut, n_faces_mean, dominant_emotion, ...)
- 12 surprise columns (surprise_L1 ... surprise_narrative)
- 12 uncertainty columns (uncertainty_L1 ... uncertainty_narrative)
- 12 interaction columns (interaction_L1 ... interaction_narrative)
- 3 aggregate columns (surprise_combined, uncertainty_combined, interaction_combined)

## Transcript Format

Pre-existing transcripts must follow this CSV format (confirmed from sample):

| Column | Type | Description |
|--------|------|-------------|
| subtitle | str | one word per row |
| start_time_new | float | word start in seconds |
| end_time_new | float | word end in seconds |
| interval_new | float | word duration |
| type | str | confidence: matched, full, similar, partial, ... |

## References

- Itti & Baldi (2009). Bayesian surprise attracts human attention. *Vision Research*.
- Cheung et al. (2019). Uncertainty and Surprise Jointly Predict Musical Pleasure. *Current Biology*.
- Yamins & DiCarlo (2016). Using goal-driven deep learning models to understand sensory cortex. *Nature Neuroscience*.
- Schrimpf et al. (2020). Brain-Score. *PNAS*.
- Radford et al. (2021). CLIP. *ICML*.
