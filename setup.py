from setuptools import setup, find_packages

setup(
    name="cinematic_surprise",
    version="0.2.0",
    description=(
        "Per-second hierarchical Bayesian surprise and uncertainty "
        "measurement in movies. Implements Itti & Baldi (2009) KL-divergence "
        "surprise and Cheung et al. (2019) uncertainty across 12 channels "
        "spanning visual, audio, motion, face/emotion, and narrative modalities."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "torch>=1.12",
        "torchvision>=0.13",
        "opencv-python>=4.5",
        "tqdm>=4.60",
    ],
    extras_require={
        "full": [
            "librosa>=0.9",
            "deepface>=0.0.75",
            "sentence-transformers>=2.2",
            "openai-whisper",
            # CLIP: pip install git+https://github.com/openai/CLIP.git
        ],
        "dev": [
            "jupyter",
            "matplotlib>=3.4",
            "seaborn>=0.11",
            "pytest",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Intended Audience :: Science/Research",
    ],
)
