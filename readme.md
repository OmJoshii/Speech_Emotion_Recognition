# Speech Emotion Recognition System

A deep learning system that detects human emotions from speech audio using
CNN and LSTM models. Built as a final year major project for Computer Engineering.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

Takes a speech audio file (or live microphone input) and predicts the
emotion of the speaker — Happy, Sad, Angry, Neutral, Calm, Fearful,
Disgust, or Surprised.

---

## Emotions Detected

| Code | Emotion   |
|------|-----------|
| 0    | Neutral   |
| 1    | Calm      |
| 2    | Happy     |
| 3    | Sad       |
| 4    | Angry     |
| 5    | Fearful   |
| 6    | Disgust   |
| 7    | Surprised |

---

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

- 2880 speech audio files
- 24 actors (12 male, 12 female)
- 8 emotions × 2 intensities × 2 statements × 2 repetitions
- Neutral has 192 files, all others have 384 files

Download from:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Extract and place inside `data/ravdess/` folder.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Librosa | Audio processing |
| NumPy & Pandas | Data handling |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | Baseline ML models |
| PyTorch | Deep learning |
| Streamlit | Live demo app |

---

## Features Extracted

| Feature | Shape | Used By |
|---------|-------|---------|
| MFCC (mean + std) | (94,) | Baseline MLP model |
| Mel Spectrogram | (128, 130) | CNN model |
| MFCC Sequence | (130, 40) | LSTM model |

---
