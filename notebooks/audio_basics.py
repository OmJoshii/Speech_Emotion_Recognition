# phase1_audio_basics.py
# Phase 1: Understanding audio visually

# ── Imports ────────────────────────────────────────────
import librosa                  # audio processing library
import librosa.display          # for plotting audio
import numpy as np              # numerical operations
import matplotlib.pyplot as plt # plotting/visualization
import matplotlib.gridspec as gridspec  # for arranging plots

print("All libraries imported successfully!")

# ── Load Audio ─────────────────────────────────────────
# librosa has built-in sample audio files for testing
# We'll use this before we have our real dataset

audio_path = librosa.example('trumpet')  # built-in sample

# librosa.load() does two things:
# 1. Reads the audio file from disk
# 2. Converts it into a numpy array of numbers
# y  = the audio signal (array of numbers)
# sr = sample rate (how many numbers per second)

y, sr = librosa.load(audio_path, duration=3.0)
# duration=3.0 means we only load the first 3 seconds

print(f"\n── Audio Info ──────────────────")
print(f"Sample rate     : {sr} Hz")
print(f"Total samples   : {len(y)}")
print(f"Duration        : {len(y)/sr:.2f} seconds")
print(f"Min amplitude   : {y.min():.4f}")
print(f"Max amplitude   : {y.max():.4f}")
print(f"Signal is just numbers: {y[:5]}")

# ── Figure Setup ───────────────────────────────────────
# We'll plot 4 things in one figure so we can compare them
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(4, 1, hspace=0.5)
# This creates a canvas with 4 rows for 4 different plots

# ── Plot 1: Waveform ───────────────────────────────────
ax1 = fig.add_subplot(gs[0])

# waveshow() plots the audio signal
# x-axis = time, y-axis = amplitude
librosa.display.waveshow(y, sr=sr, ax=ax1, color='steelblue')

ax1.set_title('1. Waveform — Raw audio signal (amplitude over time)',
              fontsize=11, fontweight='bold')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Amplitude')

# Adding a horizontal line at 0 for reference
ax1.axhline(y=0, color='red', linewidth=0.5, linestyle='--',
            label='zero line')

print("\nWaveform plotted ✅")

# ── Plot 2: Regular Spectrogram ────────────────────────
ax2 = fig.add_subplot(gs[1])

# Short-Time Fourier Transform (STFT)
# This breaks the audio into small chunks and finds
# which frequencies are present in each chunk
# Don't worry about the math — just know it converts
# audio signal → frequency information over time

D = librosa.stft(y)             # complex numbers (raw STFT)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
# amplitude_to_db converts to decibels (dB) — a log scale
# that matches how humans perceive loudness

librosa.display.specshow(S_db,
                         sr=sr,
                         x_axis='time',   # time on x-axis
                         y_axis='hz',     # frequency in Hz on y-axis
                         ax=ax2,
                         cmap='magma')    # color scheme

ax2.set_title('2. Spectrogram — Frequencies over time (brighter = stronger)',
              fontsize=11, fontweight='bold')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Frequency (Hz)')
fig.colorbar(ax2.collections[0], ax=ax2, format='%+2.0f dB')

print("Spectrogram plotted ✅")

# ── Plot 3: Mel Spectrogram ────────────────────────────
ax3 = fig.add_subplot(gs[2])

# melspectrogram() converts audio to mel scale
# n_mels=128 means we use 128 frequency bands
# (adjusted to human hearing perception)

mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
# Convert to dB scale (log scale) for better visualization

librosa.display.specshow(mel_spec_db,
                         sr=sr,
                         x_axis='time',
                         y_axis='mel',    # mel scale on y-axis
                         ax=ax3,
                         cmap='magma')

ax3.set_title('3. Mel Spectrogram — Human-perception adjusted frequencies',
              fontsize=11, fontweight='bold')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Mel Frequency')
fig.colorbar(ax3.collections[0], ax=ax3, format='%+2.0f dB')

print("Mel Spectrogram plotted ✅")

# ── Plot 4: MFCC ───────────────────────────────────────
# MFCC = Mel Frequency Cepstral Coefficients
# This is the MOST important feature for our SER model
# We'll learn this deeply in Phase 3
# For now — just see what it looks like

ax4 = fig.add_subplot(gs[3])

# n_mfcc=13 means extract 13 coefficient features
# These 13 numbers summarize the "texture" of the voice
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

librosa.display.specshow(mfcc,
                         sr=sr,
                         x_axis='time',
                         ax=ax4,
                         cmap='coolwarm')

ax4.set_title('4. MFCC — Compact summary of voice texture (our main feature!)',
              fontsize=11, fontweight='bold')
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('MFCC Coefficients')
fig.colorbar(ax4.collections[0], ax=ax4)

print("MFCC plotted ✅")

# ── Save & Show ────────────────────────────────────────
plt.savefig('notebooks/phase1_audio_visualizations.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Phase 1 complete!")
print("Image saved to src/phase1_audio_visualizations.png")