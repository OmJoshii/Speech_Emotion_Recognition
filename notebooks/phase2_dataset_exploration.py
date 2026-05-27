# phase2_dataset_exploration.py
# Phase 2: Loading and exploring the RAVDESS dataset

import os               # for navigating folders and files
import librosa          # for loading audio
import numpy as np      # numerical operations
import pandas as pd     # for organizing data in tables
import matplotlib.pyplot as plt
import seaborn as sns   # for beautiful plots

print("=" * 50)
print("   Phase 2: Dataset Exploration")
print("=" * 50)

# ── Emotion Mapping ────────────────────────────────────
# RAVDESS encodes emotions as numbers in the filename
# We create a dictionary to convert numbers to names
# A dictionary is like a lookup table: key → value

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

print("\nEmotion mapping loaded:")
for code, name in emotion_map.items():
    print(f"  {code} → {name}")

# ── Dataset Path ───────────────────────────────────────
# os.path.join builds a file path correctly on any OS
# '..' means go one folder up from current location
dataset_path = os.path.join('data', 'ravdess')

# Check if the dataset folder exists
if not os.path.exists(dataset_path):
    print(f"\n❌ Dataset not found at '{dataset_path}'")
    print("Please download RAVDESS and place it in data/ravdess/")
    exit()

print(f"\n Dataset found at: {dataset_path}")

# ── Collect File Information ───────────────────────────
# We'll build a list of dictionaries
# Each dictionary = info about one audio file

file_data = []  # empty list to store all file info

# os.walk() goes through every folder and subfolder
# root   = current folder path
# dirs   = list of subfolders inside root
# files  = list of files inside root

for root, dirs, files in os.walk(dataset_path):
    for filename in files:

        # We only want .wav files, skip anything else
        if not filename.endswith('.wav'):
            continue

        # Build the full path to the file
        # e.g. data/ravdess/Actor_01/03-01-06-01-02-01-12.wav
        file_path = os.path.join(root, filename)

        # Split filename by '-' to get each part
        # '03-01-06-01-02-01-12.wav' → ['03','01','06','01','02','01','12.wav']
        parts = filename.split('-')

        # Get emotion code — it's the 3rd part (index 2)
        # Remove .wav extension from last part just in case
        emotion_code = parts[2]

        # Look up emotion name from our dictionary
        emotion_name = emotion_map.get(emotion_code, 'unknown')

        # Get actor number — it's the last part
        # parts[-1] means last element, strip('.wav') removes extension
        actor = parts[6].replace('.wav', '')

        # Get gender — actors 01-12 are male, 13-24 are female
        # int() converts string '12' to number 12
        actor_num = int(actor)
        gender = 'male' if actor_num <= 12 else 'female'

        # Get intensity — 4th part: 01=normal, 02=strong
        intensity_code = parts[3]
        intensity = 'normal' if intensity_code == '01' else 'strong'

        # Store all info about this file as a dictionary
        file_data.append({
            'file_path'   : file_path,
            'filename'    : filename,
            'emotion_code': emotion_code,
            'emotion'     : emotion_name,
            'actor'       : actor_num,
            'gender'      : gender,
            'intensity'   : intensity
        })

print(f"\nTotal audio files found: {len(file_data)}")


# ── Create DataFrame ───────────────────────────────────
# A DataFrame is like an Excel spreadsheet in Python
# Each row = one audio file
# Each column = one property of that file

df = pd.DataFrame(file_data)

# Sort by actor then emotion for cleanliness
df = df.sort_values(['actor', 'emotion']).reset_index(drop=True)

print("\n── First 5 rows of our dataset ──")
print(df.head())
# .head() shows the first 5 rows — good for quick inspection

print("\n── Dataset shape ──")
print(f"Rows (files)   : {df.shape[0]}")
print(f"Columns (info) : {df.shape[1]}")

print("\n── Column names ──")
print(df.columns.tolist())

print("\n── Data types ──")
print(df.dtypes)


# ── Emotion Distribution ───────────────────────────────
# How many files does each emotion have?
# value_counts() counts how many times each value appears

print("\n── Emotion Distribution ──")
emotion_counts = df['emotion'].value_counts().sort_index()
print(emotion_counts)

# ── Gender Distribution ────────────────────────────────
print("\n── Gender Distribution ──")
print(df['gender'].value_counts())

# ── Emotion by Gender ──────────────────────────────────
print("\n── Emotion by Gender ──")
print(pd.crosstab(df['emotion'], df['gender']))
# crosstab creates a table showing counts for
# every combination of emotion and gender


# ── Visualizations ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RAVDESS Dataset Exploration', fontsize=14,
             fontweight='bold')

# ── Plot 1: Emotion distribution bar chart ─────────────
emotion_counts = df['emotion'].value_counts()
axes[0, 0].bar(emotion_counts.index,
               emotion_counts.values,
               color='steelblue', edgecolor='white')
axes[0, 0].set_title('Files per Emotion')
axes[0, 0].set_xlabel('Emotion')
axes[0, 0].set_ylabel('Number of Files')
axes[0, 0].tick_params(axis='x', rotation=45)
# Add count on top of each bar
for i, (emotion, count) in enumerate(emotion_counts.items()):
    axes[0, 0].text(i, count + 1, str(count),
                    ha='center', fontsize=9)

# ── Plot 2: Gender distribution pie chart ──────────────
gender_counts = df['gender'].value_counts()
axes[0, 1].pie(gender_counts.values,
               labels=gender_counts.index,
               autopct='%1.1f%%',      # show percentage
               colors=['steelblue', 'salmon'],
               startangle=90)
axes[0, 1].set_title('Gender Distribution')

# ── Plot 3: Emotion by gender grouped bar ──────────────
emotion_gender = pd.crosstab(df['emotion'], df['gender'])
emotion_gender.plot(kind='bar',
                    ax=axes[1, 0],
                    color=['salmon', 'steelblue'],
                    edgecolor='white')
axes[1, 0].set_title('Emotion Distribution by Gender')
axes[1, 0].set_xlabel('Emotion')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend(title='Gender')

# ── Plot 4: Intensity distribution ─────────────────────
intensity_counts = df['intensity'].value_counts()
axes[1, 1].bar(intensity_counts.index,
               intensity_counts.values,
               color=['steelblue', 'coral'],
               edgecolor='white')
axes[1, 1].set_title('Intensity Distribution')
axes[1, 1].set_xlabel('Intensity')
axes[1, 1].set_ylabel('Count')
for i, (intens, count) in enumerate(intensity_counts.items()):
    axes[1, 1].text(i, count + 1, str(count),
                    ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('notebooks/phase2_dataset_exploration.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\n Visualization saved!")



# ── Sample one file per emotion ────────────────────────
# Let's load one audio file from each emotion
# and check its basic properties

print("\n── Sample Audio Info per Emotion ──")
print(f"{'Emotion':<12} {'Duration':>10} {'Sample Rate':>12} {'Filename'}")
print("-" * 65)

for emotion in sorted(df['emotion'].unique()):
    # Get first file for this emotion
    # .iloc[0] means get the first row
    sample_row = df[df['emotion'] == emotion].iloc[0]

    # Load just enough to get duration — no need to load full file
    y, sr = librosa.load(sample_row['file_path'], sr=None)
    # sr=None means use the original sample rate of the file

    duration = len(y) / sr  # duration in seconds

    print(f"{emotion:<12} {duration:>8.2f}s  {sr:>10} Hz  "
          f"{sample_row['filename']}")

print("\n Dataset exploration complete!")


# ── Save DataFrame ─────────────────────────────────────
# Save our organized file list to a CSV
# So we don't have to scan folders every time

csv_path = os.path.join('data', 'ravdess_files.csv')
df.to_csv(csv_path, index=False)
# index=False means don't save row numbers

print(f"\n File list saved to: {csv_path}")
print(f"   This CSV will be used in all future phases!")
print("\n" + "=" * 50)
print("   Phase 2 Complete!")
print("=" * 50)