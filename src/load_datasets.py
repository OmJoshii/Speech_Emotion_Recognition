# Loads and combines RAVDESS + CREMA-D + TESS + SAVEE
# into one unified dataframe with 6 common emotions


import os
import pandas as pd
from collections import Counter

# ── Unified emotion mapping ────────────────────────────
# 6 emotions that exist across ALL 4 datasets
# We skip 'calm', 'surprised' from RAVDESS
# We skip 'pleasant_surprise' from TESS

UNIFIED_EMOTIONS = {
    'neutral' : 0,
    'happy'   : 1,
    'sad'     : 2,
    'angry'   : 3,
    'fearful' : 4,
    'disgust' : 5,
}


# ══════════════════════════════════════════════════════
# RAVDESS LOADER
# ══════════════════════════════════════════════════════

def load_ravdess(data_path='data/ravdess'):
    """
    Loads RAVDESS dataset.

    Filename format: 03-01-05-01-01-01-01.wav
    Position 3 (index 2) = emotion code
    Position 1 (index 0) = modality (03 = speech)

    Emotion codes used:
      01=neutral, 03=happy, 04=sad,
      05=angry, 06=fearful, 07=disgust
    Skipped:
      02=calm, 08=surprised (not in other datasets)
    """
    print("Loading RAVDESS...")

    ravdess_map = {
        '01': 'neutral',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
    }

    files = []

    if not os.path.exists(data_path):
        print(f"  ⚠️  Not found at {data_path}")
        return []

    for root, dirs, filenames in os.walk(data_path):
        for fname in filenames:
            if not fname.endswith('.wav'):
                continue

            parts = fname.split('-')
            if len(parts) < 7:
                continue

            # Only speech modality
            if parts[0] != '03':
                continue

            emotion_code = parts[2]
            if emotion_code not in ravdess_map:
                continue  # skip calm and surprised

            emotion = ravdess_map[emotion_code]
            files.append({
                'file_path': os.path.join(root, fname),
                'emotion'  : emotion,
                'label'    : UNIFIED_EMOTIONS[emotion],
                'source'   : 'ravdess'
            })

    print(f"  ✅ RAVDESS  : {len(files)} files")
    return files


# ══════════════════════════════════════════════════════
# CREMA-D LOADER
# ══════════════════════════════════════════════════════

def load_crema_d(data_path='data/crema_d'):
    """
    Loads CREMA-D dataset.

    Filename format: 1001_DFA_ANG_XX.wav
    Position 2 (index 2) after split by '_' = emotion

    Emotion codes:
      ANG=angry, DIS=disgust, FEA=fearful,
      HAP=happy, NEU=neutral, SAD=sad
    """
    print("Loading CREMA-D...")

    crema_map = {
        'NEU': 'neutral',
        'HAP': 'happy',
        'SAD': 'sad',
        'ANG': 'angry',
        'FEA': 'fearful',
        'DIS': 'disgust',
    }

    files = []

    if not os.path.exists(data_path):
        print(f"  ⚠️  Not found at {data_path}")
        return []

    for fname in os.listdir(data_path):
        if not fname.endswith('.wav'):
            continue

        # 1001_DFA_ANG_XX.wav → ['1001','DFA','ANG','XX']
        parts = fname.replace('.wav', '').split('_')

        if len(parts) < 3:
            continue

        emotion_code = parts[2]  # 'ANG', 'HAP', etc.

        if emotion_code not in crema_map:
            continue

        emotion = crema_map[emotion_code]
        files.append({
            'file_path': os.path.join(data_path, fname),
            'emotion'  : emotion,
            'label'    : UNIFIED_EMOTIONS[emotion],
            'source'   : 'crema_d'
        })

    print(f"  ✅ CREMA-D  : {len(files)} files")
    return files


# ══════════════════════════════════════════════════════
# TESS LOADER
# ══════════════════════════════════════════════════════

def load_tess(data_path='data/tess'):
    """
    Loads TESS dataset.

    Folder structure:
      OAF_angry/     → angry
      OAF_Fear/      → fearful (capital F!)
      OAF_Sad/       → sad (capital S!)
      OAF_disgust/   → disgust
      OAF_happy/     → happy
      OAF_neutral/   → neutral
      OAF_Pleasant_surprise/ → SKIP
      YAF_angry/     → angry
      YAF_fear/      → fearful (lowercase f)
      YAF_sad/       → sad (lowercase s)
      ... etc

    We extract emotion from folder name (after _ )
    and convert to lowercase for consistency.
    """
    print("Loading TESS...")

    # Map folder emotion words to unified names
    # Using lowercase comparison to handle
    # inconsistent capitalization (Fear vs fear)
    tess_map = {
        'neutral'          : 'neutral',
        'happy'            : 'happy',
        'sad'              : 'sad',
        'angry'            : 'angry',
        'fear'             : 'fearful',
        'disgust'          : 'disgust',
        'pleasant_surprise': None,  # skip this
    }

    files = []

    if not os.path.exists(data_path):
        print(f"  ⚠️  Not found at {data_path}")
        return []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)

        if not os.path.isdir(folder_path):
            continue

        # Extract emotion from folder name
        # 'OAF_angry'           → emotion part = 'angry'
        # 'OAF_Fear'            → emotion part = 'Fear'
        # 'OAF_Pleasant_surprise' → emotion part = 'Pleasant_surprise'
        # 'YAF_fear'            → emotion part = 'fear'

        # Split by first underscore only
        parts = folder.split('_', 1)
        # 'OAF_angry' → ['OAF', 'angry']
        # 'OAF_Pleasant_surprise' → ['OAF', 'Pleasant_surprise']

        if len(parts) < 2:
            continue

        # Get emotion part and lowercase it
        emotion_key = parts[1].lower()
        # 'angry', 'fear', 'sad', 'pleasant_surprise' etc.

        if emotion_key not in tess_map:
            continue

        emotion = tess_map[emotion_key]

        if emotion is None:
            # Skip pleasant_surprise
            continue

        for fname in os.listdir(folder_path):
            if not fname.endswith('.wav'):
                continue

            files.append({
                'file_path': os.path.join(
                    folder_path, fname
                ),
                'emotion'  : emotion,
                'label'    : UNIFIED_EMOTIONS[emotion],
                'source'   : 'tess'
            })

    print(f"  ✅ TESS     : {len(files)} files")
    return files


# ══════════════════════════════════════════════════════
# SAVEE LOADER
# ══════════════════════════════════════════════════════

def load_savee(data_path='data/savee'):
    """
    Loads SAVEE dataset.

    Filename format: DC_a01.wav
      DC  = actor ID
      a   = emotion code
      01  = utterance number

    Emotion codes:
      a  = angry
      d  = disgust
      f  = fear (fearful)
      h  = happy
      n  = neutral
      sa = sad
      su = surprised → SKIP

    Important: 'sa' and 'su' are TWO character codes
    so we must check carefully — 'sa' starts with 's'
    same as 'su'. We check full prefix not just first char.
    """
    print("Loading SAVEE...")

    # Map emotion prefix to unified name
    # Order matters — check 'sa' and 'su' before 's'
    savee_map = {
        'sa': 'sad',       # must check before 'a' etc.
        'su': None,        # surprised → skip
        'a' : 'angry',
        'd' : 'disgust',
        'f' : 'fearful',
        'h' : 'happy',
        'n' : 'neutral',
    }

    files = []

    if not os.path.exists(data_path):
        print(f"  ⚠️  Not found at {data_path}")
        return []

    # Walk through all files (may be flat or in subfolders)
    for root, dirs, filenames in os.walk(data_path):
        for fname in filenames:
            if not fname.endswith('.wav'):
                continue

            # DC_a01.wav → remove extension → DC_a01
            # split by '_' → ['DC', 'a01']
            name_no_ext = fname.replace('.wav', '')
            parts       = name_no_ext.split('_')

            if len(parts) < 2:
                continue

            # Emotion part: 'a01', 'sa01', 'su01' etc.
            emotion_part = parts[1]
            # e.g. 'a01', 'sa01', 'h02'

            # Determine emotion code
            # Check two-char codes first (sa, su)
            # then single char codes
            emotion_code = None

            if emotion_part.startswith('sa'):
                emotion_code = 'sa'
            elif emotion_part.startswith('su'):
                emotion_code = 'su'
            elif emotion_part.startswith('a'):
                emotion_code = 'a'
            elif emotion_part.startswith('d'):
                emotion_code = 'd'
            elif emotion_part.startswith('f'):
                emotion_code = 'f'
            elif emotion_part.startswith('h'):
                emotion_code = 'h'
            elif emotion_part.startswith('n'):
                emotion_code = 'n'

            if emotion_code is None:
                continue

            emotion = savee_map.get(emotion_code)

            if emotion is None:
                # Skip surprised
                continue

            files.append({
                'file_path': os.path.join(root, fname),
                'emotion'  : emotion,
                'label'    : UNIFIED_EMOTIONS[emotion],
                'source'   : 'savee'
            })

    print(f"  ✅ SAVEE    : {len(files)} files")
    return files


# ══════════════════════════════════════════════════════
# COMBINE ALL DATASETS
# ══════════════════════════════════════════════════════

def load_all_datasets():
    """
    Loads and combines all 4 datasets into
    one unified DataFrame.

    Returns:
        df : DataFrame with columns:
             file_path, emotion, label, source
    """
    print("=" * 55)
    print("   Loading All Datasets")
    print("=" * 55)
    print()

    # Load each dataset
    all_files = []
    all_files.extend(load_ravdess())
    all_files.extend(load_crema_d())
    all_files.extend(load_tess())
    all_files.extend(load_savee())

    # Create DataFrame
    df = pd.DataFrame(all_files)

    # ── Summary ────────────────────────────────────────
    print()
    print("=" * 55)
    print("   Dataset Summary")
    print("=" * 55)

    print(f"\n── Files per source ────────────────────")
    for source in ['ravdess', 'crema_d', 'tess', 'savee']:
        count = len(df[df['source'] == source])
        bar   = '█' * (count // 100)
        print(f"  {source:<12}: {count:>5} files  {bar}")

    print(f"\n  {'─'*35}")
    print(f"  {'TOTAL':<12}: {len(df):>5} files")

    print(f"\n── Files per emotion ───────────────────")
    for emotion in sorted(df['emotion'].unique()):
        count = len(df[df['emotion'] == emotion])
        bar   = '█' * (count // 100)
        print(f"  {emotion:<12}: {count:>5} files  {bar}")

    print(f"\n── Label mapping ───────────────────────")
    for emotion, label in UNIFIED_EMOTIONS.items():
        print(f"  {label} → {emotion}")

    # ── Verify no missing files ────────────────────────
    print(f"\n── Verifying files exist ───────────────")
    missing = 0
    for path in df['file_path']:
        if not os.path.exists(path):
            missing += 1
    if missing == 0:
        print(f"  ✅ All {len(df)} files verified!")
    else:
        print(f"  ⚠️  {missing} files not found!")

    # ── Save combined CSV ──────────────────────────────
    csv_path = 'data/all_datasets.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Combined CSV saved to: {csv_path}")
    print(f"   This will be used by extract_features.py")

    print(f"\n{'='*55}")
    print(f"   All Datasets Loaded! 🎉")
    print(f"{'='*55}")

    return df


if __name__ == '__main__':
    df = load_all_datasets()