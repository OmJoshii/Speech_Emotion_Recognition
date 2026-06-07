# Exploring preprocessing concepts visually

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

print("=" * 50)
print("        Preprocessing Exploration")
print("=" * 50)

# ── Load extracted features ────────────────────────────
print("\nLoading extracted features...")

X_fixed  = np.load('data/features/X_fixed.npy')
X_mel    = np.load('data/features/X_mel.npy')
X_mfcc   = np.load('data/features/X_mfcc.npy')
y_labels = np.load('data/features/y_labels.npy')

print(f"X_fixed  shape : {X_fixed.shape}")
print(f"X_mel    shape : {X_mel.shape}")
print(f"X_mfcc   shape : {X_mfcc.shape}")
print(f"y_labels shape : {y_labels.shape}")

# ── Emotion label mapping ──────────────────────────────
emotion_names = {
    0: 'neutral',  1: 'calm',     2: 'happy',
    3: 'sad',      4: 'angry',    5: 'fearful',
    6: 'disgust',  7: 'surprised'
}


# ── Class Imbalance Visualization ─────────────────────
print("\n── Class Distribution ──")
class_counts = Counter(y_labels)
for label, count in sorted(class_counts.items()):
    bar = '█' * (count // 10)
    print(f"  {emotion_names[label]:<10} ({label}): "
          f"{count:>4} files  {bar}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Class Distribution Analysis', fontsize=13,
             fontweight='bold')

# Before handling imbalance
labels_list  = [emotion_names[i] for i in sorted(class_counts.keys())]
counts_list  = [class_counts[i]  for i in sorted(class_counts.keys())]
colors = ['coral' if c < 300 else 'steelblue' for c in counts_list]

axes[0].bar(labels_list, counts_list, color=colors, edgecolor='white')
axes[0].set_title('Raw Class Distribution\n(Red = imbalanced class)')
axes[0].set_xlabel('Emotion')
axes[0].set_ylabel('Number of Files')
axes[0].tick_params(axis='x', rotation=45)
for i, count in enumerate(counts_list):
    axes[0].text(i, count + 3, str(count),
                 ha='center', fontsize=9)

# Class weights visualization
# sklearn can compute these automatically
from sklearn.utils.class_weight import compute_class_weight

classes    = np.unique(y_labels)
weights    = compute_class_weight('balanced',
                                   classes=classes,
                                   y=y_labels)
weight_map = dict(zip(classes, weights))

weight_labels = [emotion_names[i] for i in sorted(weight_map.keys())]
weight_values = [weight_map[i]    for i in sorted(weight_map.keys())]
colors2 = ['coral' if w > 1.0 else 'steelblue' for w in weight_values]

axes[1].bar(weight_labels, weight_values, color=colors2, edgecolor='white')
axes[1].axhline(y=1.0, color='black', linestyle='--',
                linewidth=1, label='weight = 1.0 (balanced)')
axes[1].set_title('Class Weights\n(Higher weight = model penalized more)')
axes[1].set_xlabel('Emotion')
axes[1].set_ylabel('Weight')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()
for i, w in enumerate(weight_values):
    axes[1].text(i, w + 0.01, f'{w:.2f}',
                 ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('notebooks/class_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nClass distribution plot saved!")


# ── Normalization Visualization ────────────────────────
print("\n── Before vs After Normalization ──")

# Before normalization
print("\nBefore StandardScaler:")
print(f"  Mean  : {X_fixed.mean():.4f}")
print(f"  Std   : {X_fixed.std():.4f}")
print(f"  Min   : {X_fixed.min():.4f}")
print(f"  Max   : {X_fixed.max():.4f}")

# Apply StandardScaler
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_fixed)
# fit_transform does two things:
# fit      → calculates mean and std from data
# transform → applies the standardization formula

print("\nAfter StandardScaler:")
print(f"  Mean  : {X_scaled.mean():.4f}")
print(f"  Std   : {X_scaled.std():.4f}")
print(f"  Min   : {X_scaled.min():.4f}")
print(f"  Max   : {X_scaled.max():.4f}")

# Visualize distribution of first feature before/after
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Effect of StandardScaler on Feature Distribution',
             fontsize=12, fontweight='bold')

axes[0].hist(X_fixed[:, 0], bins=50, color='steelblue',
             edgecolor='white', alpha=0.7)
axes[0].set_title('Before Scaling\n(MFCC feature 1)')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

axes[1].hist(X_scaled[:, 0], bins=50, color='coral',
             edgecolor='white', alpha=0.7)
axes[1].set_title('After Scaling\n(mean=0, std=1)')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('notebooks/normalization.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Normalization comparison saved!")


# ── Train/Val/Test Split Visualization ────────────────
print("\n── Train / Val / Test Split ──")

# Split 1: separate test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_fixed, y_labels,
    test_size=0.15,
    random_state=42,       # for reproducibility
    stratify=y_labels      # keep emotion ratio same in each split
)
# stratify=y_labels ensures each split has proportional
# representation of all emotion classes

# Split 2: separate validation from remaining (15% of total)
# 0.15 / 0.85 ≈ 0.176 gives us 15% of total
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.176,
    random_state=42,
    stratify=y_temp
)

print(f"  Total files  : {len(X_fixed)}")
print(f"  Training     : {len(X_train)} "
      f"({len(X_train)/len(X_fixed)*100:.1f}%)")
print(f"  Validation   : {len(X_val)} "
      f"({len(X_val)/len(X_fixed)*100:.1f}%)")
print(f"  Test         : {len(X_test)} "
      f"({len(X_test)/len(X_fixed)*100:.1f}%)")

# Verify class balance is maintained in each split
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Class Distribution in Each Split\n'
             '(stratify ensures balanced splits)',
             fontsize=12, fontweight='bold')

for ax, (split_name, split_y) in zip(
    axes,
    [('Training', y_train),
     ('Validation', y_val),
     ('Test', y_test)]
):
    counts = Counter(split_y)
    names  = [emotion_names[i] for i in sorted(counts.keys())]
    vals   = [counts[i]        for i in sorted(counts.keys())]
    ax.bar(names, vals, color='steelblue', edgecolor='white')
    ax.set_title(f'{split_name} Set\n({len(split_y)} files)')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('notebooks/split_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Split distribution plot saved!")

print("\n✅ Preprocessing exploration complete!")