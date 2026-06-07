# Understanding CNN concepts visually

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

print("=" * 50)
print("   Phase 6: CNN Exploration")
print("=" * 50)

# ── Load a few mel spectrograms ────────────────────────
print("\nLoading mel spectrograms...")

X_mel   = np.load('data/processed/X_mel_train.npy')
y_train = np.load('data/processed/y_train.npy')

emotion_names = ['neutral', 'calm',    'happy',    'sad',
                 'angry',   'fearful', 'disgust',  'surprised']

print(f"Mel spectrogram shape: {X_mel.shape}")
print(f"(samples, channels, mel_bands, time_frames)")

# ── Visualize spectrograms the CNN will see ────────────
print("\nVisualizing spectrograms per emotion...")

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle('Mel Spectrograms — What the CNN Sees\n'
             '(Each image = one audio file)',
             fontsize=13, fontweight='bold')

for emotion_idx in range(8):
    row = emotion_idx // 4
    col = emotion_idx % 4
    ax  = axes[row, col]

    # Find first sample of this emotion
    sample_idx = np.where(y_train == emotion_idx)[0][0]

    # X_mel shape is (samples, 1, 128, 130)
    # [0] removes the channel dimension for display
    spectrogram = X_mel[sample_idx, 0, :, :]

    ax.imshow(spectrogram,
              aspect='auto',
              origin='lower',
              cmap='magma')
    ax.set_title(f'{emotion_names[emotion_idx].upper()}')
    ax.set_xlabel('Time frames')
    ax.set_ylabel('Mel bands')

plt.tight_layout()
plt.savefig('notebooks/phase6_spectrograms.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Spectrogram visualization saved!")

# ── Visualize what conv filters detect ────────────────
print("\nVisualizing convolution concept...")

# Create a simple example filter
# This is an edge detection filter
edge_filter = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# Take one spectrogram
sample_spec = X_mel[0, 0, :, :]  # (128, 130)

# Apply filter manually using PyTorch conv2d
spec_tensor = torch.FloatTensor(
    sample_spec
).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 130)

filter_tensor = torch.FloatTensor(
    edge_filter
).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

# Apply convolution
with torch.no_grad():
    filtered = nn.functional.conv2d(
        spec_tensor, filter_tensor, padding=1
    )
filtered_np = filtered.squeeze().numpy()

# Plot original vs filtered
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('What a Conv Filter Does\n'
             '(This is ONE filter detecting edges)',
             fontsize=12, fontweight='bold')

axes[0].imshow(sample_spec, aspect='auto',
               origin='lower', cmap='magma')
axes[0].set_title('Original Mel Spectrogram')
axes[0].set_xlabel('Time frames')
axes[0].set_ylabel('Mel bands')

axes[1].imshow(filtered_np, aspect='auto',
               origin='lower', cmap='RdBu')
axes[1].set_title('After Edge Detection Filter\n'
                  '(Bright = strong edge detected)')
axes[1].set_xlabel('Time frames')
axes[1].set_ylabel('Mel bands')

plt.tight_layout()
plt.savefig('notebooks/phase6_conv_filter.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Convolution visualization saved!")
print("\nExploration complete!")