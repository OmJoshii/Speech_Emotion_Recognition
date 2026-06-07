# Understanding LSTM concepts visually

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

print("=" * 50)
print("       LSTM Exploration")
print("=" * 50)

# ── Load MFCC sequences ────────────────────────────────
print("\nLoading MFCC sequences...")

X_mfcc  = np.load('data/processed/X_mfcc_train.npy')
y_train = np.load('data/processed/y_train.npy')

print(f"MFCC sequence shape: {X_mfcc.shape}")
print(f"  Samples    : {X_mfcc.shape[0]}")
print(f"  Time steps : {X_mfcc.shape[1]}")
print(f"  Features   : {X_mfcc.shape[2]}")

emotion_names = ['neutral', 'calm',    'happy',    'sad',
                 'angry',   'fearful', 'disgust',  'surprised']

# ── Visualize MFCC sequences per emotion ───────────────
print("\nVisualizing MFCC sequences...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('MFCC Sequences — What LSTM Processes\n'
             '(Each column = one time step, '
             'each row = one coefficient)',
             fontsize=12, fontweight='bold')

for emotion_idx in range(8):
    row = emotion_idx // 4
    col = emotion_idx % 4
    ax  = axes[row, col]

    # Find first sample of this emotion
    sample_idx = np.where(y_train == emotion_idx)[0][0]

    # Shape: (130, 40) — transpose to (40, 130) for display
    mfcc_seq = X_mfcc[sample_idx].T

    ax.imshow(mfcc_seq, aspect='auto',
              cmap='coolwarm', origin='lower')
    ax.set_title(f'{emotion_names[emotion_idx].upper()}')
    ax.set_xlabel('Time steps →')
    ax.set_ylabel('MFCC coefficients')

plt.tight_layout()
plt.savefig('notebooks/mfcc_sequences.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("MFCC sequences saved!")

# ── Demonstrate LSTM processing ────────────────────────
print("\nDemonstrating LSTM processing...")

# Simple LSTM to show how it processes sequences
lstm = nn.LSTM(
    input_size=40,    # 40 MFCC features per time step
    hidden_size=128,  # 128 hidden units
    num_layers=1,
    batch_first=True  # (batch, time, features) order
)

# Take one sample
sample = torch.FloatTensor(X_mfcc[0:1])  # (1, 130, 40)

# Process through LSTM
with torch.no_grad():
    output, (hidden, cell) = lstm(sample)

print(f"\nLSTM input shape  : {sample.shape}")
print(f"  (batch=1, time_steps=130, features=40)")
print(f"\nLSTM output shape : {output.shape}")
print(f"  (batch=1, time_steps=130, hidden=128)")
print(f"  → one hidden state per time step")
print(f"\nFinal hidden state: {hidden.shape}")
print(f"  (layers=1, batch=1, hidden=128)")
print(f"  → this summarizes the ENTIRE sequence!")
print(f"  → this gets passed to FC layer for prediction")

# ── Visualize hidden states over time ─────────────────
# This shows how LSTM's memory evolves as it reads audio
hidden_states = output.squeeze().numpy()  # (130, 128)

plt.figure(figsize=(14, 5))
plt.imshow(hidden_states.T[:20],  # show first 20 units
           aspect='auto', cmap='RdBu', origin='lower')
plt.colorbar(label='Activation')
plt.title('LSTM Hidden States Over Time\n'
          '(Each row = one LSTM unit, '
          'each column = one time step)\n'
          'This is how LSTM "remembers" the audio sequence',
          fontsize=11)
plt.xlabel('Time steps (130 frames of audio)')
plt.ylabel('LSTM hidden units (showing first 20)')
plt.tight_layout()
plt.savefig('notebooks/lstm_hidden_states.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("LSTM hidden states visualization saved!")
print("\nExploration complete!")