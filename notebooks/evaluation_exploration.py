# Exploring evaluation metrics visually

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import os
from sklearn.metrics import (confusion_matrix,
                              classification_report,
                              roc_curve, auc)
from sklearn.preprocessing import label_binarize

print("=" * 50)
print("   Phase 8: Evaluation Exploration")
print("=" * 50)

EMOTION_NAMES = ['neutral', 'calm',    'happy',    'sad',
                 'angry',   'fearful', 'disgust',  'surprised']

# ── Load all model results ─────────────────────────────
print("\nLoading model results...")

results = {}
result_files = {
    'MLP'     : 'models/mlp_results.json',
    'CNN'     : 'models/cnn_results.json',
    'LSTM'    : 'models/lstm_results.json',
    'CNN-LSTM': 'models/cnn_lstm_results.json'
}

for model_name, path in result_files.items():
    if os.path.exists(path):
        with open(path) as f:
            results[model_name] = json.load(f)
        print(f"  ✅ {model_name}: "
              f"{results[model_name]['test_accuracy']}%")

# ── Model Comparison Bar Chart ─────────────────────────
print("\nPlotting model comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance Comparison',
             fontsize=13, fontweight='bold')

models      = list(results.keys())
accuracies  = [results[m]['test_accuracy'] for m in models]
parameters  = [results[m]['parameters']    for m in models]
colors      = ['steelblue', 'coral', 'green', 'purple']

# Accuracy comparison
bars = axes[0].bar(models, accuracies,
                   color=colors, edgecolor='white',
                   alpha=0.85)
axes[0].set_title('Test Accuracy Comparison')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([85, 100])
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.1,
        f'{acc}%',
        ha='center', va='bottom',
        fontweight='bold', fontsize=10
    )

# Add baseline reference line
axes[0].axhline(y=max(accuracies),
                color='gold', linestyle='--',
                linewidth=1.5, label=f'Best: {max(accuracies)}%')
axes[0].legend()

# Parameters vs accuracy scatter
axes[1].scatter(parameters, accuracies,
                c=colors, s=200, alpha=0.85,
                edgecolors='white', linewidth=2)
for i, model in enumerate(models):
    axes[1].annotate(
        model,
        (parameters[i], accuracies[i]),
        textcoords='offset points',
        xytext=(10, 5),
        fontsize=10
    )
axes[1].set_title('Parameters vs Accuracy\n'
                  '(More params ≠ better accuracy!)')
axes[1].set_xlabel('Number of Parameters')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Model comparison saved!")

# ── Precision Recall F1 explanation ───────────────────
print("\nDemonstrating Precision/Recall/F1...")

# Simulate a simple example
TP = 55   # True Positives  — correctly predicted happy
FP = 3    # False Positives — wrongly predicted as happy
FN = 2    # False Negatives — happy missed

precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * (precision * recall) / (precision + recall)

print(f"\nExample for 'Happy' emotion:")
print(f"  True Positives  (correct happy)    : {TP}")
print(f"  False Positives (wrong happy pred) : {FP}")
print(f"  False Negatives (missed happy)     : {FN}")
print(f"\n  Precision : {precision:.3f} "
      f"({precision*100:.1f}% of happy predictions correct)")
print(f"  Recall    : {recall:.3f} "
      f"({recall*100:.1f}% of actual happy detected)")
print(f"  F1 Score  : {f1:.3f}")

print("\nExploration complete!")