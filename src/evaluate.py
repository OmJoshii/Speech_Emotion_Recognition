# Comprehensive evaluation and explainability

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import our model classes
import sys
sys.path.append('src')
from train_mlp  import EmotionMLP
from train_cnn  import EmotionCNN
from train_lstm import EmotionLSTM, EmotionCNNLSTM

from sklearn.metrics import (confusion_matrix,
                              classification_report,
                              accuracy_score)
from sklearn.inspection import permutation_importance

device = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')

EMOTION_NAMES = ['neutral', 'calm',    'happy',    'sad',
                 'angry',   'fearful', 'disgust',  'surprised']

NUM_CLASSES   = 8


# ══════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════

def load_test_data():
    """Loads all test set data."""
    print("Loading test data...")

    data = {
        'X_fixed' : np.load('data/processed/X_fixed_test.npy'),
        'X_mel'   : np.load('data/processed/X_mel_test.npy'),
        'X_mfcc'  : np.load('data/processed/X_mfcc_test.npy'),
        'y_test'  : np.load('data/processed/y_test.npy')
    }

    print(f"  Test samples : {len(data['y_test'])}")
    print(f"  Fixed shape  : {data['X_fixed'].shape}")
    print(f"  Mel shape    : {data['X_mel'].shape}")
    print(f"  MFCC shape   : {data['X_mfcc'].shape}")

    return data


def get_predictions(model, X, batch_size=32):
    """
    Gets predictions from a PyTorch model.

    Parameters:
        model      : trained PyTorch model
        X          : input numpy array
        batch_size : samples per batch

    Returns:
        predictions : numpy array of predicted classes
        probabilities: numpy array of class probabilities
    """
    model.eval()
    all_preds = []
    all_probs = []

    # Process in batches to avoid memory issues
    for i in range(0, len(X), batch_size):
        batch = torch.FloatTensor(
            X[i:i+batch_size]
        ).to(device)

        with torch.no_grad():
            outputs = model(batch)
            # Softmax converts raw scores to probabilities
            probs   = torch.softmax(outputs, dim=1)
            preds   = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


# ══════════════════════════════════════════════════════
# CONFUSION MATRIX — DETAILED
# ══════════════════════════════════════════════════════

def plot_detailed_confusion_matrix(y_true, y_pred,
                                    model_name, save_path):
    """
    Plots a detailed confusion matrix with both
    raw counts AND percentages.
    """
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / \
             cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'{model_name} — Detailed Confusion Matrix',
                 fontsize=13, fontweight='bold')

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Blues',
                xticklabels=EMOTION_NAMES,
                yticklabels=EMOTION_NAMES,
                ax=axes[0])
    axes[0].set_title('Raw Counts')
    axes[0].set_ylabel('True Emotion')
    axes[0].set_xlabel('Predicted Emotion')
    axes[0].tick_params(axis='x', rotation=45)

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt='.1f',
                cmap='Blues',
                xticklabels=EMOTION_NAMES,
                yticklabels=EMOTION_NAMES,
                vmin=0, vmax=100,
                ax=axes[1])
    axes[1].set_title('Percentages (%)')
    axes[1].set_ylabel('True Emotion')
    axes[1].set_xlabel('Predicted Emotion')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════
# FEATURE IMPORTANCE — SHAP-LIKE ANALYSIS
# ══════════════════════════════════════════════════════

def analyze_feature_importance(model, X_test, y_test):
    """
    Analyzes which features are most important
    for emotion prediction using gradient-based
    sensitivity analysis.

    This shows which MFCC coefficients, chroma features
    etc. contribute most to emotion predictions.
    """
    print("\nAnalyzing feature importance...")

    model.eval()

    # Feature names for our 94-dim fixed feature vector
    feature_names = (
        [f'MFCC_mean_{i+1}'  for i in range(40)] +
        [f'MFCC_std_{i+1}'   for i in range(40)] +
        [f'Chroma_{i+1}'     for i in range(12)] +
        ['ZCR_mean'] +
        ['RMS_mean']
    )

    # Calculate gradient-based importance
    # Gradients tell us how sensitive the output is to each input feature

    X_tensor = torch.FloatTensor(X_test).to(device)
    X_tensor.requires_grad_(True)

    # Forward pass
    outputs = model(X_tensor)

    # Calculate gradients for each class
    importance_per_class = []

    for class_idx in range(NUM_CLASSES):
        model.zero_grad()
        if X_tensor.grad is not None:
            X_tensor.grad.zero_()

        # Backpropagate for this class
        outputs[:, class_idx].sum().backward(
            retain_graph=True
        )

        # Absolute gradient = importance
        grad = X_tensor.grad.abs().mean(dim=0)
        importance_per_class.append(
            grad.cpu().detach().numpy()
        )

    importance_matrix = np.array(importance_per_class)
    # Shape: (8 classes, 94 features)

    # Overall importance = mean across all classes
    overall_importance = importance_matrix.mean(axis=0)

    # Get top 20 most important features
    top_indices  = np.argsort(overall_importance)[-20:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_values   = overall_importance[top_indices]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Feature Importance Analysis\n'
                 '(Which features matter most for emotion?)',
                 fontsize=12, fontweight='bold')

    # Top 20 overall features
    colors_bar = ['steelblue' if 'MFCC' in f
                  else 'coral' if 'Chroma' in f
                  else 'green'
                  for f in top_features]

    axes[0].barh(range(20), top_values[::-1],
                 color=colors_bar[::-1],
                 edgecolor='white')
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels(top_features[::-1], fontsize=8)
    axes[0].set_title('Top 20 Most Important Features')
    axes[0].set_xlabel('Importance Score')

    # Feature importance heatmap per emotion
    im = axes[1].imshow(
        importance_matrix[:, top_indices[:15]],
        aspect='auto', cmap='YlOrRd'
    )
    axes[1].set_xticks(range(15))
    axes[1].set_xticklabels(
        [f.replace('MFCC_', '').replace('_mean', '')
         for f in top_features[:15]],
        rotation=45, ha='right', fontsize=7
    )
    axes[1].set_yticks(range(NUM_CLASSES))
    axes[1].set_yticklabels(EMOTION_NAMES)
    axes[1].set_title('Feature Importance per Emotion\n'
                      '(Brighter = more important)')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig('models/feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Feature importance saved!")

    # Print top 10
    print("\nTop 10 most important features:")
    for i, (feat, val) in enumerate(
        zip(top_features[:10], top_values[:10])
    ):
        bar = ' ' * int(val * 1000)
        print(f"  {i+1:>2}. {feat:<20} {val:.5f}  {bar}")

    return overall_importance, feature_names


# ══════════════════════════════════════════════════════
# GRAD-CAM FOR CNN
# ══════════════════════════════════════════════════════

def gradcam_cnn(model, X_sample, true_label):
    """
    Generates Grad-CAM visualization for CNN.
    Shows which parts of the spectrogram the CNN
    focused on when making its prediction.

    Parameters:
        model      : trained CNN model
        X_sample   : one spectrogram (1, 1, 128, 130)
        true_label : actual emotion index
    """
    model.eval()

    # Storage for gradients and activations
    gradients   = []
    activations = []

    # Hook functions capture intermediate values
    # during forward/backward pass
    def save_gradient(grad):
        gradients.append(grad)

    def save_activation(module, input, output):
        activations.append(output)
        # Register gradient hook on output
        output.register_hook(save_gradient)

    # Attach hook to last conv block
    hook = model.conv_block4.register_forward_hook(
        save_activation
    )

    # Forward pass
    input_tensor = torch.FloatTensor(X_sample).to(device)
    input_tensor.requires_grad_(True)
    output = model(input_tensor)

    predicted_class = output.argmax(dim=1).item()

    # Backward pass for predicted class
    model.zero_grad()
    output[0, predicted_class].backward()

    # Remove hook
    hook.remove()

    # Calculate Grad-CAM
    # Global average pool the gradients
    grad     = gradients[0]          # (1, C, H, W)
    act      = activations[0]        # (1, C, H, W)

    weights  = grad.mean(dim=[2, 3], keepdim=True)
    # weights: importance of each filter

    # Weighted combination of activation maps
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)            # only positive
    cam = cam.squeeze().cpu().detach().numpy()

    # Normalize to 0-1
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Resize cam to match spectrogram size
    from torch.nn.functional import interpolate
    cam_tensor = torch.FloatTensor(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = interpolate(
        cam_tensor,
        size=(128, 130),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f'Grad-CAM Visualization\n'
        f'True: {EMOTION_NAMES[true_label]}  |  '
        f'Predicted: {EMOTION_NAMES[predicted_class]}',
        fontsize=11, fontweight='bold'
    )

    spec = X_sample[0, 0]  # (128, 130)

    # Original spectrogram
    axes[0].imshow(spec, aspect='auto',
                   origin='lower', cmap='magma')
    axes[0].set_title('Original Mel Spectrogram')
    axes[0].set_xlabel('Time frames')
    axes[0].set_ylabel('Mel bands')

    # Grad-CAM heatmap
    axes[1].imshow(cam_resized, aspect='auto',
                   origin='lower', cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n'
                      '(Red = CNN focused here)')
    axes[1].set_xlabel('Time frames')
    axes[1].set_ylabel('Mel bands')

    # Overlay
    axes[2].imshow(spec, aspect='auto',
                   origin='lower', cmap='magma',
                   alpha=0.6)
    axes[2].imshow(cam_resized, aspect='auto',
                   origin='lower', cmap='jet',
                   alpha=0.5)
    axes[2].set_title('Overlay\n'
                      '(What CNN sees)')
    axes[2].set_xlabel('Time frames')
    axes[2].set_ylabel('Mel bands')

    plt.tight_layout()
    plt.savefig('models/gradcam_visualization.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Grad-CAM saved!")

    return predicted_class


# ══════════════════════════════════════════════════════
# ATTENTION VISUALIZATION FOR LSTM
# ══════════════════════════════════════════════════════

def visualize_lstm_attention(model, X_sample,
                              true_label):
    """
    Visualizes attention weights from LSTM model.
    Shows which time steps (moments in audio) the
    LSTM focused on most for its prediction.
    """
    model.eval()

    input_tensor = torch.FloatTensor(
        X_sample
    ).unsqueeze(0).to(device)  # (1, 130, 40)

    with torch.no_grad():
        # Get LSTM outputs
        lstm_out1, _ = model.lstm1(input_tensor)
        lstm_out1    = model.bn1(
            lstm_out1.permute(0, 2, 1)
        ).permute(0, 2, 1)

        lstm_out2, _ = model.lstm2(lstm_out1)
        lstm_out2    = model.bn2(
            lstm_out2.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # Get attention weights
        scores  = model.attention(lstm_out2)
        weights = torch.softmax(scores, dim=1)
        weights_np = weights.squeeze().cpu().numpy()

        # Final prediction
        context = (lstm_out2 * weights).sum(dim=1)
        output  = model.classifier(context)
        pred    = output.argmax(dim=1).item()

    # Plot attention weights over time
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f'LSTM Attention Visualization\n'
        f'True: {EMOTION_NAMES[true_label]}  |  '
        f'Predicted: {EMOTION_NAMES[pred]}',
        fontsize=11, fontweight='bold'
    )

    # MFCC sequence (input)
    axes[0].imshow(
        X_sample.T,    # (40, 130)
        aspect='auto', cmap='coolwarm',
        origin='lower'
    )
    axes[0].set_title('Input MFCC Sequence')
    axes[0].set_xlabel('Time steps')
    axes[0].set_ylabel('MFCC coefficients')

    # Attention weights
    axes[1].fill_between(
        range(len(weights_np)),
        weights_np,
        alpha=0.7,
        color='steelblue',
        label='Attention weight'
    )
    axes[1].plot(weights_np, color='navy', linewidth=1)

    # Highlight peak attention moments
    peak_idx = np.argmax(weights_np)
    axes[1].axvline(x=peak_idx, color='red',
                    linestyle='--', linewidth=1.5,
                    label=f'Peak attention at step {peak_idx}')
    axes[1].set_title(
        'Attention Weights Over Time\n'
        '(Higher = LSTM focused here more)'
    )
    axes[1].set_xlabel('Time steps (130 frames of audio)')
    axes[1].set_ylabel('Attention weight')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/lstm_attention.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("LSTM attention visualization saved!")

    return pred, weights_np


# ══════════════════════════════════════════════════════
# COMPREHENSIVE COMPARISON REPORT
# ══════════════════════════════════════════════════════

def generate_comparison_report(all_results):
    """
    Generates a comprehensive visual comparison
    of all 4 models side by side.
    """
    print("\nGenerating comparison report...")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        'Speech Emotion Recognition — Complete Model Comparison',
        fontsize=14, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.4, wspace=0.3)

    models      = list(all_results.keys())
    accuracies  = [all_results[m]['accuracy']    for m in models]
    f1_scores   = [all_results[m]['f1_macro']    for m in models]
    parameters  = [all_results[m]['parameters']  for m in models]
    colors      = ['steelblue', 'coral', 'green', 'purple']

    # ── Plot 1: Accuracy bar chart ─────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(models, accuracies,
                   color=colors, edgecolor='white',
                   alpha=0.85)
    ax1.set_title('Test Accuracy (%)', fontweight='bold')
    ax1.set_ylim([85, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{acc:.2f}%',
            ha='center', fontsize=8, fontweight='bold'
        )

    # ── Plot 2: F1 Score comparison ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(models, f1_scores,
                    color=colors, edgecolor='white',
                    alpha=0.85)
    ax2.set_title('Macro F1 Score', fontweight='bold')
    ax2.set_ylim([0.85, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, f1 in zip(bars2, f1_scores):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001,
            f'{f1:.3f}',
            ha='center', fontsize=8, fontweight='bold'
        )

    # ── Plot 3: Parameters vs accuracy ────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for i, model in enumerate(models):
        ax3.scatter(parameters[i], accuracies[i],
                    c=colors[i], s=200,
                    edgecolors='white', linewidth=2,
                    label=model, zorder=5)
        ax3.annotate(model,
                     (parameters[i], accuracies[i]),
                     textcoords='offset points',
                     xytext=(5, 5), fontsize=8)
    ax3.set_title('Parameters vs Accuracy\n'
                  '(More params ≠ better!)',
                  fontweight='bold')
    ax3.set_xlabel('Parameters')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)

    # ── Plot 4: Per-emotion F1 heatmap ────────────────
    ax4 = fig.add_subplot(gs[1, :2])

    f1_matrix = np.array(
        [all_results[m]['f1_per_emotion'] for m in models]
    )

    sns.heatmap(
        f1_matrix,
        annot=True, fmt='.2f',
        cmap='RdYlGn',
        xticklabels=EMOTION_NAMES,
        yticklabels=models,
        vmin=0.7, vmax=1.0,
        ax=ax4
    )
    ax4.set_title('F1 Score per Emotion per Model\n'
                  '(Green = good, Red = poor)',
                  fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    # ── Plot 5: Summary table ──────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    table_data = []
    headers    = ['Model', 'Accuracy', 'F1', 'Params']
    for model in models:
        table_data.append([
            model,
            f"{all_results[model]['accuracy']:.2f}%",
            f"{all_results[model]['f1_macro']:.3f}",
            f"{all_results[model]['parameters']:,}"
        ])

    table = ax5.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.2, 1, 0.7]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(4)))

    # Highlight best model row
    best_idx = accuracies.index(max(accuracies))
    for col in range(4):
        table[best_idx + 1, col].set_facecolor('#90EE90')

    ax5.set_title('Summary Table\n'
                  '(Green = best model)',
                  fontweight='bold')

    plt.savefig('models/final_comparison_report.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Final comparison report saved!")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("   Phase 8: Evaluation & Explainability")
    print("=" * 55)

    # ── Load test data ─────────────────────────────────
    data   = load_test_data()
    y_test = data['y_test']

    all_results = {}

    # ══════════════════════════════════════════════════
    # EVALUATE MLP
    # ══════════════════════════════════════════════════
    print("\n" + "="*55)
    print("Evaluating MLP...")
    print("="*55)

    mlp_model = EmotionMLP().to(device)
    mlp_model.load_state_dict(
        torch.load('models/mlp_best.pth',
                   map_location=device)
    )

    mlp_preds, mlp_probs = get_predictions(
        mlp_model, data['X_fixed']
    )

    mlp_report = classification_report(
        y_test, mlp_preds,
        target_names=EMOTION_NAMES,
        output_dict=True
    )

    print(classification_report(
        y_test, mlp_preds,
        target_names=EMOTION_NAMES
    ))

    plot_detailed_confusion_matrix(
        y_test, mlp_preds,
        'MLP Model',
        'models/mlp_detailed_cm.png'
    )

    # Feature importance for MLP
    mlp_importance, feat_names = analyze_feature_importance(
        mlp_model, data['X_fixed'][:100], y_test[:100]
    )

    all_results['MLP'] = {
        'accuracy'       : mlp_report['accuracy'] * 100,
        'f1_macro'       : mlp_report['macro avg']['f1-score'],
        'f1_per_emotion' : [mlp_report[e]['f1-score']
                            for e in EMOTION_NAMES],
        'parameters'     : sum(p.numel()
                               for p in mlp_model.parameters())
    }

    # ══════════════════════════════════════════════════
    # EVALUATE CNN
    # ══════════════════════════════════════════════════
    print("\n" + "="*55)
    print("Evaluating CNN...")
    print("="*55)

    cnn_model = EmotionCNN().to(device)
    cnn_model.load_state_dict(
        torch.load('models/cnn_best.pth',
                   map_location=device)
    )

    cnn_preds, cnn_probs = get_predictions(
        cnn_model, data['X_mel']
    )

    cnn_report = classification_report(
        y_test, cnn_preds,
        target_names=EMOTION_NAMES,
        output_dict=True
    )

    print(classification_report(
        y_test, cnn_preds,
        target_names=EMOTION_NAMES
    ))

    plot_detailed_confusion_matrix(
        y_test, cnn_preds,
        'CNN Model',
        'models/cnn_detailed_cm.png'
    )

    # Grad-CAM for one sample per emotion
    print("\nGenerating Grad-CAM visualizations...")
    for emotion_idx in range(NUM_CLASSES):
        sample_indices = np.where(y_test == emotion_idx)[0]
        if len(sample_indices) > 0:
            sample = data['X_mel'][sample_indices[0]:
                                   sample_indices[0]+1]
            gradcam_cnn(cnn_model, sample, emotion_idx)
            break  # just one example for now

    all_results['CNN'] = {
        'accuracy'       : cnn_report['accuracy'] * 100,
        'f1_macro'       : cnn_report['macro avg']['f1-score'],
        'f1_per_emotion' : [cnn_report[e]['f1-score']
                            for e in EMOTION_NAMES],
        'parameters'     : sum(p.numel()
                               for p in cnn_model.parameters())
    }

    # ══════════════════════════════════════════════════
    # EVALUATE LSTM
    # ══════════════════════════════════════════════════
    print("\n" + "="*55)
    print("Evaluating LSTM...")
    print("="*55)

    lstm_model = EmotionLSTM().to(device)
    lstm_model.load_state_dict(
        torch.load('models/lstm_best.pth',
                   map_location=device)
    )

    lstm_preds, lstm_probs = get_predictions(
        lstm_model, data['X_mfcc']
    )

    lstm_report = classification_report(
        y_test, lstm_preds,
        target_names=EMOTION_NAMES,
        output_dict=True
    )

    print(classification_report(
        y_test, lstm_preds,
        target_names=EMOTION_NAMES
    ))

    plot_detailed_confusion_matrix(
        y_test, lstm_preds,
        'LSTM Model',
        'models/lstm_detailed_cm.png'
    )

    # Attention visualization for LSTM
    print("\nGenerating LSTM attention visualization...")
    sample_idx  = np.where(y_test == 4)[0][0]  # angry sample
    lstm_sample = data['X_mfcc'][sample_idx]
    visualize_lstm_attention(
        lstm_model, lstm_sample, y_test[sample_idx]
    )

    all_results['LSTM'] = {
        'accuracy'       : lstm_report['accuracy'] * 100,
        'f1_macro'       : lstm_report['macro avg']['f1-score'],
        'f1_per_emotion' : [lstm_report[e]['f1-score']
                            for e in EMOTION_NAMES],
        'parameters'     : sum(p.numel()
                               for p in lstm_model.parameters())
    }

    # ══════════════════════════════════════════════════
    # EVALUATE CNN-LSTM
    # ══════════════════════════════════════════════════
    print("\n" + "="*55)
    print("Evaluating CNN-LSTM...")
    print("="*55)

    cnnlstm_model = EmotionCNNLSTM().to(device)
    cnnlstm_model.load_state_dict(
        torch.load('models/cnn_lstm_best.pth',
                   map_location=device)
    )

    cnnlstm_preds, _ = get_predictions(
        cnnlstm_model, data['X_mel']
    )

    cnnlstm_report = classification_report(
        y_test, cnnlstm_preds,
        target_names=EMOTION_NAMES,
        output_dict=True
    )

    print(classification_report(
        y_test, cnnlstm_preds,
        target_names=EMOTION_NAMES
    ))

    plot_detailed_confusion_matrix(
        y_test, cnnlstm_preds,
        'CNN-LSTM Model',
        'models/cnnlstm_detailed_cm.png'
    )

    all_results['CNN-LSTM'] = {
        'accuracy'       : cnnlstm_report['accuracy'] * 100,
        'f1_macro'       : cnnlstm_report['macro avg']['f1-score'],
        'f1_per_emotion' : [cnnlstm_report[e]['f1-score']
                            for e in EMOTION_NAMES],
        'parameters'     : sum(p.numel()
                               for p in cnnlstm_model.parameters())
    }

    # ══════════════════════════════════════════════════
    # GENERATE FINAL COMPARISON REPORT
    # ══════════════════════════════════════════════════
    generate_comparison_report(all_results)

    # ── Save complete evaluation results ───────────────
    eval_summary = {
        model: {
            'accuracy'  : results['accuracy'],
            'f1_macro'  : results['f1_macro'],
            'parameters': results['parameters']
        }
        for model, results in all_results.items()
    }

    with open('models/evaluation_summary.json', 'w') as f:
        json.dump(eval_summary, f, indent=4)

    # ── Final printed summary ──────────────────────────
    print(f"\n{'='*55}")
    print(f"   FINAL EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"\n  {'Model':<12} {'Accuracy':>10} "
          f"{'F1 Macro':>10} {'Params':>12}")
    print(f"  {'-'*48}")

    for model, res in sorted(
        all_results.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    ):
        marker = ' ✅ BEST' if res['accuracy'] == max(
            r['accuracy'] for r in all_results.values()
        ) else ''
        print(f"  {model:<12} "
              f"{res['accuracy']:>9.2f}% "
              f"{res['f1_macro']:>10.3f} "
              f"{res['parameters']:>12,}"
              f"{marker}")

    print(f"\n{'='*55}")
    print(f"   Phase 8 Complete! 🎉")
    print(f"{'='*55}")
    print(f"\n  All evaluation files saved to models/")


if __name__ == '__main__':
    main()