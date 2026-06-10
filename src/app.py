# Live Speech Emotion Recognition Demo App

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for streamlit
import sounddevice as sd
import scipy.io.wavfile as wav
import os
import sys
import time
import joblib
import io
import warnings
warnings.filterwarnings('ignore')

# Add src to path so we can import our models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_mlp  import EmotionMLP
from train_lstm import EmotionLSTM

# ── Constants ──────────────────────────────────────────
SAMPLE_RATE  = 22050
DURATION     = 3.0
FIXED_LENGTH = int(SAMPLE_RATE * DURATION)
N_MFCC       = 40
HOP_LENGTH   = 512
N_FFT        = 2048
NUM_CLASSES  = 6

# Emotion display config
EMOTIONS = {
    0: {'name': 'Neutral',  'emoji': '😐', 'color': '#808080'},
    1: {'name': 'Happy',    'emoji': '😄', 'color': '#FFC107'},
    2: {'name': 'Sad',      'emoji': '😢', 'color': '#2196F3'},
    3: {'name': 'Angry',    'emoji': '😡', 'color': '#F44336'},
    4: {'name': 'Fearful',  'emoji': '😨', 'color': '#9C27B0'},
    5: {'name': 'Disgust',  'emoji': '🤢', 'color': '#795548'},
}

device = torch.device('cpu')  # always CPU for demo app


# ══════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """
    Loads all trained models.
    @st.cache_resource caches the models so they're
    only loaded ONCE — not every time app reruns.
    """
    models = {}

    # Load LSTM (best model)
    try:
        lstm = EmotionLSTM().to(device)
        lstm.load_state_dict(
            torch.load('models/lstm_best.pth',
                       map_location=device)
        )
        lstm.eval()
        models['LSTM (Best — 96.53%)'] = {
            'model'     : lstm,
            'input_type': 'mfcc'
        }
    except Exception as e:
        st.error(f"Could not load LSTM: {e}")

    # Load MLP
    try:
        mlp = EmotionMLP().to(device)
        mlp.load_state_dict(
            torch.load('models/mlp_best.pth',
                       map_location=device)
        )
        mlp.eval()
        models['MLP (93.75%)'] = {
            'model'     : mlp,
            'input_type': 'fixed'
        }
    except Exception as e:
        st.error(f"Could not load MLP: {e}")

    return models


@st.cache_resource
def load_scaler():
    """Loads the fitted StandardScaler for MLP."""
    try:
        return joblib.load('models/scaler.pkl')
    except:
        return None


@st.cache_resource
def load_norm_stats():
    """Loads MFCC normalization statistics."""
    try:
        mean = np.load('models/mfcc_mean.npy')[0]
        std  = np.load('models/mfcc_std.npy')[0]
        return mean, std
    except:
        return 0.0, 1.0


# ══════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════

def preprocess_audio(y, sr):
    """
    Preprocesses raw audio for model input.
    Same steps as training — must match exactly!

    Parameters:
        y  : raw audio signal
        sr : sample rate

    Returns:
        fixed_features : (94,) array for MLP
        mfcc_sequence  : (130, 40) array for LSTM
    """
    # Resample if needed
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr,
                             target_sr=SAMPLE_RATE)

    # Pad or trim to exactly 3 seconds
    if len(y) < FIXED_LENGTH:
        y = np.pad(y, (0, FIXED_LENGTH - len(y)))
    else:
        y = y[:FIXED_LENGTH]

    # ── Fixed features for MLP ─────────────────────────
    mfcc   = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE,
                                   n_mfcc=N_MFCC,
                                   hop_length=HOP_LENGTH,
                                   n_fft=N_FFT)
    chroma = librosa.feature.chroma_stft(y=y,
                                          sr=SAMPLE_RATE,
                                          hop_length=HOP_LENGTH,
                                          n_fft=N_FFT)
    zcr    = librosa.feature.zero_crossing_rate(
        y, hop_length=HOP_LENGTH
    )
    rms    = librosa.feature.rms(y=y,
                                  hop_length=HOP_LENGTH)

    fixed_features = np.hstack([
        np.mean(mfcc,   axis=1),
        np.std(mfcc,    axis=1),
        np.mean(chroma, axis=1),
        [np.mean(zcr)],
        [np.mean(rms)]
    ])

    # ── MFCC sequence for LSTM ─────────────────────────
    mfcc_sequence = mfcc.T  # (130, 40)

    return fixed_features, mfcc_sequence, y


def normalize_features(fixed_features, mfcc_sequence,
                        scaler, mfcc_mean, mfcc_std):
    """
    Normalizes features using saved training statistics.
    Must use same normalization as training!
    """
    # Normalize fixed features using scaler
    if scaler is not None:
        fixed_norm = scaler.transform(
            fixed_features.reshape(1, -1)
        )[0]
    else:
        fixed_norm = fixed_features

    # Normalize MFCC sequence
    mfcc_norm = (mfcc_sequence - mfcc_mean) / (mfcc_std + 1e-8)

    return fixed_norm, mfcc_norm


# ══════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════

def predict_emotion(model_info, fixed_features,
                    mfcc_sequence):
    """
    Runs emotion prediction using selected model.

    Parameters:
        model_info     : dict with model and input_type
        fixed_features : (94,) normalized features
        mfcc_sequence  : (130, 40) normalized sequence

    Returns:
        predicted_class : int (0-7)
        probabilities   : (8,) confidence scores
    """
    model      = model_info['model']
    input_type = model_info['input_type']

    with torch.no_grad():
        if input_type == 'mfcc':
            # LSTM input: (1, 130, 40)
            x = torch.FloatTensor(
                mfcc_sequence
            ).unsqueeze(0).to(device)

        elif input_type == 'fixed':
            # MLP input: (1, 94)
            x = torch.FloatTensor(
                fixed_features
            ).unsqueeze(0).to(device)

        output = model(x)
        probs  = torch.softmax(output, dim=1)
        pred   = torch.argmax(output, dim=1).item()

    return pred, probs.squeeze().cpu().numpy()


# ══════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════

def plot_waveform(y):
    """Plots audio waveform."""
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    time_axis = np.linspace(0, DURATION, len(y))
    ax.plot(time_axis, y, color='#00D4FF',
            linewidth=0.8, alpha=0.9)
    ax.fill_between(time_axis, y, alpha=0.3,
                    color='#00D4FF')
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Audio Waveform', color='white',
                 fontsize=10)
    plt.tight_layout()
    return fig


def plot_mfcc(mfcc_sequence):
    """Plots MFCC heatmap."""
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    im = ax.imshow(
        mfcc_sequence.T,   # (40, 130)
        aspect='auto',
        origin='lower',
        cmap='coolwarm'
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Time steps', color='white')
    ax.set_ylabel('MFCC coefficients', color='white')
    ax.tick_params(colors='white')
    ax.set_title('MFCC Features (What LSTM sees)',
                 color='white', fontsize=10)
    plt.tight_layout()
    return fig


def plot_probabilities(probabilities):
    """Plots emotion probability bar chart."""
    emotion_names = [EMOTIONS[i]['name']
                     for i in range(NUM_CLASSES)]
    emotion_colors = [EMOTIONS[i]['color']
                      for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    bars = ax.barh(
        emotion_names,
        probabilities * 100,
        color=emotion_colors,
        edgecolor='none',
        alpha=0.85
    )

    # Add percentage labels
    for bar, prob in zip(bars, probabilities):
        if prob > 0.02:
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%',
                va='center', color='white',
                fontsize=9
            )

    ax.set_xlabel('Confidence (%)', color='white')
    ax.set_xlim([0, 110])
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Emotion Probabilities',
                 color='white', fontsize=10)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
# AUDIO RECORDING
# ══════════════════════════════════════════════════════

def record_audio(duration=3.0, sr=SAMPLE_RATE):
    """
    Records audio from microphone.

    Parameters:
        duration : recording length in seconds
        sr       : sample rate

    Returns:
        audio : numpy array of recorded audio
    """
    audio = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # wait for recording to finish
    return audio.flatten()


# ══════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════

def main():

    # ── Page Config ────────────────────────────────────
    st.set_page_config(
        page_title="Speech Emotion Recognition",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Custom CSS ─────────────────────────────────────
    st.markdown("""
        <style>
        .main { background-color: #0E1117; }
        .stButton>button {
            width: 100%;
            background: linear-gradient(
                135deg, #667eea 0%, #764ba2 100%
            );
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }
        .stButton>button:hover {
            background: linear-gradient(
                135deg, #764ba2 0%, #667eea 100%
            );
        }
        .emotion-card {
            background: linear-gradient(
                135deg, #1e1e2e 0%, #2d2d44 100%
            );
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            border: 1px solid #444;
            margin: 10px 0;
        }
        .metric-card {
            background: #1e1e2e;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            border: 1px solid #333;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Load Models ────────────────────────────────────
    models     = load_models()
    scaler     = load_scaler()
    mfcc_mean, mfcc_std = load_norm_stats()

    # ── Sidebar ────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎤 Speech Emotion Recognition")
        st.markdown("---")

        st.markdown("### ⚙️ Settings")

        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=list(models.keys()),
            index=0
        )

        # Recording duration
        rec_duration = st.slider(
            "Recording Duration (seconds)",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )

        st.markdown("---")
        st.markdown("### 📊 Model Performance")

        # Model metrics display
        metrics = {
            'LSTM (Best — 96.53%)' : {'acc': 96.53, 'f1': 0.962},
            'MLP (93.75%)'         : {'acc': 93.75, 'f1': 0.932},
        }

        for model_name, metric in metrics.items():
            is_selected = model_name == selected_model
            border = '2px solid #667eea' if is_selected \
                     else '1px solid #333'
            st.markdown(
                f"""
                <div style='background:#1e1e2e;
                            border-radius:8px;
                            padding:10px;
                            margin:5px 0;
                            border:{border}'>
                    <b style='color:{"#667eea" if is_selected
                                     else "white"}'>
                        {model_name}
                    </b><br>
                    <small style='color:#aaa'>
                        Accuracy: {metric['acc']}% |
                        F1: {metric['f1']}
                    </small>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 🎭 Emotions")
        for idx, info in EMOTIONS.items():
            st.markdown(
                f"{info['emoji']} **{info['name']}**"
            )

    # ── Main Content ───────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center; color:white;'>"
        "🎤 Speech Emotion Recognition</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#aaa;'>"
        "Record your voice and let AI detect your emotion</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Initialize session state ───────────────────────
    if 'audio_data'    not in st.session_state:
        st.session_state.audio_data    = None
    if 'prediction'    not in st.session_state:
        st.session_state.prediction    = None
    if 'probabilities' not in st.session_state:
        st.session_state.probabilities = None
    if 'history'       not in st.session_state:
        st.session_state.history       = []

    # ── Three column layout ────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # ── Recording buttons ──────────────────────────
        tab1, tab2 = st.tabs(
            ["🎙️ Record Audio", "📁 Upload Audio"]
        )

        with tab1:
            st.markdown(
                f"<p style='text-align:center; "
                f"color:#aaa;'>Will record for "
                f"{rec_duration} seconds</p>",
                unsafe_allow_html=True
            )

            if st.button("🎙️ Start Recording",
                          use_container_width=True):
                with st.spinner(
                    f"🔴 Recording for {rec_duration}s... "
                    f"Speak now!"
                ):
                    audio = record_audio(
                        duration=rec_duration,
                        sr=SAMPLE_RATE
                    )
                    st.session_state.audio_data = audio
                st.success("✅ Recording complete!")

        with tab2:
            uploaded = st.file_uploader(
                "Upload a WAV file",
                type=['wav'],
                help="Upload a .wav file of speech"
            )

            if uploaded is not None:
                audio_bytes = uploaded.read()
                audio_buffer = io.BytesIO(audio_bytes)
                y_upload, sr_upload = librosa.load(
                    audio_buffer,
                    sr=SAMPLE_RATE,
                    duration=3.0
                )
                st.session_state.audio_data = y_upload
                st.success("✅ Audio uploaded!")

    # ── Analysis Section ───────────────────────────────
    if st.session_state.audio_data is not None:
        y = st.session_state.audio_data

        st.markdown("---")
        st.markdown(
            "<h3 style='color:white'>📈 Audio Analysis</h3>",
            unsafe_allow_html=True
        )

        # Waveform and MFCC
        col_w, col_m = st.columns(2)

        with col_w:
            fig_wave = plot_waveform(y)
            st.pyplot(fig_wave)
            plt.close()

        # Feature extraction
        with st.spinner("Extracting features..."):
            fixed_feat, mfcc_seq, y_processed = \
                preprocess_audio(y, SAMPLE_RATE)

            fixed_norm, mfcc_norm = normalize_features(
                fixed_feat, mfcc_seq,
                scaler, mfcc_mean, mfcc_std
            )

        with col_m:
            fig_mfcc = plot_mfcc(mfcc_norm)
            st.pyplot(fig_mfcc)
            plt.close()

        # ── Prediction ─────────────────────────────────
        with st.spinner("Analyzing emotion..."):
            model_info = models[selected_model]
            pred_class, probs = predict_emotion(
                model_info, fixed_norm, mfcc_norm
            )

            st.session_state.prediction    = pred_class
            st.session_state.probabilities = probs

        # ── Emotion Result Display ─────────────────────
        st.markdown("---")
        emotion_info = EMOTIONS[pred_class]

        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])

        with col_r2:
            confidence = probs[pred_class] * 100
            st.markdown(
                f"""
                <div class='emotion-card'>
                    <div style='font-size:80px'>
                        {emotion_info['emoji']}
                    </div>
                    <h1 style='color:{emotion_info["color"]};
                               font-size:48px;
                               margin:10px 0'>
                        {emotion_info['name']}
                    </h1>
                    <h3 style='color:#aaa'>
                        Confidence: {confidence:.1f}%
                    </h3>
                    <div style='background:#333;
                                border-radius:10px;
                                height:10px;
                                margin:10px 0'>
                        <div style='background:
                            {emotion_info["color"]};
                            width:{confidence}%;
                            height:100%;
                            border-radius:10px'>
                        </div>
                    </div>
                    <small style='color:#666'>
                        Model: {selected_model}
                    </small>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ── Probability chart ──────────────────────────
        st.markdown("---")
        fig_probs = plot_probabilities(probs)
        st.pyplot(fig_probs)
        plt.close()

        # ── Metrics row ────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Predicted Emotion",
                f"{emotion_info['emoji']} "
                f"{emotion_info['name']}"
            )
        with m2:
            st.metric("Confidence",
                      f"{confidence:.1f}%")
        with m3:
            st.metric("Model Used",
                      selected_model.split('(')[0].strip())
        with m4:
            runner_up_idx = np.argsort(probs)[-2]
            st.metric(
                "Runner-up",
                f"{EMOTIONS[runner_up_idx]['emoji']} "
                f"{EMOTIONS[runner_up_idx]['name']} "
                f"({probs[runner_up_idx]*100:.1f}%)"
            )

        # ── Add to history ─────────────────────────────
        history_entry = {
            'emotion'   : emotion_info['name'],
            'emoji'     : emotion_info['emoji'],
            'confidence': f"{confidence:.1f}%",
            'model'     : selected_model.split('(')[0]
        }
        if (len(st.session_state.history) == 0 or
            st.session_state.history[-1] != history_entry):
            st.session_state.history.append(history_entry)

    # ── History Section ────────────────────────────────
    if len(st.session_state.history) > 0:
        st.markdown("---")
        st.markdown(
            "<h3 style='color:white'>📜 Prediction History"
            "</h3>",
            unsafe_allow_html=True
        )

        cols = st.columns(
            min(len(st.session_state.history), 5)
        )
        for i, entry in enumerate(
            st.session_state.history[-5:]
        ):
            with cols[i % 5]:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div style='font-size:30px'>
                            {entry['emoji']}
                        </div>
                        <b style='color:white'>
                            {entry['emotion']}
                        </b><br>
                        <small style='color:#aaa'>
                            {entry['confidence']}
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

    # ── Footer ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:#555;
                    padding:20px'>
            <p>Speech Emotion Recognition System</p>
            <p>Final Year Major Project —
               Computer Engineering</p>
            <p>Built with PyTorch + Librosa + Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()