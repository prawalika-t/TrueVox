import streamlit as st
import numpy as np
import librosa
import joblib
import io

# ---------------------------------------
# Inject custom CSS for a classy, funky UI with black and green colors
# ---------------------------------------
st.markdown(
    """
    <style>
    /* Overall page background and text */
    .reportview-container {
        background: #000000;
        color: #00ff00;
    }
    .sidebar .sidebar-content {
        background: #000000;
        color: #00ff00;
    }
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #00ff00;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Description paragraph styling */
    .description {
        font-size: 18px;
        font-family: 'Courier New', Courier, monospace;
        color: #00ff00;
        background: rgba(0, 0, 0, 0.5);
        padding: 15px;
        border-radius: 8px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #00ff00;
        color: #000000;
        border: none;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    /* File uploader styling */
    .stFileUploader label {
        color: #00ff00;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------
# Function to load the pre-trained model.
# Using st.cache_resource to cache the model.
# ---------------------------------------
@st.cache_resource
def load_model():
    model_path = "lie-detection-model.pkl"  # Ensure this file is in the same directory.
    model_data = joblib.load(model_path)
    return model_data

# ---------------------------------------
# Function to extract features from an uploaded audio file.
# This function replicates the feature extraction used during training.
# ---------------------------------------
def extract_features(audio_file):
    # Reset the file pointer to the beginning.
    audio_file.seek(0)
    audio_bytes = audio_file.read()
    
    # Load audio from the in-memory bytes.
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None

    # --- Compute Power ---
    power = np.sum(y**2) / len(y)
    
    # --- Compute Pitch Mean and Standard Deviation ---
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    threshold = np.max(magnitudes) / 2
    pitch_values = pitches[magnitudes > threshold]
    if pitch_values.size > 0:
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
    else:
        pitch_mean = 0.0
        pitch_std = 0.0

    # --- Compute Voiced Fraction ---
    try:
        voiced_intervals = librosa.effects.split(y, top_db=20)
    except Exception as e:
        st.error(f"Error detecting voiced segments: {e}")
        return None

    voiced_total = sum([end - start for start, end in voiced_intervals])
    voiced_fraction = voiced_total / len(y)

    return np.array([power, pitch_mean, pitch_std, voiced_fraction])

# ---------------------------------------
# Main Streamlit App
# ---------------------------------------
def main():
    # Header & description
    st.title("TrueVox: Audio Lie Detection")
    st.markdown(
        """
        <div class="description">
        **TrueVox** uses state-of-the-art machine learning to analyze your audio recordings and determine if the voice is authentic or deceptive. 
        Upload an audio file below, and let TrueVox reveal the truth hidden within the sound.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Audio file uploader widget.
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        with st.spinner("Extracting audio features..."):
            features = extract_features(uploaded_file)
            
        if features is None:
            st.error("Failed to extract features from the audio file.")
            return
        
        st.write("Extracted features:", features)
        
        # Load the pre-trained model.
        model_data = load_model()
        model = model_data["model"]
        label_encoder = model_data["label_encoder"]
        
        # Reshape features to match model input (1 sample with 4 features).
        features_reshaped = features.reshape(1, -1)
        prediction_numeric = model.predict(features_reshaped)
        prediction_label = label_encoder.inverse_transform(prediction_numeric)[0]
        
        st.write("Prediction:", prediction_label)
        
        if prediction_label.lower() in ["true", "truth", "authentic"]:
            st.success("Authentic (Truth) detected!")
        else:
            st.error("Deceptive (Lie) detected!")

if __name__ == "__main__":
    main()
