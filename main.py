import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os
from tempfile import NamedTemporaryFile
import pandas as pd

# Load the model and encoder
@st.cache_resource
def load_model_and_encoder():
    model = load_model('model.h5')  # Load the Keras model
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)  # Load the label encoder
    return model, encoder

model, encoder = load_model_and_encoder()

# Set up the Streamlit app UI
st.title("Parkinson Classification")
st.markdown("""
Upload a WAV audio file for Parkinson detection.

""")

# File uploader for WAV files
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

# Function to make predictions on the uploaded audio file
def predict_audio(audio_path):
    # Load and process the audio file
    y, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Pad or trim the MFCCs to ensure consistent length
    max_pad_len = 50
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    # Reshape for model input
    features = mfccs.T.reshape(1, mfccs.shape[1], mfccs.shape[0])
    
    # Get predictions from the model
    probs = model.predict(features)
    pred_class = encoder.inverse_transform([np.argmax(probs)])
    confidence = np.max(probs) * 100
    
    return pred_class[0], confidence, probs

if uploaded_file:
    # Save uploaded file temporarily so we can process it
    with NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    # Show an audio player for the uploaded file
    st.audio(uploaded_file, format='audio/wav')
    
    # Make predictions and show a loading spinner while processing
    with st.spinner('Analyzing your audio...'):
        class_name, confidence, all_probs = predict_audio(tmp_path)
    
    # Clean up temp file after we're done with it
    os.unlink(tmp_path)
    
    # Display the results to the user
    st.subheader("Results")
    
    # Show predicted class and confidence level in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", class_name)
    with col2:
        st.metric("Confidence Level", f"{confidence:.1f}%")
    
    # Show a bar chart for confidence distribution across classes
    st.markdown("### Confidence Distribution")
    prob_df = pd.DataFrame({
        'Class': encoder.classes_,
        'Confidence': all_probs[0]
    }).sort_values('Confidence', ascending=False)
    
    st.bar_chart(prob_df.set_index('Class'))
    
    # Show detailed probabilities for each class in progress bars
    st.markdown("### Detailed Probabilities")
    for class_name, prob in zip(encoder.classes_, all_probs[0]):
        st.progress(float(prob), text=f"{class_name}: {prob*100:.1f}%")
