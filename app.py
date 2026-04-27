import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Emotion Detector", page_icon="😊")

st.title("😊 Real-Time Emotion Recognition")
st.write("Show your face to the camera!")

# Load model
@st.cache_resource
def load_emotion_model():
    return load_model('emotion_model.h5')

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Camera input
camera_image = st.camera_input("Take a picture")

if camera_image:
    # Convert to OpenCV format
    image = Image.open(camera_image)
    image = np.array(image)
    
    # Convert to grayscale and resize
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face / 255.0
    face = face.reshape(1, 48, 48, 1)
    
    # Predict
    predictions = model.predict(face)
    emotion_idx = np.argmax(predictions[0])
    emotion = emotion_labels[emotion_idx]
    confidence = np.max(predictions[0]) * 100
    
    # Display result
    st.success(f"**Detected Emotion: {emotion}**")
    st.info(f"Confidence: {confidence:.1f}%")
    
    # Show confidence bars
    st.write("### Confidence Scores:")
    for i, (label, score) in enumerate(zip(emotion_labels, predictions[0])):
        st.progress(float(score), text=f"{label}: {score*100:.1f}%")