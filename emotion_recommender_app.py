import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load the emotion detection model
model = load_model("emotion_models.h5")

# Initialize Spotify API
client_id = 'c7b47d01737b4bf6bc5d43e4b715807f'
client_secret = 'fdfc1372f318456281d679a0767087b7'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to preprocess the uploaded image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)

# Function to predict emotion from an image
def predict_emotion(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    emotion_labels = ['Happy', 'Sad', 'Angry', 'Neutral']
    return emotion_labels[np.argmax(prediction)]

# Function to recommend songs based on emotion
def recommend_songs(emotion):
    genre_map = {
        'Happy': 'pop',
        'Sad': 'acoustic',
        'Angry': 'rock',
        'Neutral': 'ambient'
    }
    seed_genre = genre_map.get(emotion, 'pop')
    recommendations = sp.recommendations(seed_genres=[seed_genre], limit=5)
    return recommendations['tracks']

# Streamlit App UI
st.title("Emotion-Based Music Recommendation")
st.write("Upload an image to detect emotion and get music recommendations!")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    image_np = np.array(image)
    emotion = predict_emotion(image_np)
    st.write(f"Detected Emotion: **{emotion}**")

    # Get song recommendations
    tracks = recommend_songs(emotion)
    st.write("### Recommended Songs:")
    for track in tracks:
        st.write(f"- {track['name']} by {', '.join(artist['name'] for artist in track['artists'])}")
