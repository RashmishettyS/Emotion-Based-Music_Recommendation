import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load the Spotify API credentials
client_id = 'f47ce3e0ce6440969448f280c592be88'
client_secret = '2a9c299d82074289b51c4b406f2fd65d'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load the trained model
model = load_model('emotion_models.h5')

# Function to preprocess image for model prediction
def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Resize the image to match the input shape of the model (48x48)
    resized = cv2.resize(gray, (48, 48))
    # Convert image to array and normalize pixel values
    normalized = resized / 255.0
    # Expand dimensions to match the input shape expected by the model
    processed_image = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
    return processed_image

# Function to predict emotion from image
def predict_emotion(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Predict emotion using the model
    prediction = model.predict(processed_image)
    # Get the index of the predicted emotion
    predicted_class_index = np.argmax(prediction)
    # Map the index to the corresponding emotion label
    emotion_labels = ['Happy', 'Sad', 'Angry', 'Neutral']
    predicted_emotion = emotion_labels[predicted_class_index]
    return predicted_emotion

# Function to recommend songs based on emotion
def recommend_songs(emotion_list):
    # Define mapping of emotions to Spotify genre seeds
    genre_seeds = {
        'Happy': 'pop',
        'Sad': 'acoustic',
        'Angry': 'rock',
        'Neutral': 'ambient'
    }
    # Get genre seeds based on detected emotions
    seed_genres = [genre_seeds.get(emotion, 'pop') for emotion in emotion_list]
    # Get song recommendations based on seed genres
    recommendations = sp.recommendations(seed_genres=seed_genres, limit=5)
    return recommendations['tracks']

# Streamlit app
st.title('Emotion Detection and Music Recommendation')

# Option to choose webcam input
option = st.sidebar.radio("Select Input Method", ('Webcam',))

if option == 'Webcam':
    # Webcam input
    stframe = st.empty()
    camera = cv2.VideoCapture(0)

    recording = st.checkbox("Start Recording")
    if recording:
        st.write("Recording...")

    emotion_list = []

    while recording:
        _, frame = camera.read()
        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the frame
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Predict emotion
        predicted_emotion = predict_emotion(frame_rgb)
        st.write(f"Predicted Emotion: {predicted_emotion}")
        emotion_list.append(predicted_emotion)

        if not st.checkbox("Stop Recording"):
            break

    # Release the camera
    camera.release()

    st.write("Recording stopped. Processing emotions...")

    # Remove video display after recording
    stframe.empty()

    # Recommend songs based on detected emotions
    if emotion_list:
        st.write("Emotions recorded:")
        st.write(emotion_list)
        st.write("Recommended Songs:")
        recommendations = recommend_songs(emotion_list)
        for track in recommendations:
            st.write(f"- [{track['name']}]({track['external_urls']['spotify']}) by {', '.join([artist['name'] for artist in track['artists']])}")
