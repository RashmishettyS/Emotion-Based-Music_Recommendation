# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from PIL import Image
from collections import Counter
from tensorflow.keras.models import load_model

# Load the provided emotion model
MODEL_PATH = 'emotion_models.h5'
model = load_model(MODEL_PATH)

# Ensure model compatibility
input_shape = model.input_shape[1:3]  # Extract expected input dimensions
print(f"Model loaded with expected input shape: {input_shape}")

# Define emotion mapping
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load and preprocess dataset
df = pd.read_csv("muse_v3.csv")
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index(drop=True, inplace=True)

# Split dataset by emotion
df_splits = {
    "Sad": df[:18000],
    "Fearful": df[18000:36000],
    "Angry": df[36000:54000],
    "Neutral": df[54000:72000],
    "Happy": df[72000:]
}

# Recommendation function
def recommend_songs(emotions):
    data = pd.DataFrame()
    for emotion in emotions:
        if emotion in df_splits:
            data = pd.concat([data, df_splits[emotion].sample(n=5)], ignore_index=True)
    return data

# Streamlit UI
st.title("Emotion-Based Music Recommendation")
st.write("Upload an image or use your webcam to detect emotions and get music recommendations.")

# Image Upload and Emotion Detection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_np = np.array(image)

    # Preprocess image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    emotions_detected = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray, input_shape)
        cropped_img = cropped_img.reshape(1, input_shape[0], input_shape[1], 1) / 255.0

        predictions = model.predict(cropped_img)
        emotion_index = np.argmax(predictions)
        emotion_label = emotion_dict.get(emotion_index, "Unknown")
        emotions_detected.append(emotion_label)

    # Display detected emotions
    if emotions_detected:
        st.success(f"Detected emotions: {', '.join(emotions_detected)}")
        recommended_songs = recommend_songs(emotions_detected)
        st.write("### Recommended Songs:")
        for _, row in recommended_songs.iterrows():
            st.markdown(f"- [{row['name']}]({row['link']}) by {row['artist']}")
    else:
        st.warning("No emotions detected. Try uploading a clearer image.")
