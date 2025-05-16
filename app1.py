import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model('emotion_models.h5')

# Function to preprocess image for model prediction
def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

# Streamlit app
st.title('Emotion Detection')

# Option to choose between image upload and webcam
option = st.sidebar.radio("Select Input Method", ('Image Upload', 'Webcam'))

if option == 'Image Upload':
    # Upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Predict emotion
        predicted_emotion = predict_emotion(np.array(image))
        st.write(f"Predicted Emotion: {predicted_emotion}")

elif option == 'Webcam':
    # Webcam input
    stframe = st.empty()
    camera = cv2.VideoCapture(0)

    capture_button = st.button('Capture Webcam Image', key='capture_button')

    if capture_button:
        _, frame = camera.read()
        predicted_emotion = predict_emotion(frame)
        st.write(f"Predicted Emotion: {predicted_emotion}")
        stframe.image(frame, channels="BGR", use_column_width=True)

    # Release the camera
    camera.release()
