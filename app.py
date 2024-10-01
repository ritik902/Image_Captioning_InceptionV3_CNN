import streamlit as st
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import zipfile
import os

# Function to unzip the model
def unzip_model(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Unzip the model file (ensure the zip file is in the same folder as the app)
zip_file_path = "caption_model.zip"  # Path to your zipped model in the same folder as app.py
unzip_model(zip_file_path, extract_to=".")

# Load the trained captioning model
model = load_model('caption_model.keras')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the InceptionV3 model for image feature extraction (without the top layers)
inception_model = tf.keras.applications.InceptionV3(weights='imagenet')
model_inception = tf.keras.models.Model(inputs=inception_model.input, outputs=inception_model.layers[-2].output)

# Preprocess the image for InceptionV3 model
def preprocess_image(image):
    img = image.resize((299, 299))  # InceptionV3 expects 299x299 images
    img = np.array(img)
    img = img / 127.5 - 1.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Extract features from the image using InceptionV3
def extract_features(image):
    img = preprocess_image(image)
    features = model_inception.predict(img)
    return features

# Generate caption based on the image features
def generate_caption(model, image_feature, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq ', '')

# Streamlit App
st.title("Image Captioning App")
st.write("Upload an image to generate a caption:")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Extract image features
    features = extract_features(image)

    # Generate caption
    caption = generate_caption(model, features, tokenizer, max_length=20)

    # Display the caption
    st.write(f"Caption: {caption}")
