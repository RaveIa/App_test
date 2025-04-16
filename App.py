# app.py - Streamlit App de reconnaissance de composants électroniques

import os
import gdown
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# === Configuration initiale ===
st.set_page_config(page_title="Reconnaissance de composants", layout="centered")
st.title("🔍 Reconnaissance de composants électroniques")
st.write("Prends une photo ou upload une image pour identifier un composant.")

# === Téléchargement du modèle si absent ===
MODEL_PATH = "model_composants.h5"
GDRIVE_FILE_ID = "1sHbmE0P2rAHRI9LukMn1DzEf16UtphZo"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DGRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Téléchargement du modèle depuis Google Drive..."):
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# === Chargement du modèle ===
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()
classes = ["diode", "resistance", "condensateur"]  # à adapter si besoin

# === Descriptions des composants ===
descriptions = {
    "diode": "Une diode laisse passer le courant dans un seul sens.",
    "resistance": "Une résistance limite le courant électrique.",
    "condensateur": "Un condensateur stocke temporairement de l'énergie."
}

# === Interface Streamlit ===
image_file = st.file_uploader("Uploader une image", type=["jpg", "png"])
camera_file = st.camera_input("Ou prends une photo avec ta caméra")

# === Traitement et prédiction ===
image = None
if camera_file:
    image = Image.open(camera_file)
elif image_file:
    image = Image.open(image_file)

if image:
    st.image(image, caption="Image analysée", use_column_width=True)
    img = image.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    st.success(f"Composant identifié : **{predicted_class}** ({confidence:.2f}%)")
    st.info(f"Description : {descriptions.get(predicted_class, 'Non disponible.')}")
