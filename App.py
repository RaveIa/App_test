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
DOWNLOAD_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle depuis Google Drive..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# === Chargement du modèle ===
model = download_model()

# === Liste des classes
classes = ["led", "potentiometer", "push_button", "resistor", "ultrasonic_sensor"]

# === Vérification de cohérence
if model.output_shape[-1] != len(classes):
    st.error(f"⚠️ Le modèle retourne {model.output_shape[-1]} classes, mais la liste `classes` en a {len(classes)}.")
    st.stop()

# === Descriptions des composants
descriptions = {
    "led": "Une LED est une diode électroluminescente qui émet de la lumière.",
    "potentiometer": "Un potentiomètre permet de régler une résistance variable.",
    "push_button": "Un bouton poussoir permet de fermer temporairement un circuit.",
    "resistor": "Une résistance limite le courant électrique.",
    "ultrasonic_sensor": "Un capteur à ultrasons mesure les distances grâce au son."
}

# === Réinitialisation via bouton
if st.button("🔄 Réinitialiser la photo"):
    st.session_state["camera"] = None
    st.session_state["uploaded"] = None
    st.experimental_rerun()

# === Formulaire d'upload et caméra
with st.form("image_form"):
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Uploader une image", type=["jpg", "png"], key="uploaded")
    with col2:
        camera = st.camera_input("Ou prends une photo", key="camera")
    
    submit = st.form_submit_button("Analyser")

# === Traitement et prédiction
if submit:
    image = None
    if camera:
        image = Image.open(camera)
    elif uploaded:
        image = Image.open(uploaded)

    if image:
        st.image(image, caption="Image analysée", use_container_width=True)
        img = image.resize((150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        try:
            prediction = model.predict(img_array)
            predicted_class = classes[np.argmax(prediction)]
            confidence = float(np.max(prediction) * 100)

            st.success(f"Composant identifié : **{predicted_class}** ({confidence:.2f}%)")
            st.info(f"Description : {descriptions.get(predicted_class, 'Non disponible.')}")
        except Exception as e:
            st.error("Erreur pendant la prédiction :")
            st.code(str(e))
    else:
        st.warning("Aucune image fournie.")
