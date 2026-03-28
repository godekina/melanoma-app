import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Teachable Machine App", page_icon="🧠", layout="centered")

st.title("🧠 Teachable Machine Classifier")
st.markdown("Upload an image and get a prediction from your trained model.")

@st.cache_resource
def load_teachable_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_teachable_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)

    raw_class = class_names[index].strip()
    class_name = raw_class.split(" ", 1)[1] if " " in raw_class else raw_class
    confidence_score = prediction[0][index]

    st.subheader("Prediction")
    st.success(f"{class_name}")

    st.subheader("Confidence")
    st.info(f"{confidence_score * 100:.2f}%")

    st.subheader("All Class Probabilities")
    for i, score in enumerate(prediction[0]):
        raw_label = class_names[i].strip()
        label = raw_label.split(" ", 1)[1] if " " in raw_label else raw_label
        st.write(f"{label}: {score * 100:.2f}%")