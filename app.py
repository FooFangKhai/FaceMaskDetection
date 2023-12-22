import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

mdl = tf.keras.models.load_model("mdl.h5")


classes = ["Not Wearing Mask", "Wearing Mask"]

def predict_mask(image):
   img = Image.open(image)
   img = img.resize((128, 128))  
   img = img.convert("RGB") 
   img = np.array(img)
   img = img / 255.0
   img = np.resize(img, (1, 128, 128, 3))

   prediction = mdl.predict(img)

   predicted_class = classes[np.argmax(prediction)]

   return predicted_class

st.title("Mask Detection App")
st.markdown("Upload an image for mask detection and click 'Predict'.")

st.sidebar.markdown("### Upload Image")
image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if image is not None:
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if st.button("Predict"):
        predicted_class = predict_mask(image)
        st.markdown(f"**Prediction**: {predicted_class}")