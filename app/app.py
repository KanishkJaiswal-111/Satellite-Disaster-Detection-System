import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

CLASS_NAMES = ['damage', 'flood', 'normal', 'wildfire']

# --------------------------
# Prediction Function
# --------------------------
def predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)

    return CLASS_NAMES[class_idx], preds[0]

# --------------------------
# Grad-CAM
# --------------------------
def gradcam(img, model, layer_name="top_conv"):
    img_array = np.array(img.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# --------------------------
# UI
# --------------------------
st.title("🌍 Satellite Disaster Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, probs = predict(image)

    st.subheader(f"Prediction: {label}")

    st.write("Confidence Scores:")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {probs[i]:.2f}")

    heatmap = gradcam(image, model)

    img = np.array(image.resize((224,224)))
    superimposed = heatmap * 0.4 + img

    st.image(superimposed.astype("uint8"), caption="Grad-CAM Heatmap")