import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "../app/model.h5"
model = load_model(MODEL_PATH)

CLASS_NAMES = ['damage', 'flood', 'normal', 'wildfire']

# =========================
# GRADCAM FUNCTION
# =========================
def get_gradcam(model, img_array, layer_name="top_conv"):
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
    return heatmap.numpy()

# =========================
# VISUALIZATION
# =========================
def show_gradcam(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))

    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    print("Prediction:", CLASS_NAMES[class_idx])

    heatmap = get_gradcam(model, img_array)

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + img

    plt.imshow(cv2.cvtColor(superimposed.astype("uint8"), cv2.COLOR_BGR2RGB))
    plt.title(CLASS_NAMES[class_idx])
    plt.axis("off")
    plt.show()

# =========================
# TEST
# =========================
if __name__ == "__main__":
    show_gradcam("../test.jpg")