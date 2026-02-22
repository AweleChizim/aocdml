import os
import uuid
import numpy as np
import cv2
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("aocdml.keras")
background_data = np.load("background_data.npy")
last_conv_layer_name = "conv2d_2"

if background_data.ndim == 3:
    background_data = np.expand_dims(background_data, axis=-1)


def build_gradcam_model():
    inputs = tf.keras.Input(shape=(224, 224, 1))
    x = inputs
    conv_output = None

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x

    return tf.keras.Model(inputs=inputs, outputs=[conv_output, x])

grad_model = build_gradcam_model()


def make_gradcam_heatmap(img_array, pred_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def make_shap_explanation(img_array):
    explainer = shap.GradientExplainer(model, background_data)
    shap_values = explainer.shap_values(img_array)
    return shap_values


@app.post("/predict")
async def predict(img_file: UploadFile = File(...)):

    # Save uploaded file temporarily
    unique_id = str(uuid.uuid4())
    upload_path = f"results/{unique_id}.jpg"

    with open(upload_path, "wb") as buffer:
        buffer.write(await img_file.read())

    # Preprocess image
    img = image.load_img(upload_path, target_size=(224,224), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    img_array /= 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    percentages = (prediction[0] * 100).round(2)

    classes = [
        "Benign Ovarian Tumor",
        "Normal Ovary",
        "Ovarian Cancer",
        "Uterine Fibroid"
    ]

    category = classes[predicted_class]
    probability = float(percentages[predicted_class])

    # Grad-CAM XAI
    heatmap = make_gradcam_heatmap(img_array, predicted_class)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_display = np.uint8(255.0 * img_array[0].squeeze())
    img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
    superimposed_img = cv2.addWeighted(img_display, 0.6, heatmap, 0.4, 0)

    gradcam_path = f"results/{unique_id}_gradcam.png"
    cv2.imwrite(gradcam_path, superimposed_img)

    # SHAP XAI
    shap_values = make_shap_explanation(img_array)

    if isinstance(shap_values, list):
        shap_map = shap_values[predicted_class][0]
    else:
        shap_map = shap_values[0, ..., predicted_class]

    shap_map = shap_map.squeeze()
    max_val = np.max(np.abs(shap_map))

    plt.figure(figsize=(5,5))
    plt.imshow(img_array[0].squeeze(), cmap='gray')
    plt.imshow(shap_map, cmap='seismic', alpha=0.7,
               vmin=-max_val, vmax=max_val)
    plt.axis('off')

    shap_path = f"results/{unique_id}_shap.png"
    plt.savefig(shap_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Return JSON
    return JSONResponse({
        "prediction": category,
        "probability": probability,
        "gradcam_url": f"/download/{unique_id}_gradcam.png",
        "shap_url": f"/download/{unique_id}_shap.png"
    })


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = f"results/{filename}"
    return FileResponse(file_path, media_type="image/png")