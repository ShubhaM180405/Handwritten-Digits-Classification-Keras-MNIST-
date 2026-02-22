import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("mnist_model.h5")

def predict_digit(img):
    img = img.convert("L")
    img = img.resize((28, 28))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    return str(np.argmax(prediction))

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Handwritten Digit Classifier",
    description="Upload an image of a handwritten digit (0-9)"
)

interface.launch()
