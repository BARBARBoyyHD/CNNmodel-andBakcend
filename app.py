import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


app = Flask(__name__)

# Load your saved model
model_path = os.path.join('./models', 'happymodel.h5')  # Make sure this path is correct
model = load_model(model_path)

def preprocess_image(image):
    """Preprocess the uploaded image to match model input requirements."""
    image = image.resize((256, 256))  # Resize to the expected input size (256x256)
    image = np.array(image) / 255.0     # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


@app.route('/api/predict', methods=['POST'])
def predict():
    # Ensure an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the image file
    file = request.files['image']
    try:
        image = Image.open(file.stream)
    except:
        return jsonify({"error": "Invalid image format"}), 400

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions using the loaded model
    predictions = model.predict(processed_image)

    # Assuming binary classification (Happy/Sad), interpret the results
    class_names = ['Sad', 'Happy']  # Adjust to your model's labels
    predicted_class = class_names[np.argmax(predictions[0])]

    return jsonify({
        "prediction": predicted_class,
        "confidence": float(np.max(predictions[0]))
    })


if __name__ == '__main__':
    app.run(debug=True)
