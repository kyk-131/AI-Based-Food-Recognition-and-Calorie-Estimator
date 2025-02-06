!pip install kagglehub
!pip install google-colab

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import kagglehub
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from PIL import Image
from google.colab import files
from io import BytesIO

# Download the Food-101 dataset
dataset_path = kagglehub.dataset_download("dansbecker/food-101")
print("Dataset downloaded at:", dataset_path)

# Upload nutrition CSV file using files.upload()
uploaded = files.upload()
csv_path = list(uploaded.keys())[0]  # Get the uploaded file name

# Load the selected CSV file
nutrition_data = pd.read_csv(BytesIO(uploaded[csv_path]))  # Read from BytesIO
print("CSV file loaded successfully:", csv_path)
print(nutrition_data.head())  # Display first few rows

# Define Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Clean dataset function (optimized for Colab)
def clean_dataset(dataset_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(valid_extensions):
                os.remove(file_path)
            else:
                try:
                    # Using PIL's verify() for faster image validation
                    Image.open(file_path).verify() 
                except (IOError, SyntaxError):
                    os.remove(file_path)

clean_dataset(dataset_path)

# Load dataset using tf.data API (optimized for performance)
dataset_dir = os.path.join(dataset_path, "food-101")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # You can adjust this based on your Colab RAM

def process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img

list_ds = tf.data.Dataset.list_files(str(dataset_dir + '/*/*'))
train_ds = list_ds.map(process_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class_names = [item.name for item in os.scandir(dataset_dir) if item.is_dir()]
print("Classes detected:", class_names)

# Define Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
model.fit(train_ds, epochs=10, callbacks=[lr_scheduler])
model.save("food_model.h5")
print("Model saved.")

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

# Route for HTML upload page (modified for image upload)
@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # Handle image upload
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Redirect to prediction route with uploaded file path
        return redirect(url_for('predict', filepath=filepath))

    # Display upload form
    # Assuming you have an 'upload.html' file in your templates folder
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """

# Route for prediction
@app.route('/predict')
def predict():
    filepath = request.args.get('filepath')

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    img = preprocess_image(filepath)
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    nutrition_info = nutrition_data[nutrition_data['food_name'] == predicted_class]
    if not nutrition_info.empty:
        nutrition = nutrition_info.iloc[0].to_dict()
    else:
        nutrition = {"calories": "Unknown", "protein": "Unknown", "carbohydrates": "Unknown", "fat": "Unknown"}

    return jsonify({"food": predicted_class, "nutrition": nutrition, "confidence": float(np.max(predictions))})


# Run the API (modified for Colab)
def run_app():
    from google.colab.output import eval_js
    print(eval_js("google.colab.kernel.proxyPort(5000)"))

if __name__ == '__main__':
    run_app()
