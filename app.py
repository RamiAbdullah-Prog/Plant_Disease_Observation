from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Setting up the environment
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Creating the app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model('plant_disease_cnn.h5')
print("Model loaded successfully")
print(model.summary())

# Class labels for the plant diseases
class_labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
    5: 'Corn_(maize)___Common_rust_',
    6: 'Corn_(maize)___Northern_Leaf_Blight',
    7: 'Corn_(maize)___healthy',
    8: 'Grape___Black_rot',
    9: 'Grape___Esca_(Black_Measles)',
    10: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    11: 'Grape___healthy',
    12: 'Potato___Early_blight',
    13: 'Potato___Late_blight',
    14: 'Tomato___Bacterial_spot',
    15: 'Tomato___Late_blight',
    16: 'Tomato___Septoria_leaf_spot',
    17: 'Tomato___Target_Spot',
    18: 'Tomato___Tomato_mosaic_virus'
}

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))  # Resize the image
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add a new dimension to match the model's input
    return image

# Main page
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if the file is part of the request
        if 'file' not in request.files:
            return "No file part", 400
        
        file = request.files['file']
        
        # Ensure that a file is uploaded
        if file and file.filename != '':
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process the image
            processed_image = preprocess_image(filepath)

            # Predict using the model
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            label = class_labels[predicted_class]
            confidence = round(float(np.max(prediction)) * 100, 2)

            print(f"Label: {label}, Confidence: {confidence}")  # Print the result

            # Return the result to the user
            return render_template('index.html', label=label, confidence=confidence, file=file)

        return "No file selected", 400

    # Display the main page in case of a GET request
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
