```markdown
# Plant Disease Detection Using Flask and TensorFlow

This project is a web application developed using Flask and TensorFlow to detect plant diseases from images uploaded by the user. The model is based on a Convolutional Neural Network (CNN) that classifies images into different plant disease categories.

## Features
- Upload an image of a plant leaf to the web application.
- The application predicts the plant disease and provides a confidence score.
- The model supports various plant diseases including Apple, Corn, Grape, Potato, and Tomato diseases.

## Technologies Used
- **Flask**: Web framework for creating the server and handling HTTP requests.
- **TensorFlow**: Used to load and run the trained CNN model.
- **OpenCV**: Used for image processing.
- **NumPy**: Used for handling numerical operations and image data manipulation.

## Requirements

To run this project locally, ensure you have the following dependencies:

- Python 3.x
- TensorFlow
- Flask
- OpenCV
- NumPy

You can install the required libraries using the following:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following content:

```
Flask==2.1.2
tensorflow==2.8.0
opencv-python==4.5.5.64
numpy==1.21.4
```

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. Install the required dependencies (as shown above).

3. Ensure you have the trained model file (`plant_disease_cnn.h5`) in the same directory as the Flask app.

4. Run the Flask app:

   ```bash
   python app.py
   ```

5. Open a browser and navigate to `http://127.0.0.1:5000/` to access the application.

## How it Works

- The user uploads an image through the web interface.
- The image is processed (resized and normalized) and passed to the pre-trained CNN model for prediction.
- The model predicts the class of the plant disease and returns the label along with the confidence score.
- The result is displayed to the user.

## Class Labels

The model can predict the following plant diseases:

- Apple___Apple_scab
- Apple___Black_rot
- Apple___Cedar_apple_rust
- Apple___healthy
- Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot
- Corn_(maize)___Common_rust_
- Corn_(maize)___Northern_Leaf_Blight
- Corn_(maize)___healthy
- Grape___Black_rot
- Grape___Esca_(Black_Measles)
- Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
- Grape___healthy
- Potato___Early_blight
- Potato___Late_blight
- Tomato___Bacterial_spot
- Tomato___Late_blight
- Tomato___Septoria_leaf_spot
- Tomato___Target_Spot
- Tomato___Tomato_mosaic_virus

## File Structure

```
/project-directory
    /static
        /uploads
    app.py
    plant_disease_cnn.h5
    templates/
        index.html
    requirements.txt
    README.md
```

## Contributing

Feel free to fork this repository and submit pull requests. Contributions are welcome!
