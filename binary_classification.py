import os

import numpy as np
from flask import Blueprint, render_template, request
from keras.models import load_model
from keras.preprocessing import image

from module import allowed_file

bp = Blueprint("binary_classification", __name__)

# Load the pre-trained model
model_path = "models/cat_vs_dog.h5"
model = load_model(model_path)


@bp.route("/binary_classification", methods=["GET", "POST"])
def binary_classification():
    result = None  # Initialize result as None

    if request.method == "POST":
        # Handle the image upload and prediction here
        image_data = request.files['image']
        if image_data and allowed_file(image_data.filename):
            # Save the uploaded image to the "static/uploads/" directory
            upload_folder = "static/uploads/"
            os.makedirs(upload_folder, exist_ok=True)
            image_path = os.path.join(upload_folder, image_data.filename)
            image_data.save(image_path)

            # Load and preprocess the uploaded image for prediction
            img = image.load_img(image_path, target_size=(160, 160))
            img = image.img_to_array(img)

            # Make a prediction
            result = make_prediction(img)

        return render_template("pages/binary_classification.html",
                               result=result,
                               uploaded_image_path=image_path)
    else:
        return render_template("pages/binary_classification.html")

# Function to make a prediction using the loaded model


def make_prediction(image):
    # Ensure the image has the correct shape and type for the model
    image = np.expand_dims(image, axis=0)

    image = np.vstack([image])

    # Make the prediction
    prediction = model.predict(image, batch_size=10)

    print(prediction)

    # Assuming a binary classification model (cat vs. dog)
    if prediction[0] > 0.5:
        return "It's a dog!"
    else:
        return "It's a cat!"
