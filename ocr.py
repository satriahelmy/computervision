import os

import easyocr
from flask import Blueprint, render_template, request
from PIL import Image

bp = Blueprint("ocr", __name__)

reader = easyocr.Reader(['en'])


@bp.route("/ocr", methods=["GET", "POST"])
def ocr():
    if request.method == "POST":
        image = request.files["image"]

        upload_folder = "static/uploads/"
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, image.filename)
        image.save(image_path)

        text = perform_ocr(image_path)

        return render_template("pages/ocr.html",
                               result=text,
                               uploaded_image_path=image_path)
    else:
        return render_template("pages/ocr.html")


def perform_ocr(image_data):
    # Perform OCR on the image data using EasyOCR
    # image = cv2.imread(ima)
    all_text = []
    result = reader.readtext(image_data)
    for detection in result:
        all_text.append(detection[1])
    splitted_text = [text.split(" ") for text in all_text]
    formatted_sentences = [' '.join(words) + '<br>' for words in splitted_text]
    formatted_text = ''.join(formatted_sentences)
    return formatted_text
