import base64

import cv2
import joblib
import numpy as np
from flask import Blueprint, render_template, request
from skimage.feature import graycomatrix, graycoprops

from module import allowed_file

bp = Blueprint("orange_diseases", __name__)

class_names = ["Greening", "Blackspot", "Canker", "Fresh"]
knn = joblib.load('models/knn_model.pkl')

glcm_distances = [1]
glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm_properties = ['contrast', 'homogeneity', 'energy', 'correlation']
hsv_properties = ['hue', 'saturation', 'value']


def extract_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    glcm_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(glcm_image, distances=glcm_distances,
                        angles=glcm_angles, symmetric=True, normed=True)

    hsv_features = []
    for property_name in hsv_properties:
        property_value = hsv_image[:, :,
                                   hsv_properties.index(property_name)].ravel()
        hsv_features.extend([np.mean(property_value), np.std(property_value)])

    glcm_features = []
    for property_name in glcm_properties:
        property_value = graycoprops(glcm, property_name).ravel()
        glcm_features.extend([np.mean(property_value), np.std(property_value)])

    features = hsv_features + glcm_features
    return features


@bp.route("/orange_diseases_classification", methods=["GET", "POST"])
def orange_diseases():
    if request.method == "POST":
        image = request.files['image']
        if image and allowed_file(image.filename):
            image_data = image.read()

            image_base64 = base64.b64encode(image_data).decode()

            image_cv = cv2.imdecode(np.frombuffer(
                image_data, np.uint8), cv2.IMREAD_COLOR)
            image_cv = cv2.resize(image_cv, (64, 64))
            features = extract_features(image_cv)
            prediction = knn.predict([features])
            predicted_class = prediction[0].item()
            result = class_names[predicted_class]

            # Add the description, prevention, and treatment based on the predicted class
            description = ""
            prevention = ""
            treatment = ""

            if predicted_class == 0:  # Greening
                description = "Greening adalah penyakit yang disebabkan oleh bakteri dan umumnya menyerang tanaman jeruk. Gejalanya termasuk daun berwarna kuning muda, penurunan pertumbuhan, dan pucuk tanaman yang mati."
                prevention = "Beberapa tindakan pencegahan Greening meliputi memperhatikan kebersihan kebun, memotong dan membuang tanaman yang terinfeksi, serta penggunaan bibit yang bebas penyakit."
                treatment = "Sayangnya, tidak ada pengobatan yang efektif untuk Greening. Tindakan yang dapat dilakukan adalah memantau dan mengendalikan serangga vektor penyakit serta merawat tanaman dengan baik agar tetap sehat."

            elif predicted_class == 1:  # Blackspot
                description = "Blackspot adalah penyakit yang umum terjadi pada tanaman jeruk. Gejalanya berupa bintik-bintik hitam pada daun dan buah, yang kemudian dapat menyebabkan penurunan kualitas dan produksi tanaman."
                prevention = "Tindakan pencegahan Blackspot meliputi pemangkasan dan pemotongan bagian tanaman yang terinfeksi, penggunaan fungisida yang sesuai, serta memastikan kelembaban udara yang baik di sekitar tanaman."
                treatment = "Pengobatan Blackspot melibatkan penggunaan fungisida yang efektif dan penerapan tindakan pengendalian penyakit yang tepat waktu. Penting untuk memantau dan merawat tanaman secara teratur."

            elif predicted_class == 2:  # Canker
                description = "Canker adalah penyakit yang disebabkan oleh jamur atau bakteri dan dapat menginfeksi batang dan cabang tanaman jeruk. Gejalanya berupa lesi atau luka pada kulit tanaman."
                prevention = "Beberapa tindakan pencegahan Canker meliputi pemangkasan dan pembuangan cabang yang terinfeksi, penggunaan fungisida atau antibakteri yang direkomendasikan, serta memastikan sanitasi yang baik di kebun."
                treatment = "Pengobatan Canker melibatkan penggunaan fungisida atau antibakteri yang sesuai, pemangkasan tanaman untuk menghilangkan jaringan yang terinfeksi, dan pemeliharaan kebersihan yang baik."

            elif predicted_class == 3:  # Fresh
                description = "Tanaman Anda sehat dan tidak terkena penyakit."
                prevention = ""
                treatment = ""

            return render_template("pages/orange_diseases.html",
                                   result=result,
                                   image_path=image_base64,
                                   description=description,
                                   prevention=prevention,
                                   treatment=treatment)
    else:
        return render_template("pages/orange_diseases.html")
