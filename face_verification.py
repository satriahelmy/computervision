import os

import cv2
import numpy as np
from deepface import DeepFace
from flask import Blueprint, jsonify, request

bp = Blueprint("face_verification", __name__)


@bp.route("/face_verification", methods=["GET", "POST"])
def face_verification():
    if request.method == "POST":
        profile_picture = request.files["profile_picture"]
        verification_picture = request.files["verification_picture"]

        profile_picture_path = os.path.join(
            'static/uploads/', profile_picture.filename)
        profile_picture.save(profile_picture_path)

        image_bytes = verification_picture.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        verification_picture_path = os.path.join(
            'static/uploads/', verification_picture.filename)
        cv2.imwrite(verification_picture_path, img)

        try:
            result = DeepFace.verify(img1_path=profile_picture_path,
                                     img2_path=verification_picture_path,
                                     model_name="VGG-Face",
                                     detector_backend="retinaface",
                                     distance_metric="euclidean_l2"
                                     )
            result['verified'] = bool(result['verified'])
            os.remove(profile_picture_path)
            os.remove(verification_picture_path)
            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success verifying the faces"
                },
                "data": result
            }), 200

        except Exception as e:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "An error occurred while processing the images",
                    "error": str(e)
                },
                "data": None,
            }), 400

    else:
        return jsonify({
            "status": {
                "code": 200,
                "message": "Success fetching the API"
            }
        }), 200
