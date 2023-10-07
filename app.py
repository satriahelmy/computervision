import os

from flask import Flask

from index import bp as index_bp
from orange_diseases import bp as orange_bp
from custom_object_detection import bp as custom_bp
from binary_classification import bp as binary_bp
from multiclass_classification import bp as multiclass_bp
from color_palette import bp as color_bp
from ocr import bp as ocr_bp
from face_verification import bp as face_bp

app = Flask(__name__)


app.register_blueprint(index_bp)
app.register_blueprint(orange_bp)
app.register_blueprint(custom_bp)
app.register_blueprint(binary_bp)
app.register_blueprint(multiclass_bp)
app.register_blueprint(color_bp)
app.register_blueprint(ocr_bp)
app.register_blueprint(face_bp)

if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
