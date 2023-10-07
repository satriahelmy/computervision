import base64
from io import BytesIO
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, render_template, request
from PIL import Image
from sklearn.cluster import KMeans


matplotlib.use("agg")

bp = Blueprint("color_palette", __name__)


def get_palette(
    img: Image, n_colors: int, resize_shape: Tuple[int, int] = (100, 100)
) -> np.ndarray:
    img = np.asarray(img.resize(resize_shape)) / 255.0
    h, w, c = img.shape
    img_arr = img.reshape(h * w, c)
    kmeans = KMeans(n_clusters=n_colors, n_init="auto").fit(img_arr)
    palette_rgb = (kmeans.cluster_centers_ * 255).astype(int)
    palette_hex = [matplotlib.colors.rgb2hex(
        color) for color in palette_rgb/255]
    return palette_rgb, palette_hex


def get_palette_image(color_palette):
    color_palette_sorted = np.array(
        sorted(color_palette, key=lambda x: x.mean())[::-1])

    plt.imshow(color_palette_sorted[np.newaxis, :, :])
    plt.axis("off")
    palette_image_path = "static/palette_image.png"
    plt.savefig(palette_image_path, bbox_inches="tight",
                pad_inches=0, format="png")
    plt.close()

    with open(palette_image_path, "rb") as img_file:
        palette_image_base64 = base64.b64encode(img_file.read()).decode()

    return palette_image_base64


@bp.route("/color_palette", methods=["GET", "POST"])
def color_palette():
    if request.method == "POST":
        image = request.files["image"].read()
        uploaded_image = base64.b64encode(
            image).decode()  # Convert to base64
        img = Image.open(BytesIO(image))
        num_colors = 8
        color_palette, color_palette_hex = get_palette(img, num_colors)

        color_palette_hex.sort(reverse=True)

        palette_image_base64 = get_palette_image(color_palette)

        return render_template(
            "pages/color_palette.html",
            color_palette=color_palette,
            color_palette_hex=color_palette_hex,
            palette_image_base64=palette_image_base64,
            uploaded_image=uploaded_image
        )

    else:
        return render_template("pages/color_palette.html")
