"""Flask API and web UI for YOLOv8 pneumonia classification."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "runs" / "classify" / "pneumonia" / "weights" / "best.pt"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
_model: YOLO | None = None


def get_model() -> YOLO:
    """Load the trained model once, on the first prediction request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_yolov8.py first."
            )
        _model = YOLO(str(MODEL_PATH))
    return _model


def is_allowed_image(filename: str) -> bool:
    """Return whether a filename has a supported image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image: Image.Image) -> dict[str, Any]:
    """Run classification and return JSON-serializable prediction data."""
    result = get_model().predict(source=image, verbose=False)[0]
    probabilities = result.probs
    scores = {
        str(result.names[index]): round(float(probability), 6)
        for index, probability in enumerate(probabilities.data.tolist())
    }
    top_index = int(probabilities.top1)
    return {
        "prediction": str(result.names[top_index]),
        "confidence": round(float(probabilities.top1conf), 6),
        "probabilities": scores,
    }


@app.get("/")
def index() -> str:
    return render_template("index.html", model_available=MODEL_PATH.exists())


@app.get("/health")
def health() -> tuple[Any, int]:
    return jsonify({"status": "ok", "model_available": MODEL_PATH.exists()}), 200


@app.post("/api/predict")
def predict() -> tuple[Any, int]:
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload an image in the 'image' field."}), 400
    if not is_allowed_image(uploaded_file.filename):
        return jsonify({"error": "Supported formats: JPG, JPEG, PNG, BMP, and WEBP."}), 400

    try:
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
        return jsonify(predict_image(image)), 200
    except UnidentifiedImageError:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the server logs."}), 500


@app.errorhandler(413)
def file_too_large(_error: Any) -> tuple[Any, int]:
    return jsonify({"error": "Image is too large. Maximum size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
