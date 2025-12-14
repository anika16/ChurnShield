# src/api/app.py
from flask import Flask, request, jsonify
from loguru import logger
from src.models.predict import predict_single
import os

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json
    if not payload:
        return jsonify({"error": "Please provide JSON payload"}), 400
    try:
        result = predict_single(payload)
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
