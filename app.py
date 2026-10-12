
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = (
    Path(__file__).parent
    / "artifacts"
    / "placement_model.joblib"
)
model = None


def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Placement Prediction API"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    required = [
        "cgpa",
        "aptitude_score",
        "communication_score",
        "internships"
    ]

    if any(key not in data for key in required):
        return jsonify({"error": "All four student fields are required."}), 400

    try:
        cgpa = float(data["cgpa"])
        aptitude = float(data["aptitude_score"])
        communication = float(data["communication_score"])
        internships = int(data["internships"])

        if not 0 <= cgpa <= 10:
            raise ValueError("CGPA must be between 0 and 10.")
        if not 0 <= aptitude <= 100:
            raise ValueError("Aptitude score must be between 0 and 100.")
        if not 0 <= communication <= 100:
            raise ValueError("Communication score must be between 0 and 100.")
        if not 0 <= internships <= 3:
            raise ValueError("Internships must be between 0 and 3.")

    except (ValueError, TypeError):
        return jsonify({"error": "Invalid student input values."}), 400

    features = [[cgpa, aptitude, communication, internships]]
    result = int(load_model().predict(features)[0])

    return jsonify({
        "placement_prediction": "PLACED" if result == 1 else "NOT PLACED"
    }), 200


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
