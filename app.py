from flask import Flask, request, jsonify

from model import get_prediction_by

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    """
    Initial page
    """
    return jsonify("Dog Breed Classifier")


@app.post("/classify")
def classify_image():
    """
    Get predictions by file received from request
    """
    file = request.files["image"]
    dog, prob = get_prediction_by(file)
    return jsonify({"class": dog, "prob": int(prob * 100)})


if __name__ == "__main__":
    app.run(debug=True)
