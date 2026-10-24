"""Flask web app that predicts hand-sign digits from uploaded images."""
from flask import Flask, render_template, request

from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Render the upload page."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Preprocess the uploaded image and render the predicted digit."""
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return render_template("result.html", err="No file uploaded.")
    try:
        img = preprocess_img(uploaded.stream)
        pred = predict_result(img)
    except (OSError, ValueError):
        return render_template("result.html", err="File cannot be processed.")
    return render_template("result.html", predictions=str(pred))


if __name__ == "__main__":
    app.run(port=9000, debug=True)
