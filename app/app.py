import os
import sys

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classifier.ImageClassifier import ImageClassifier

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize the classifier
classifier = ImageClassifier()
classifier.load_model("results/best_model.keras")


# Route for uploading the file and displaying the result
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == "":
            return "No selected file", 400
        if file:
            # Save and preprocess the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            prediction = classifier.predict(file_path)

            return render_template(
                "result.html", filename=filename, prediction=prediction
            )
    return render_template("upload.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
