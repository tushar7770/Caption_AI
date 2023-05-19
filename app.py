from flask import Flask, render_template, request, jsonify, flash
from flask.wrappers import Response
from PIL import Image
import caption_gen
import gpt
import cv2
import numpy as np
from io import BytesIO
import base64

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg', 'webp'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/caption', methods=['GET', 'POST'])
def caption():
    if request.method == 'POST':

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return "Error file not selected"

        if file and allowed_file(file.filename):

            image_data = BytesIO(file.read())

            # Convert the image data into a numpy array
            arr = np.frombuffer(image_data.getvalue(), np.uint8)

            # Decode the numpy array into an OpenCV image
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            captions = caption_gen.image_description(image)
            _, processed_img_data = cv2.imencode('.png', img)
            img = base64.b64encode(processed_img_data).decode('utf-8')

            return render_template("index.html", captions=captions, image=img)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
