import os
from pathlib import Path

import torch
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

import Model1
from torch_utils import transform_image, get_prediction

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "/static/Images"


@app.route("/")
def home():
    return render_template("view.html")


classes = ('Airplane', 'Car', 'Cat', 'Dog', 'Flower', 'Fruit', 'Motorbike', 'Person')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = {'prediction': 'None', 'filename': None}
    if request.method == 'POST':
        image = request.files['filename']
        filename = secure_filename(image.filename)
        path = Path(os.path.abspath(os.path.dirname(__file__)) + app.config["IMAGE_UPLOADS"]).as_posix()
        print(path + filename)
        image.save(path + filename)
        if image is None or image.filename == "":
            return jsonify({"error": 'No file'})
    try:
        img = open(path + filename, 'rb')
        img_bytes = img.read()
        tensor = transform_image(img_bytes)
        prediction = get_prediction(tensor)
        data['prediction'] = classes[prediction.item()]
        data['filename'] = filename
        print(data)
    except:
        return jsonify({'error': 'error during prediction'})
    return render_template('display.html', data=data)


@app.route('/display/<filename>',methods=['GET'])
def display_image(filename):
    return redirect(url_for('static',filename='Images'+filename))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
