
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from keras.preprocessing import image

import cv2
from flask import Flask, render_template, request

# export the model to a json file
exported_model = model_from_json(
    open("model/myModel.json", "r").read())
exported_model.load_weights('model/myModel.h5')

faceDetector = cv2.CascadeClassifier(
    'model/computer_vision.xml')
# initialize the Flask application
flask_web_app = Flask(__name__)


# route for index page
@flask_web_app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

# route for visualization page


@flask_web_app.route('/visualization', methods=['GET'])
def visualization():
    return render_template('visualization.html')

# route for prediction page


@flask_web_app.route('/detect', methods=['POST', 'GET'])
def detect():
    if request.method == 'POST':
        uploadedFile = request.files['file']
        uploadedFile.save('static/img/' + uploadedFile.filename)
        prediction = get_Image('static/img/'+str(uploadedFile.filename))
        return render_template('detect.html', prediction=prediction, filename=uploadedFile.filename)


# get the pixeleted image from the uploaded image
def get_Image(pathOfImage):
    global uploadedImage, img_data
    uploadedImage = cv2.imread(pathOfImage)
    uploadedImage = getPixelArrayImage(uploadedImage)
    return uploadedImage

# convert image into array of pixels


def getPixelArrayImage(img):
    detectedImgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect = faceDetector.detectMultiScale(detectedImgGray, 1.32, 5)
    for (x_coord, y_coord, wid, ht) in detect:
        cv2.rectangle(img, (x_coord, y_coord), (x_coord+wid,
                      y_coord+ht), (255, 0, 0), thickness=7)

        grayImage = detectedImgGray[y_coord:y_coord+wid, x_coord:x_coord+ht]
        grayImage = cv2.resize(grayImage, (48, 48))
        arrayPix = image.img_to_array(grayImage)
        arrayPix = np.expand_dims(arrayPix, axis=0)
        arrayPix /= 255

        pred = exported_model.predict(arrayPix).argmax(axis=1)
        label = ["Angry", "Disgust", "Scared",
                          "Happy", "Sad", "Surprised", "Neutral"]
        print("Prediction Output: ", label[pred[0]])

    return label[pred[0]]


if __name__ == "__main__":
    flask_web_app.run(debug=True)
