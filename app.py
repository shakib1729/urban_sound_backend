from flask import Flask, jsonify, request
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from keras.models import load_model
from werkzeug.utils import secure_filename

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'models/UrbanSoundNNComplete.h5'
model = load_model(MODEL_PATH)
# model._make_predict_function()

# The path where to store the uploaded file and save the spectrogram
app.config["UPLOADS_PATH"] = 'C:/Users/AS/Documents/Urban Sound Classification/urban sound backend/uploads'

def extract_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

def model_predict(wav_path):
    y, sr = librosa.load(wav_path)
    x = extract_mfcc(y, sr)
    x = np.expand_dims(x, axis = 0)
    pred = model.predict(x)
    class_idx = np.argmax(pred[0])
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 
                   'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 
                   'jackhammer', 'siren', 'street_music']
    predicted_class = class_names[class_idx]
    return predicted_class


@app.route('/')
def index():
    # Main page, nothing to render here
    return None

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		# Get the file from the POST request
		currFile = request.files['file']
		print (request)
		# Save the file to the specified path
		file_name = secure_filename(currFile.filename)
		file_path = os.path.join(app.config["UPLOADS_PATH"], file_name)
		currFile.save(file_path)	

		# Make prediction
		prediction = model_predict(file_path)

		# Return the predicted result in JSON format
		return jsonify({'prediction' : prediction}),200


	return None


if __name__ == "__main__":
	app.run(debug=False, threaded=False)