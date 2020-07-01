from flask import Flask, jsonify, request
import os
import numpy as np
import librosa
from PIL import Image
from keras.models import load_model
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import matplotlib.pyplot as plt

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'models/UrbanSoundCNN4.h5'
model = load_model(MODEL_PATH)

def save_spectrogram(curr_audio_path, curr_audio_name):
    X, sr = librosa.load(curr_audio_path)  # librosa.load() returns an np array and sampling rate(by default 22050)
    plt.specgram(X, Fs=22050)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.plot
    plt.savefig(curr_audio_name,  bbox_inches= 'tight' , pad_inches = 0, dpi = 25)

class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

def model_predict(wav_path):
	wav_name = os.path.splitext(wav_path)[0]
	save_spectrogram(wav_path, wav_name)
	png_path = wav_name + '.png'
	png_img = image.load_img(png_path, target_size = (64, 64))
	x = image.img_to_array(png_img)
	x = np.expand_dims(x, axis = 0)
	pred = model.predict(x)
	class_idx = np.argmax(pred[0])
	predicted_class = class_names[class_idx]
	return predicted_class


@app.route('/')
def index():
    # Main page, nothing to render here
    return "Hello From Server, Hii"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		# Get the file from the POST request
		currFile = request.files['file']
		# Save the file to the specified path
		file_name = secure_filename(currFile.filename)
		file_path = os.path.join('./uploads', file_name)
		currFile.save(file_path)	

		# Make prediction
		prediction = model_predict(file_path)

		# Return the predicted result in JSON format
		return jsonify({'prediction' : prediction}),200


	return None


if __name__ == "__main__":
	app.run(debug=False, threaded=False)