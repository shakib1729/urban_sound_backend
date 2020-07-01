<h1 align="center">Sound Classification using Images - Urban Sound Classification</h1>
<p align="center">
<img src="https://user-images.githubusercontent.com/39847281/78693980-a05fd600-7919-11ea-96d9-e12e2a12ba3d.JPG"></p>
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/39847281/78693594-1879cc00-7919-11ea-81bc-0ce58aae8b19.JPG"></p>

## 
This repository contains the backend part of the web application implemented using Flask<br/>
Frontend using ReactJS: https://github.com/shakib1729/urban_sound_frontend<br/>
Jupyter Notebooks: https://github.com/shakib1729/urban-sound-cnn-jupyter<br/>
#### Deployed on Heroku: https://urban-sound.herokuapp.com/ <br/>

## About the project.
The goal of this project is to classify urban sounds into ten classes ('air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'). <br/>
This project classifies the sounds by first getting the visual representation of the sound and then using a CNN classifier to classify the sounds.<br/>
These visual representations of sounds are called spectrograms. <br/>
In a spectrogram representation plot — one axis represents the time, the second axis represents frequencies and the colors represent magnitude (amplitude) of the observed frequency at a particular time.<br/>

For example, the Spectrogram of a 'siren' sound is: <br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/39847281/86238925-04fa9c00-bbbc-11ea-87e0-7d2b0c26f2d3.png"></p>
The Spectrogram of a 'jackhammer' sound is: <br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/39847281/86238994-1fcd1080-bbbc-11ea-928c-ad67a64aa3bc.png"></p> <br/>
Using these spectrogram images, we classify the sounds<br/>

## Dataset:
UrbanSound8k dataset: https://urbansounddataset.weebly.com <br/>
It contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.<br/>

## Libraries/Frameworks used:
1) Librosa - To load the sound files.
2) Matplotlib - To save the spectrogram of the audio files.
3) NumPy - For array manipulation.
4) Keras - To build the Convolutional neural network and to load the image files.
5) Scikit-learn - To split the dataset into training and testing part and also for analyzing the performance of the model using sklearn.metrics.
6) Python Imaging Library (PIL) - The load_img() function of Keras loads an image into PIL format.
7) Tensorflow - Keras uses TensorFlow for its backend.
8) Flask - To deploy the CNN model.

#### &nbsp; React - To build the front-end part of the web application
<br/>


## Implementation:
1) Load the audio files using librosa and save their spectrograms using matplotlib.
2) Create 'X' and 'Y' training dataset by loading the saved spectrograms using Keras.
3) Build and train the CNN Model using training dataset. After training, save the model.
4) Build API for the CNN Model using Flask which will serve as the backend of the web application.
5) Build the frontend part of the web application using ReactJS which calls the API created using Flask.
6) Deploy the web application on Heroku.
<br/>

## Results:
1) Model containing 2 dropout layers (dropout rates as 0.5 and 0.5 respectively) and learning_rate=0.001 (default value):<br/>
    Train Accuracy: 64.6% <br/>
    Test Accuracy: 59.9% <br/>
    The plot of accuracy on the training and validation datasets over training epochs.:<br/>
    <p align="center">
    <img src="https://user-images.githubusercontent.com/39847281/86269265-23c25800-bbe7-11ea-9079-f4636a9dd959.JPG"></p> <br/>
    (This plot shows the accuracies while training where dropout layers are only activated for training dataset, hence train accuracy varies than when no dropout layer is activated) <br/><br/>
 2) Model containing 1 dropout layer (dropout rate as 0.5) and learning_rate=0.001 (default value):<br/>
    Train Accuracy: 97% <br/>
    Test Accuracy: 79.4% <br/>
    The plot of accuracy on the training and validation datasets over training epochs.:<br/>
    <p align="center">
     <img src="https://user-images.githubusercontent.com/39847281/86270193-76e8da80-bbe8-11ea-98c5-201046a0f4bb.JPG"></p> <br/> <br/>
 3) Model containing 2 dropout layers (dropout rates as 0.5 and 0.5 respectively) and learning_rate=0.0005:<br/>
    Train Accuracy: 90.19% <br/>
    Test Accuracy: 80.2% <br/>
    The plot of accuracy on the training and validation datasets over training epochs.:<br/>
    <p align="center">
     <img src="https://user-images.githubusercontent.com/39847281/86270663-39388180-bbe9-11ea-8a8e-ee4ee700f876.JPG"></p> <br/> <br/>
 4) Model containing 2 dropout layers (dropout rates as 0.3 and 0.5 respectively) and learning_rate=0.0005:<br/>
    Train Accuracy: 96.4% <br/>
    Test Accuracy: 83.3% <br/>
    The plot of accuracy on the training and validation datasets over training epochs.:<br/>
    <p align="center">
     <img src="https://user-images.githubusercontent.com/39847281/86270942-b2d06f80-bbe9-11ea-9206-cd047447a2f8.JPG"></p> <br/> <br/>
