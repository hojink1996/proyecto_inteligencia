"""
Probar Imagenes Proyecto Final de Imagenes del Ramo EL5206-1 Laboratorio de Inteligencia Computacional.

@authors: Hojin Kang and Eduardo Salazar
"""

import pandas as pd
import pickle

from lib.VideoOperator import VideoOperator

# Maps the video type to the needed name
mapping_video_type = {'ataque': 'videos_ataque', 'original': 'videos'}
mapping_video = {'ataque': 'ataque', 'original': 'usuario'}

# Get the results for a single video (user must be between 1-6, example_number must be between 1-10)
user = 1
example_number = 9

# Must be either 'ataque' or 'original'
video_type = 'original'

# Process the data for the single video
video_test = VideoOperator(f'{mapping_video_type[video_type]}/{mapping_video[video_type]}_{user}_{example_number}.mp4')
results = list(video_test.obtain_values())
results = pd.DataFrame(results).transpose()

# Standarize the data
std_values = pd.read_csv('std_values.csv')
mean_values = pd.read_csv('mean_values.csv')
results = (results - mean_values) / std_values

# Load the model
filename = 'svm_model.sav'
svm = pickle.load(open(filename, 'rb'))

# Get the prediction for the video
predict_video = svm.predict(results)

# Calculate the mean of the video
mean_value_video = predict_video.mean()

# Play the video with the prediction
video_test.play_video_prediction(mean_value_video)

