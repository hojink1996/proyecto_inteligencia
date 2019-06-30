"""
Calcular Curva ROC Proyecto Final de Imagenes del Ramo EL5206-1 Laboratorio de Inteligencia Computacional.

@authors: Hojin Kang and Eduardo Salazar
"""

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lib.VideoOperator import VideoOperator

# Maps the video type to the needed name
mapping_video_type = {'ataque': 'videos_ataque', 'original': 'videos'}
mapping_video = {'ataque': 'ataque', 'original': 'usuario'}

# Standarize the data
std_values = pd.read_csv('std_values.csv')
mean_values = pd.read_csv('mean_values.csv')
std_values = std_values['0']
mean_values = mean_values['0']

# Load the model
filename = 'svm_model.sav'
svm = pickle.load(open(filename, 'rb'))

predict_ataque = []

# Must be either 'ataque' or 'original'
video_type = 'ataque'
for user in range(1, 7):
    for example_number in range(8, 11):
        # Process the data for the single video
        video_test = VideoOperator(f'{mapping_video_type[video_type]}/{mapping_video[video_type]}_{user}_{example_number}.mp4')
        results = list(video_test.obtain_values())
        results = pd.DataFrame(results).transpose()

        results = (results - mean_values) / std_values

        # Get the prediction for the video
        predict_video = svm.predict(results)

        # Calculate the mean of the video
        mean_value_video = predict_video.mean()
        predict_ataque.append(mean_value_video)

predict_original = []
# Must be either 'ataque' or 'original'
video_type = 'original'
for user in range(1, 7):
    for example_number in range(8, 11):
        # Process the data for the single video
        video_test = VideoOperator(f'{mapping_video_type[video_type]}/{mapping_video[video_type]}_{user}_{example_number}.mp4')
        results = list(video_test.obtain_values())
        results = pd.DataFrame(results).transpose()

        results = (results - mean_values) / std_values

        # Get the prediction for the video
        predict_video = svm.predict(results)

        # Calculate the mean of the video
        mean_value_video = predict_video.mean()
        predict_original.append(mean_value_video)

threshold_vals = []
TPR = []
FPR = []
for threshold in np.arange(0.0, 1.0, 0.01):
    threshold_vals.append(threshold)
    TP = sum(i < threshold for i in predict_ataque)
    FP = sum(i < threshold for i in predict_original)

    TPR.append(TP / len(predict_ataque))
    FPR.append(FP / len(predict_original))


# Plot the ROC Curve
plt.plot(FPR, TPR, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for the Attack Detection')
plt.legend()
plt.grid()
plt.show()
plt.savefig('roc_curve.png', dpi=300)
