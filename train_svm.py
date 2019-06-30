"""
Entrenamiento de SVM Proyecto Final de Imagenes del Ramo EL5206-1 Laboratorio de Inteligencia Computacional.

@authors: Hojin Kang and Eduardo Salazar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import preprocessing
from sklearn.svm import NuSVC
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from lib.VideoOperator import VideoOperator


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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
results['Type'] = 'Test'
results['Output'] = 1
results['Video'] = True
print('Done')

# Get the data
root_path = 'C:/Users/Hojin/PycharmProjects/proyecto_inteligencia/'

# Get the original data and attack data for training
original_train = pd.read_csv(f'{root_path}original_train.csv', index_col=0).reset_index(drop=True)
attack_train = pd.read_csv(f'{root_path}attack_train.csv', index_col=0).reset_index(drop=True)

# Tag the data
original_train['Type'] = 'Train'
original_train['Output'] = 1
attack_train['Type'] = 'Train'
attack_train['Output'] = 0

# Make so there are the same number of samples for both dataframes
original_train = original_train.iloc[attack_train.index].reset_index(drop=True)

# Get the original data and attack data for testing
original_test = pd.read_csv(f'{root_path}original_test.csv', index_col=0).reset_index(drop=True)
attack_test = pd.read_csv(f'{root_path}attack_test.csv', index_col=0).reset_index(drop=True)

# Tag the data
original_test['Type'] = 'Test'
original_test['Output'] = 1
attack_test['Type'] = 'Test'
attack_test['Output'] = 0

# Make so there are the same number of samples for both dataframes
original_test = original_test.iloc[attack_test.index].reset_index(drop=True)

# Get the full DataFrame to normalize
full_df = pd.concat([original_train, attack_train, original_test, attack_test]).reset_index(drop=True)
full_df['Video'] = False
results.columns = full_df.columns
full_df = pd.concat([full_df, results]).reset_index(drop=True)

# Get the type and output of each value
type_series = full_df['Type']
output_series = full_df['Output']
video = full_df['Video']
full_df = full_df.drop('Type', axis=1)
full_df = full_df.drop('Output', axis=1)
full_df = full_df.drop('Video', axis=1)

# Obtain the mean and standard deviation
std_values = pd.DataFrame(full_df.std()).to_csv('std_values.csv')
mean_values = pd.DataFrame(full_df.mean()).to_csv('mean_values.csv')

# Scale the values
x = full_df.values
x_scaled = preprocessing.scale(x)
full_df = pd.DataFrame(x_scaled)

# Get the results
results = full_df[video].copy()

# Get the training and testing sets
original_test = full_df[((type_series == 'Test') & (output_series == 1) & (video == False))].copy()
attack_test = full_df[(type_series == 'Test') & (output_series == 0) & (video == False)].copy()
original_train = full_df[(type_series == 'Train') & (output_series == 1) & (video == False)].copy()
attack_train = full_df[(type_series == 'Train') & (output_series == 0) & (video == False)].copy()

# Get the training and testing set
original_test['Output'] = 1
attack_test['Output'] = 0
original_train['Output'] = 1
attack_train['Output'] = 0

# Make the sets
training_set = pd.concat([original_train, attack_train]).sample(frac=1).reset_index(drop=True)
testing_set = pd.concat([original_test, attack_test]).sample(frac=1).reset_index(drop=True)

# Separate the values
train_y = training_set['Output']
train_x = training_set.drop('Output', axis=1)
test_y = testing_set['Output']
test_x = testing_set.drop('Output', axis=1)

# Train the SVM
svm = NuSVC(nu=0.2)
svm.fit(train_x, train_y)
predict = svm.predict(test_x)
score = svm.score(test_x, test_y)

# Over the training set
predict_train = svm.predict(train_x)
score_train = svm.score(train_x, train_y)

# Get the confusion matrix
plot_confusion_matrix(np.array(test_y).flatten(), np.array(predict).flatten(), np.array(['Attack', 'Original']),
                      normalize=True, title='Confusion Matrix del Clasificador de Liveliness')
plt.show()


# Get the confusion matrix for training
plot_confusion_matrix(np.array(train_y).flatten(), np.array(predict_train).flatten(), np.array(['Attack', 'Original']),
                      normalize=True, title='Confusion Matrix del Clasificador de Liveliness Testing Set')
plt.show()

# Get the prediction for the video
predict_video = svm.predict(results)

# Calculate the mean of the video
mean_value_video = predict_video.mean()

# Play the video with the prediction
video_test.play_video_prediction(mean_value_video)

# Save the model
filename = 'svm_model.sav'
pickle.dump(svm, open(filename, 'wb'))
