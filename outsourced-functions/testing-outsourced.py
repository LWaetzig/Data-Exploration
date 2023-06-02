import os
import pandas as pd
from tensorflow import keras
from functions import load_data


# Ordnerstruktur bereitstellen
data_dir = 'data'
test_dir = os.path.join(data_dir, 'Test')
train_dir = os.path.join(data_dir, 'Train')


# CSV-Dateien einlesen
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))


# Pfad zur gespeicherten Modelldatei
model_path = os.path.join('saved-models', 'final_model.h5')


# Trainingsdaten laden
train_images, train_labels = load_data(train_df)


# Testdaten laden
test_images, test_labels = load_data(test_df)


# Modell laden
model = keras.models.load_model(model_path)


# Modell auf Testdaten evaluieren
train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=2)
print(f'Trainings accuracy: {train_acc}')


# Modell auf Testdaten evaluieren
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')