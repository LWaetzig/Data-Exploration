import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



# Ordnerstrukturen bereitstellen
data_dir = 'data'
meta_dir = os.path.join(data_dir, 'Meta')
test_dir = os.path.join(data_dir, 'Test')
train_dir = os.path.join(data_dir, 'Train')


# CSV-Dateien einlesen
meta_df = pd.read_csv(os.path.join(data_dir, 'meta.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))


# Bildgröße für Bildvorverarbeitung festlegen
image_size = (32, 32)


# Funktionen zum Laden aller Daten definieren
# Funktion zum Laden und Vorverarbeiten der Bilder
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Einlesen des Bildes in Graustufen (alternativ das ", cv2.IMREAD_GRAYSCALE" entfernen)
    img = cv2.resize(img, image_size)
    img = img / 255.0  # Normalisierung der Pixelwerte auf den Bereich [0, 1]
    return img


# Funktion zum Einlesen der Bilder und deren Labels
def load_data(df):
    images = []
    labels = []
    for i, row in df.iterrows():
        path = row['Path']
        label = row['ClassId']
        image_path = os.path.join(path)
        image = load_image(image_path)
        images.append(image)
        labels.append(label)
        
    return np.array(images), np.array(labels)


# Trainingsdaten laden
train_images, train_labels = load_data(train_df)


# Testdaten laden
test_images, test_labels = load_data(test_df)


# Aufteilen der Trainingsdaten in Trainings- und Validierungsdaten um den Lernprozess zu überwachen
train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)


# Anzahl der Klassen/Labels auslesen um Modell entsprechend vorzubereiten
num_classes = len(np.unique(train_labels))


# Funktion zur Modellerstellung (Die verschiedenen Layer festlegen und anschließend Modell testen um während des Trainings zu überwachen)
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


# Erstellen des Modells, sowie Schnittstelle zwischen Tensorflow und Scikit-Learn, da die Bibliotheken normalerweise nicht kompatibel sind, wir aber unser Tensorflow Modell in Scikit-Learn Funktionen verwenden wollen
keras_model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=1)


# Vorgeschlagene Hyperparameter, aus denen die Grid-Suche die optimale Kombination wählen soll 
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30]
}


# Grid-Suche durchführen (Mit Cross-Validation) (ACHTUNG: MEHR ALS 2h RUNTIME)
grid_search = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3)
grid_search.fit(train_images, train_labels)
best_params = grid_search.best_params_


# Modell mit gefundenen besten Hyperparametern trainieren
model = create_model()
model.fit(train_images, train_labels, epochs=best_params['epochs'], batch_size=best_params['batch_size'])


# Modell auf Validierungsdaten evaluieren
valid_loss, valid_acc = model.evaluate(valid_images, valid_labels, verbose=2)
print(f'Validation accuracy: {valid_acc}')


# Modell auf Testdaten evaluieren
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')