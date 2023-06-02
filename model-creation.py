import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import GridSearchCV



# Ordnerstruktur bereitstellen
data_dir = 'data'
train_dir = os.path.join(data_dir, 'Train')
save = os.path.join('saved-models', 'final_model.h5')


# CSV-Datei einlesen
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


# Anzahl der Klassen/Labels auslesen um Modell entsprechend vorzubereiten
num_classes = len(np.unique(train_labels))


# Funktion zur Modellerstellung (Die verschiedenen Layer festlegen und anschließend Modell testen um während des Trainings zu überwachen)
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)), # Bei input_shape die letzte Zahl von 1 auf 3 ändern, sollten die Bilder NICHT in Graustufen eingelesen werden
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


# Modell im dafür vorgesehenen Ordner speichern
model.save(save)
