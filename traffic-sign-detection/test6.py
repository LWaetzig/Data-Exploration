import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# Pfad zum Datenordner
data_dir = 'data'
train_dir = os.path.join(data_dir, 'Train')
test_dir = os.path.join(data_dir, 'Test')
meta_dir = os.path.join(data_dir, 'Meta')

# CSV-Dateien einlesen
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Bildgröße
image_size = (32, 32)

# Funktion zum Laden der Bilder
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = img / 255.0  # Normalisierung der Pixelwerte auf den Bereich [0, 1]
    return img

# Funktion zum Laden der Trainingsdaten
def load_train_data(df):
    images = []
    labels = []
    for i, row in df.iterrows():
        image_path = os.path.join(row['Path'])
        image = load_image(image_path)
        label = row['ClassId']
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Funktion zum Laden der Testdaten
def load_test_data(df):
    images = []
    labels = []
    for i, row in df.iterrows():
        image_path = os.path.join(row['Path'])
        image = load_image(image_path)
        label = row['ClassId']
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Trainingsdaten laden
train_images, train_labels = load_train_data(train_df)

# Testdaten laden
test_images, test_labels = load_test_data(test_df)

# Aufteilen der Trainingsdaten in Trainings- und Validierungsdaten
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Anzahl der Klassen
num_classes = len(np.unique(train_labels))

# Modell erstellen
def create_model():
    model = tf.keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# KerasClassifier erstellen
keras_model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=1)

# Hyperparameter für die Grid-Suche festlegen
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30]
}

# Grid-Suche durchführen
grid_search = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3)
grid_search.fit(train_images, train_labels)
best_params = grid_search.best_params_

# Modell mit besten Hyperparametern trainieren
model = create_model()
model.fit(train_images, train_labels, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

# Modell auf Validierungsdaten evaluieren
val_loss, val_acc = model.evaluate(val_images, val_labels, verbose=2)
print(f'Validation accuracy: {val_acc}')

# Modell auf Testdaten evaluieren
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Vorhersagen für Testdaten
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Vergleich der Vorhersagen mit den wahren Labels
accuracy = accuracy_score(test_labels, predicted_labels)
print(f'Accuracy on test data: {accuracy}')
