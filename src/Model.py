import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


class Model:

    def __init__(self, image_size=(32, 32), color='RGB'):
        """
        Initialisierungsmethode:
        image_size: Gewünschte Größe zur Formatierung der Bilder.                           Werte: (32, 32) oder andere Zahlen einsetzen.
        color: Angeben ob das Modell mit Farbbildern oder Graustufen arbeiten soll.         Werte: 'RGB' oder 'GRAY'
        """
        self.image_size = image_size

        if color == 'GRAY':
            self.colorload = cv2.IMREAD_GRAYSCALE
            self.colorchannels = 1
        else:
            self.colorload = None
            self.colorchannels = 3

        self.model = None


    def load_image(self, image_path, image_size):
        img = cv2.imread(image_path, self.colorload)
        img = cv2.resize(img, image_size)
        img = img / 255.0
        return img

    def load_data(df, image_size):
        images = []
        labels = []
        for i, row in df.iterrows():
            path = row['Path']
            label = row['ClassId']
            image_path = os.path.join(path)
            image = Model.load_image(image_path, image_size)
            images.append(image)
            labels.append(label)
        return np.array(images), np.array(labels)

    def create_model(self, num_classes):
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], self.colorchannels)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def find_best_hyperparams(self, param_grid, cv=3):
        keras_model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=self.create_model, verbose=1)

        # Grid-Suche durchführen
        grid_search = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=cv)
        grid_search.fit(self.train_images, self.train_labels)

        # Beste Hyperparameter auslesen
        best_params = grid_search.best_params_

        return best_params

    def train_model(self, train_df, validation_split=0.2, batch_size=16, epochs=10):
        train_images, train_labels = self.load_data(train_df, self.image_size)

        num_classes = len(np.unique(train_labels))

        self.model = self.create_model(num_classes)
        self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def evaluate_model(self, test_df):
        test_images, test_labels = self.load_data(test_df, self.image_size)
        predictions = self.model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        report = classification_report(test_labels, predicted_labels)
        return report

    def model_prediction(self, image_path):
        image = self.load_image(image_path, self.image_size)
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)
        predicted_label = np.argmax(prediction)
        return predicted_label

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
