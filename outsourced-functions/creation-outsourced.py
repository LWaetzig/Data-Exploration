import os
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from functions import load_data, create_model



# Ordnerstruktur bereitstellen
data_dir = 'data'
train_dir = os.path.join(data_dir, 'Train')
save = os.path.join('saved-models', 'final_model.h5')


# CSV-Datei einlesen
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))


# Trainingsdaten laden
train_images, train_labels = load_data(train_df)


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
