import argparse
import os

from src.Model import Model


def main(args):


    model_path = os.path(args.model)
    image_path = os.path(args.image)

    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} does not exist")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} does not exist")
        return

    model = Model(model_path)
    model.load_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", "-i", type=str, required=True, help='Path to image')
    parser.add_argument("--model", "-m", type=str, required=True, help='Path to model to use for prediction')
    args = parser.parse_args()
    
    main(args)




'''
# Beispielverwendung
model = Model()

# Trainingsdaten laden
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
model.train_model(train_df, test_df, validation_split=0.2, batch_size=16, epochs=10)

# Modell evaluieren
evaluation_report = model.evaluate_model(test_df)
print(evaluation_report)

# Einzelne Bildvorhersage
image_path = 'path/to/image.jpg'
predicted_label = model.model_prediction(image_path)
print(f'Predicted label: {predicted_label}')

# Gespeichertes Modell laden
model.load_model('path/to/saved/model')

# Vorgeschlagene Hyperparameter für die Grid-Suche
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30]
}

# Finden der besten Hyperparameter
best_params = model.find_best_hyperparams(param_grid)

# Modell mit den besten Hyperparametern trainieren
model.train_model(best_params['epochs'], best_params['batch_size'])
'''