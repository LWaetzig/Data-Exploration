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