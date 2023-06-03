import cv2 as cv
import numpy as np


class ImagePreprocessing:
    def __init__(self) -> None:
        pass

    def preprocess_image(self, image_path: str) -> np.array:
        """Apply some preprocessing steps to an image

        Args:
            image_path (str): path to image

        Returns:
            np.array: processed image as numpy array
        """
        image = cv.imread(image_path)
        # convert image to grayscale and resize to 32x32
        processed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        processed_image = cv.resize(processed_image, (32, 32))
        # apply sharpen filter
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_image = cv.filter2D(image, -1, kernel)
        # normalize pixel values
        processed_image = processed_image / 255
        return processed_image

    def augment_data(self) -> None:
        pass
