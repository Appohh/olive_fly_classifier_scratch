import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

MODEL_FILE = 'olive_model_params.npz'
DATA_FILE = 'olive_data.npz'

def sigmoid(z):
    """
    Sigmoid function to map any real value into the (0, 1) interval.
    
    :param z: Input value or array.
    """

    return 1 / (1 + np.exp(-z))

def predict(w: np.ndarray, b: float, X: np.ndarray) -> bool:
  """
  Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
  """

  Y_hat = sigmoid(np.dot(X, w) + b)

  Y_prediction = (Y_hat > 0.5).astype(int)

  return Y_prediction

def detect_olive_fly(image):
    """
    Detects the presence of olive fly in the given image using a pre-trained logistic regression model.
    
    :param image: Input image in which to detect olive fly.
    :return: True if olive fly is detected, False otherwise.
    """
    # Load model parameters
    model_params = np.load(MODEL_FILE)
    w = model_params['w']
    b = model_params['b']
    
    # Preprocess the image (resize, flatten, normalize)
    img_resized = cv2.resize(image, (64, 64))  # Resize to 64x64
    img_flattened = img_resized.flatten().reshape(1, -1)  # Flatten and reshape
    img_normalized = img_flattened / 255.0  # Normalize pixel values
    
    # Predict using the logistic regression model
    prediction = predict(w, b, img_normalized)

    return prediction

def main():
    # Example usage
    # Select a random subdirectory and image
    dataset_dir = 'dataset'
    subdirs = ['not_olive_fly', 'olive_fly']
    random_subdir = random.choice(subdirs)
    subdir_path = os.path.join(dataset_dir, random_subdir)

    # Get all images from the selected subdirectory
    images = [f for f in os.listdir(subdir_path) if f.endswith(('.JPG', '.jpeg', '.png', '.jpg'))]
    random_image = random.choice(images)
    test_image_path = os.path.join(subdir_path, random_image)

    # Load the image
    image = cv2.imread(test_image_path)
    
    if image is None:
        print(f"Error: Could not read image {test_image_path}")
        return
    
    if detect_olive_fly(image):
        print("Olive fly detected in the image.")
    else:
        print("No olive fly detected in the image.")

if __name__ == "__main__":
    main()
  