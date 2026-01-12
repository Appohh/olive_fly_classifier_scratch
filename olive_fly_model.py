import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random

from skimage.feature import hog
from skimage.measure import label
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

def extract_foreground(img, kernel_size=9, background_color=255) :
    """
    Extract the foreground from an image. Works by assuming we are looking
    for something dark on a light background. It does an inverse Otsu
    threshold to get a binary image. This is cleaned by morphological closing. 
    Finally, we select the largest connected component by labelling and sorting.
    
    @param img: the image to be processed. Should be three channel image.
    @param kernel_size: kernel size for Morphological closing.
        Larger values will result in less noise, but lower resolution masks.
    @param background color. All parts of the image not in the foreground
    will be replaced by this color. Can also be a tuple eg: (255,255,0)

    returns two matrices: the foreground and a mask. 
    """
    # convert to grayscale, and make sure result is 8 bit
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    # use inverse OTSU threshold to get the dark parts (likely insects)
    thresh, img_bw = cv2.threshold(img_gray,-1, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # cleanup a bit with morphological closing
    kernel = np.ones((kernel_size, kernel_size))
    img_bw_cleaned = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
    
    # label with skimage
    labels = label(img_bw_cleaned)
    
    # get the largest labeled region, other than the background
    label_of_largest_region = np.argmax(np.bincount(labels.flat, 
                                                    weights=img_bw_cleaned.flat))
    largest_region = labels == label_of_largest_region
    

    # apply largest region as mask
    x, y = np.where(np.invert(largest_region))
    foreground = img.copy()
    foreground[x,y] = background_color
    
    return foreground, largest_region
  
def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(6, 6),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

def extract_color_histogram(img, bins=(8, 8, 8)):
    """
    Extract a color histogram from an image and normalize it.
    
    Parameters:
    - img: BGR image
    - bins: tuple, number of bins per channel (R,G,B)
    
    Returns:
    - 1D feature vector
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Compute 3D histogram
    hist = cv2.calcHist([img_rgb], [0,1,2], None, bins,
                        [0,256, 0,256, 0,256])
    
    # Normalize histogram
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

def pre_process_data(image_path: str) -> np.ndarray:
    X_combined = []
    
    image = cv2.imread(image_path)
    image = cv2.resize(src=image, dsize=(170, 170))
    
    forground, mask = extract_foreground(image)
            
    hog_features = extract_hog_features(forground)
    color_features = extract_color_histogram(forground)
    
    combined = np.concatenate([hog_features, color_features])
    X_combined.append(combined)

    # Convert to numpy arrays
    X_combined = np.array(X_combined)

    return X_combined

def detect_olive_fly(image_path: str) -> bool:
    """
    Detects the presence of olive fly in the given image using a pre-trained logistic regression model.
    
    :param image: Input image in which to detect olive fly.
    :return: True if olive fly is detected, False otherwise.
    """
    # Load model parameters
    model_params = np.load(MODEL_FILE)
    w = model_params['w']
    b = model_params['b']
    
    # Preprocess the image
    img_normalized = pre_process_data(image_path)
    
    # Predict using the logistic regression model
    prediction = predict(w, b, img_normalized)

    return prediction

def main():
    # Example usage
    # Select a random subdirectory and image
    dataset_dir = 'Dataset-for-miniproject'
    subdirs = ['not_olive_fly', 'olive_fly']
    random_subdir = random.choice(subdirs)
    subdir_path = os.path.join(dataset_dir, random_subdir)

    # Get all images from the selected subdirectory
    images = [f for f in os.listdir(subdir_path) if f.endswith(('.JPG', '.jpeg', '.png', '.jpg'))]
    random_image = random.choice(images)
    test_image_path = os.path.join(subdir_path, random_image)

    # Load the imag
    
    if test_image_path is None:
        print(f"Error: Could not read image {test_image_path}")
        return
      
    print(f"Testing image: {test_image_path}")
    
    if detect_olive_fly(test_image_path):
        print("Olive fly detected in the image.")
    else:
        print("No olive fly detected in the image.")

if __name__ == "__main__":
    main()
  