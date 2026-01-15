import numpy as np
import cv2
import os
import random

from skimage.feature import hog
from skimage.measure import label

MODEL_FILE = 'olive_model_params.npz'
DATA_FILE = 'olive_data.npz'

model_params = np.load(MODEL_FILE)
weights = model_params['w']
bias = model_params['b']

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(w: np.ndarray, b: float, X: np.ndarray) -> bool:
  Y_hat = sigmoid(np.dot(X, w) + b)
  Y_prediction = (Y_hat > 0.5).astype(int)

  return Y_prediction

def extract_foreground(img, kernel_size=9, background_color=255) :
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    thresh, img_bw = cv2.threshold(img_gray,-1, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((kernel_size, kernel_size))
    img_bw_cleaned = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
    
    labels = label(img_bw_cleaned)
    
    label_of_largest_region = np.argmax(np.bincount(labels.flat, 
                                                    weights=img_bw_cleaned.flat))
    largest_region = labels == label_of_largest_region
    

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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hist = cv2.calcHist([img_rgb], [0,1,2], None, bins,
                        [0,256, 0,256, 0,256])
    
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

def pre_process_data(image) -> np.ndarray:
    X_combined = []
    
    image = cv2.resize(src=image, dsize=(170, 170))
    
    forground, mask = extract_foreground(image)
            
    hog_features = extract_hog_features(forground)
    color_features = extract_color_histogram(forground)
    
    combined = np.concatenate([hog_features, color_features])
    X_combined.append(combined)

    # Convert to numpy arrays
    X_combined = np.array(X_combined)

    return X_combined

def detect_olive_fly(image) -> bool:
    img_normalized = pre_process_data(image)
    prediction = predict(weights, bias, img_normalized)

    return prediction

def load_random_image_from_dataset(dataset_dir: str) -> str:
    subdirs = ['not_olive_fly', 'olive_fly']
    random_subdir = random.choice(subdirs)
    subdir_path = os.path.join(dataset_dir, random_subdir)

    # Get all images from the selected subdirectory
    images = [f for f in os.listdir(subdir_path) if f.endswith(('.JPG', '.jpeg', '.png', '.jpg'))]
    random_image = random.choice(images)
    return os.path.join(subdir_path, random_image)

def main():
    dataset_dir = 'Dataset-for-miniproject'
    test_image_path = load_random_image_from_dataset(dataset_dir)
    
    if test_image_path is None:
        print(f"Error: Could not read image {test_image_path}")
        return
      
    print(f"Testing image: {test_image_path}")

    image = cv2.imread(test_image_path)
    
    if detect_olive_fly(image):
        print("Olive fly detected in the image.")
    else:
        print("No olive fly detected in the image.")

if __name__ == "__main__":
    main()
  