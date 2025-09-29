import glob
import os
from sklearn.utils import shuffle
import numpy as np
from sklearn import svm
import pickle

import cv2



"""Train an SVM classifier on the provided digit images and labels."""
digit_images = []
digit_labels = []

hog = cv2.HOGDescriptor(_winSize=(20, 20), _blockSize=(10, 10),
                            _blockStride=(5, 5), _cellSize=(10, 10), _nbins=9)

def extract_hog_features(images):
    hog_features = []
    for img in images:
        hog_feature = hog.compute(img).flatten()
        hog_features.append(hog_feature)
    return np.array(hog_features)




# Used chatgpt to help with this part
# Load training images and labels from folders named 0-9    
folder = f"/home/jaeden/training_data/0"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(0)

folder = f"/home/jaeden/training_data/1"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(1)
        
folder = f"/home/jaeden/training_data/2"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(2)
        
    folder = f"/home/jaeden/training_data/3"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(3)

folder = f"/home/jaeden/training_data/4"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(4)
        
folder = f"/home/jaeden/training_data/5"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(5)
        
    folder = f"/home/jaeden/training_data/6"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(6)

folder = f"/home/jaeden/training_data/7"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(7)
        
folder = f"/home/jaeden/training_data/8"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(8)
        
folder = f"/home/jaeden/training_data/9"
for img_path in glob.glob(os.path.join(folder, "*.png")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(9)
        
        
        
# Below is my own made dataset with letters A-D and Digit 0-9    
folder = f"/home/jaeden/training_data/CustomData/0"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(0)
        
        
folder = f"/home/jaeden/training_data/CustomData/1"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(1)
        
        
folder = f"/home/jaeden/training_data/CustomData/2"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(2)
        
        
folder = f"/home/jaeden/training_data/CustomData/3"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(3)
        
        
folder = f"/home/jaeden/training_data/CustomData/4"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(4)
        
        
folder = f"/home/jaeden/training_data/CustomData/5"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(5)
        
        
folder = f"/home/jaeden/training_data/CustomData/6"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(6)
        
        
folder = f"/home/jaeden/training_data/CustomData/7"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(7)
        
        
             
folder = f"/home/jaeden/training_data/CustomData/8"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(8)
        
        
folder = f"/home/jaeden/training_data/CustomData/9"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(9)
        
        
folder = f"/home/jaeden/training_data/CustomData/C"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append("C")
        
        
folder = f"/home/jaeden/training_data/CustomData/D"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append("D")


# Randomise order of set of data so its passed to the model in a random order
digit_images, digit_labels = shuffle(digit_images, digit_labels, random_state=42)
        

split_idx = int(0.8 * len(digit_images))
train_hog = extract_hog_features(digit_images[:split_idx]).astype(np.float32)
test_hog = extract_hog_features(digit_images[split_idx:]).astype(np.float32)

train_hog_labels = np.array(digit_labels[:split_idx])
test_hog_labels = np.array(digit_labels[split_idx:])
print(np.shape(train_hog))
print(np.shape(train_hog_labels))


# Train SVM
svm_linear = svm.SVC(kernel="linear", C=1.0)
svm_linear.fit(train_hog, train_hog_labels)

# Predict
train_result = svm_linear.predict(train_hog)
test_result = svm_linear.predict(test_hog)

# Calculate accuracy
train_accuracy = np.mean(train_result == train_hog_labels) * 100
test_accuracy = np.mean(test_result == test_hog_labels) * 100

print(f"Train accuracy linear svm: {train_accuracy:.2f}%")
print(f"Test accuracy linear svm: {test_accuracy:.2f}%")

with open('data/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_linear, f)