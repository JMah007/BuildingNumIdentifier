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
# Microsoft. (2025). Copilot (GPT-4) [Large Language Model]. https://copilot.microsoft.com/
# Below is the dataset from online database https://www.kaggle.com/datasets/dhruvmomoman/printed-digits?resource=download
charList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for char in charList:
    folder = f"/home/jaeden/training_data/{char}"
    for img_path in glob.glob(os.path.join(folder, "*.png")): 
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (20, 20))
        if img is not None:
            digit_images.append(img_resized)
            digit_labels.append(char)

        
# Below is my own made dataset with letters A-D and Digit 0-9   
charList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
for char in charList:
    folder = f"/home/jaeden/training_data/CustomData/{char}"
for img_path in glob.glob(os.path.join(folder, "*.jpg")): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    if img is not None:
        digit_images.append(img_resized)
        digit_labels.append(char)
 


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