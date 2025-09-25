

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: [Your Name]
# Last Modified: 2024-09-09

import os
import glob
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn import svm


def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")
        




def run_task3(image_path, config):
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


    
    digit_images, digit_labels = shuffle(digit_images, digit_labels, random_state=42)
            
            
    
    print(np.shape(digit_images))
    print(np.shape(digit_labels))
 

    #need tos plit data and extarct hog features
    train_hog = extract_hog_features(digit_images[ :2825], ).astype(np.float32)
    test_hog = extract_hog_features(digit_images[2826:5650]).astype(np.float32)
    
    train_hog_labels = np.array(digit_labels[ : 2825])
    test_hog_labels = np.array(digit_labels[2826:5650])
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
    

    
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (20, 20))
    
    processed = cv2.equalizeHist(img_resized)

    processed = cv2.medianBlur(processed, ksize=5)

    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 5)
    
    # Extract HOG features
    hog_feature = hog.compute(processed).flatten().reshape(1, -1)
    
    # Predict
    predicted_label = svm_linear.predict(hog_feature)
    print(f"Predicted digit: {predicted_label[0]}")

    #output_path = f"output/task3/result.txt"
    #save_output(output_path, "Task 3 output", output_type='txt')
