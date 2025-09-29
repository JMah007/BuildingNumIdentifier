

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
import pickle
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
        


def preprocess_image(image):    
    """ Preprocess the input image for better segment detection.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The preprocessed image.
    """

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (20, 20))
    
    processed = cv2.equalizeHist(img_resized)

    processed = cv2.medianBlur(processed, ksize=5)

    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 5)
    
    return processed


def run_task3(image_path, config):
    
    with open('data/svm_model.pkl', 'rb') as f:
        svm_linear = pickle.load(f)


    hog = cv2.HOGDescriptor(_winSize=(20, 20), _blockSize=(10, 10),
                                _blockStride=(5, 5), _cellSize=(10, 10), _nbins=9)
    
    def extract_hog_features(images):
        hog_features = []
        for img in images:
            hog_feature = hog.compute(img).flatten()
            hog_features.append(hog_feature)
        return np.array(hog_features)
    

    # Extract label number from filename
    filename = os.path.basename(image_path)  
    digit = ''.join(filter(str.isdigit, filename)) 
    
        
    # Loop through all bn* folders in image_path
    bn_folders = sorted(glob.glob(os.path.join(image_path, 'bn*')))
    for bn_folder in bn_folders:
        digit = ''.join(filter(str.isdigit, os.path.basename(bn_folder)))
        digit_folder = f"output/task3/bn{digit}"
        os.makedirs(digit_folder, exist_ok=True)

        image_files = sorted(glob.glob(os.path.join(bn_folder, '*.png')))
        for img_path in image_files:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Unable to load image at {img_path}")
                continue

            processed = preprocess_image(image)
            hog_feature = hog.compute(processed).flatten().reshape(1, -1)
            predicted_label = svm_linear.predict(hog_feature)
            print(f"Predicted digit: {predicted_label[0]}")

            # Use the image filename for the output text file
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(digit_folder, f"{img_name}.txt")
            save_output(output_path, f"{predicted_label[0]}", output_type='txt')
