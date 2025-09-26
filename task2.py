

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

import glob
import os
import cv2 as cv2
from matplotlib import pyplot as plt
import numpy as np


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
    """ Preprocess the input image for better contour detection.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The preprocessed image.
    """

    # Convert image from RGB to HSV to separate hue from intensity and use value channel (brightness) for thresholding to eleminate different background colours
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
    processed = cv2.equalizeHist(gray_img)
    
    # Get dimensions of the cropped number region so i can apply different preprocessing for different sized images
    h, w = processed.shape[:2]
    
    if w > 300 or h > 200: # image is fairly large
        #using median blur to remove salt and pepper noise while preserving edges better than gaussian blur
        processed = cv2.medianBlur(processed, ksize=5)

    # might use adaptive thresholding as lighting might be uneven across the image
    # Thresholding to get binary image as contour only works on binary images. Smaller 5th parameter preserves more detail in digits
        processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 7)
        
        # could apply dilation to make digits more visible for detector but only if their white as dilation makes white areas larger and black areas smaller (need to eleminate white in background)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=2)
    
    
    else: # image is fairly small
        processed = cv2.medianBlur(processed, ksize=3)
        
        # Use otsu's thresholding as its better for smaller res images
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # could apply dilation to make digits more visible for detector but only if their white as dilation makes white areas larger and black areas smaller (need to eleminate white in background)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)

    return processed


def extract_digits(image, contours, file_name_digit):
    """ Extract and save digit regions from the image based on contours.

    Args:
        image (numpy.ndarray): The input image.
        contours (list): List of contours detected in the image.
        digit_folder (str): Folder path to save the extracted digit images.
    """
    count = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) 
        if (w > 10 and h > 20) and (1.0 < aspect_ratio):  # Only consider contours that are likely to be digits based on size and aspect ratio
            digit = image[y:y+h, x:x+w]

            output_path = os.path.join(f"output/task2/bn{file_name_digit}", f"c{count}.png")
            save_output(output_path, digit, output_type='image')
            count += 1


def run_task2(image_path, config):
    """Run the second task of the pipeline.

    Args:
        image_path (str): Path to the directory holding all the input images.
        config (dict): Configuration parameters for the task.
    """
    
    image_files = sorted(glob.glob(os.path.join(image_path, '*.png'))) # Extract the image files from the directory
    
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return
        
        # Extract building filename label number to be later used in naming output files
        filename = os.path.basename(img_path)  
        file_name_digit = ''.join(filter(str.isdigit, filename)) 

        pre_processed_image = preprocess_image(image)            
        
        contours, _ = cv2.findContours(pre_processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        extract_digits(image, contours, file_name_digit)

        
        
        
        
        # # Save contour image and preprocessed image. Can delete this and below later
        # output_dir = f"output/task2/bn{file_name_digit}"

        # # Save contour image to visualise detected contours
        # contour_img = cv2.cvtColor(pre_processed_image, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
        # contour_img_path = os.path.join(output_dir, "contours.png")
        # save_output(contour_img_path, contour_img, output_type='image')

        # # Save preprocessed image
        # preprocessed_img_path = os.path.join(output_dir, "preprocessed.png")
        # save_output(preprocessed_img_path, pre_processed_image, output_type='image')
