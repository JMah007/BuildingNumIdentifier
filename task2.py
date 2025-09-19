

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


def run_task2(image_path, config):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return


    # Convert image from RGB to HSV to separate hue from intensity and use value channel (brightness) for thresholding to eleminate different background colours
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        
    processed = cv2.equalizeHist(gray_img)
    
    #using median blur to remove salt and pepper noise while preserving edges better than gaussian blur
    processed = cv2.medianBlur(processed, ksize=5)


    # need to make white parts even more whiter and black parts even more blacker to make thresholdin  g easier
    # might use adaptive thresholding as lighting might be uneven across the image
    
   # Thresholding to get binary image as contour only works on binary images. Smaller 5th parameter preserves more detail in digits
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 5)
    
     # could apply dilation to make digits more visible for detector but only if their white as dilation makes white areas larger and black areas smaller (need to eleminate white in background)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=2)


    
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and extract digit regions
    import shutil
    digit_folder = "output/task2/bn1"

    shutil.rmtree("output/task2/bn1")
    os.makedirs(digit_folder, exist_ok=True)
    count = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 7 and h > 14:  # Filter small contours
            digit = image[y:y+h, x:x+w]
            digit_path = os.path.join(digit_folder, f"c{count}.png")
            cv2.imwrite(digit_path, digit)
            count += 1
            
    contour_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
    cv2.imwrite("output/task2/contours_visual.png", contour_img)



    output_path = f"output/task2/processed_image.png"
    save_output(output_path, processed, output_type='image')
    save_output(output_path, gray_img, output_type='image')
