

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


def run_task1(image_path, config):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    # convert to grayscale 
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    processed = cv2.equalizeHist(gray_img)
    # apply gaussian blur to reduce noise and improve thresholding
    processed = cv2.GaussianBlur(processed, (5, 5), 0)
    # apply adaptive thresholding to get binary image
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)






# -	Keep white-on-black regions.
# -	Drop regions that look like bricks/other colors/textures.
# -	This prunes to a small set (ideally ~10–20 candidate regions).

    
    # could apply dilation to make digits more visible for detector but only if their white as dilation makes white areas larger and black areas smaller 
    output_path = f"output/task1/result.txt"
    save_output(output_path, "Task 1 output", output_type='txt')
