

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


def run_task2(image_path, config):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Using gaussion filtering to remove noise

    # Convert image from RGB to HSV to separate hue from intensity and use value channel (brightness) for thresholding to eleminate different background colours
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_image)
    
    # Use an if statrement to see if overall croped image is darker or lighter. If its mostly light then text must be dark so invert image to make text white on darker background
    
    # could apply dilation to make digits more visible for detector but only if their white as dilation makes white areas larger and black areas smaller 
    
    #  Contour Detection for Segmentation of individual Digits
     
    output_path = f"output/task2/result.txt"
    save_output(output_path, "Task 2 output", output_type='txt')
