

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

from fileinput import filename
import os
import torch
import cv2 as cv2
import glob

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


def find_highest_confidence(image, detections):
    """ Find the candidate with highest confidence score and return the corresponding image crop.

    Args:
        image (numpy.ndarray): The input image.
        detections (list): A list of detections, where each detection is a list
            containing [x1, y1, x2, y2, confidence, class].

    Returns:
        numpy.ndarray: The cropped image of the highest confidence detection,
            or None if no valid detections are found.
    """
    # Compare leftover candidates and choose one with highest confidence score
    highest_conf = 0
    best_crop = None
    for det in detections:  
        if det[4] > highest_conf:
            highest_conf = det[4] # 5th parameter is confidence score
            x1, y1, x2, y2, conf, cls = map(int, det[:6])  # convert to integers
            best_crop = image[y1:y2, x1:x2]
    return best_crop


def filter_detections(image, results, conf_threshold):
    """ Filter detections that arent possible and then select the one with highest confidence score. Only one candidate should remain.

    Args:
        detections (list): A list of detections, where each detection is a list
            containing [x1, y1, x2, y2, confidence, class].
        conf_threshold (float): The confidence threshold to filter detections.

    Returns:
        list: A list of filtered detections.
    """
    # filter out candidates that have lowest possibility of being a building number
    
    for det in results.xyxy[0]:
        print("Candidate has level : ", det[4])
        filtered_detections = results.xyxy[0][results.xyxy[0][:, 4] >= conf_threshold]
        
    # make sure after filtering there is still remaining candidates
    best_crop = None
    if filtered_detections.shape[0] != 0:
        best_crop = find_highest_confidence(image, filtered_detections)

        # Print results
        results.print()
        return best_crop
    
def run_task1(image_path, config):
    
    image_files = sorted(glob.glob(os.path.join(image_path, '*.jpg')))

    # Load the YOLOv5 model once
    model = torch.hub.load('/home/jaeden/yolov5', 'custom',
                           path='/home/jaeden/yolov5/runs/train/exp7/weights/best.pt',
                           source='local')

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Unable to load image at {img_path}")
            continue

        # Extract label number from filename
        filename = os.path.basename(img_path)
        digit = ''.join(filter(str.isdigit, filename))

        # Prediction using the model
        results = model(image)

        # If detections exist then loop through possible candidates and find most likely one
        if results.xyxy[0].shape[0] != 0:
            best_crop = filter_detections(image, results, 0.6) # hard code minimum confidence to 0.6
            if best_crop is not None:
                output_path = f"output/task1/bn{digit}.png"
                save_output(output_path, best_crop, output_type='image')
            else:
                print(f"Detections found but none of them were deemed valid for {img_path}.")
        else:
            print(f"No detections found for {img_path}.")


# -	Keep white-on-black regions.
# -	Drop regions that look like bricks/other colors/textures.
# -	This prunes to a small set (ideally ~10–20 candidate regions).
