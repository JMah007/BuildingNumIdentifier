

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


# Author: Jaeden Mah
# Last Modified: 01/10/2025

from ultralytics import YOLO
import os
import cv2 as cv2
import glob
import shutil

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
    # Choose detection with highest confidence score
    highest_conf = 0
    best_crop = None
    for det in detections:  
        if det[4] > highest_conf:
            highest_conf = det[4] # 5th parameter is confidence score
            x1, y1, x2, y2 = map(int, det[:4])  # convert to integers
            best_crop = image[y1:y2, x1:x2]
    return best_crop


def filter_detections(image, detections, conf_threshold):
    """ Filter detections that arent possible and then select the one with highest confidence score. Only one candidate should remain.

    Args:
        detections (list): A list of detections, where each detection is a list
            containing [x1, y1, x2, y2, confidence, class].
        conf_threshold (float): The confidence threshold to filter detections.

    Returns:
        list: A list of filtered detections.
    """
    
    # Get detections from YOLOv8 results object
    boxes = detections.boxes
    dets = boxes.xyxy.cpu().numpy()  # shape: (N, 4)
    confs = boxes.conf.cpu().numpy()  # shape: (N,)
    # Combine into one array for filtering
    all_dets = []
    for i in range(len(dets)):
        all_dets.append([*dets[i], confs[i]])

    filtered_detections = [det for det in all_dets if det[4] >= conf_threshold]
    
    # If there is still detections left with conf score over threshold
    if filtered_detections:
        best_crop = find_highest_confidence(image, filtered_detections)
        return best_crop
    return None
    
    
def run_task1(image_path, config):
    # Wipe output/task1 directory before saving new detections
    # Assistance provided by GitHub Copilot (AI programming assistant) for removing file contents.
    output_dir = "output/task1"
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    image_files = sorted(glob.glob(os.path.join(image_path, '*.jpg')))

    # Load the YOLOv8 model 
    model = YOLO('./data/best.pt')

    print("\n\nFinal results for task 1...........\n")
   
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Unable to load image at {img_path}")
            continue

        # Extract label number from filename
        filename = os.path.basename(img_path)
        digit = ''.join(filter(str.isdigit, filename))

        # Prediction using the model
        detections_list = model.predict(source=img_path, save=False)
        
        if not detections_list or detections_list[0].boxes is None: # If no detections found
            print(f"No detections found for {img_path}.")
            return
        
        detections = detections_list[0]
        best_crop = filter_detections(image, detections, 0.2) # hard code minimum confidence to 0.2
        if best_crop is not None:
            output_path = f"output/task1/bn{digit}.png"
            save_output(output_path, best_crop, output_type='image')
        else:
            print(f"Detections found but none of them were deemed valid for {img_path}.")
            