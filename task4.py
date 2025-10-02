

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
# Last Modified: 02/10/2025

import glob
import os
import cv2 as cv2
import shutil


from task1 import run_task1
from task2 import run_task2    
from task3 import run_task3 



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


def run_task4(image_path, config):
    # Wipe contents of task 1, task 2 and task 3 output directories before running
    for outdir in ["output/task1", "output/task2", "output/task3", "output/task4"]:
        abs_outdir = os.path.join(os.getcwd(), outdir)
        if os.path.exists(abs_outdir):
            for filename in os.listdir(abs_outdir):
                file_path = os.path.join(abs_outdir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    
                    
    if not os.path.exists(image_path):
        print(f"Error: Invalid image path provided: {image_path}")
        return

    # Task 1 processes all images in the provided directory
    run_task1(image_path, config)

    # Task 2 processes all images produced by task 1
    output_task1_dir = os.path.join(os.getcwd(), "output/task1")
    run_task2(output_task1_dir, config)
    
    # Task 3 processes all images produced by task 2
    output_task2_dir = os.path.join(os.getcwd(), "output/task2")
    run_task3(output_task2_dir, config)


    # combine characters for each building from output of task 3 into one file for each building 
    output_task3_dir = os.path.join(os.getcwd(), "output/task3")
    bn_folders = sorted(glob.glob(os.path.join(output_task3_dir, 'bn*')))
    
    print("\n\nFinal results for task 4...........\n")
    for bn_folder in bn_folders:
        building_num = os.path.basename(bn_folder)  # e.g., 'bn3'
        task3_bn_folder = os.path.join(output_task3_dir, building_num)
        if not os.path.exists(task3_bn_folder):
            print(f"Warning: Task 3 output folder for {building_num} does not exist. Skipping.")
            
            continue
        
        char_files = sorted(glob.glob(os.path.join(task3_bn_folder, '*.txt')))
        combined_text = ""
        
        
        for char_file in char_files:
            with open(char_file, 'r') as f:
                char = f.read().strip()
                combined_text += char
        
        # Save combined text for the building number
        output_path = os.path.join("output/task4", f"{building_num}.txt")
        save_output(output_path, combined_text, output_type='txt')
        print(f"Combined building number for {building_num}: {combined_text}")
    