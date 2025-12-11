

# Machine Perception Assignment
#### Note this should be run on linux environment

Directory Structure:
Inside this folder it has the structure:

- task1.py : script for detecting if an image contains a valid building number and if so isolates it. The output is saved to `output/task1`
- task2.py : script for segmenting isolated building number from task1. The segmented digits are saved to `output/task2`
- task3.py : script for classifying what digits have been segmented from task2. The output is saved to `output/task3`
- task4.py : script combining all the steps into one complete pipeline. The final classification of the building number is saved as text file to `output/task4`
    
- packages/ : Folder containing any Python packages that are not installable via pip (optional).
- data/best.pt : contains weights for the YOLO pretrained model obtained from `training_script_task1.py`
- data/svm_model.pkl : contains weights for the pretrained SVM model obtained from `training_script_task3.py`

- training_script_task1.py : contains the training script for the building number detection model built off YOLOv8
- training_script_task3.py : contains the training script for the digit classifier SVM model

How to Run:
------
- assignment.py : **Do not modify this file**. It handles execution for tasks 1, 2, 3, and 4. 
  
  Example usage:
  `python assignment.py task1 /path/to/images/for/task1`
  `python assignment.py task2 /path/to/images/for/task2`

- the path to images is the images used for input for that task

- requirements.txt : List of acceptable Python libraries for your project.
