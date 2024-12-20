#!/bin/bash

# Set the file and directory paths
path_to_test_input_csv='./input_ecg_features_12-lead_test.csv' # Path to the input CSV file for ECG features
path_to_test_label_csv='./label_test.csv' # Path to the input CSV file for classificaiton labels. Set to 'None' if the labels are not available.
label_name='prediabetes_diabetes' # Name of the column in the label CSV file that contains the labels. Set to 'None' if the labels are not available.
model_dir='model_12-lead' # Directory where the saved models are stored.
output_dir='./test_reusults' # Directory to save the output  

# Assume the code for testing and the directory of the saved models are in the current directory
python test.py \
    --input_test_csv ${path_to_test_input_csv} \
    --label_test_csv ${path_to_test_label_csv} \
    --label_name ${label_name} \
    --output_dir ${output_dir} \
    --model_dir ${model_dir} \
    --evaluation # Uncomment this line if you want to evaluate the model