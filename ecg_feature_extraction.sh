#!/bin/zsh

input_directory_path='' # Path to the directory containing the ECG data files
output_directory_path='' # Path to the directory where the extracted features will be saved

# Run the Python script for ECG feature extraction
python ecg_feature_extraction.py \
    --input_directory_path ${input_directory_path} \
    --output_directory_path ${output_directory_path} \
    --sample_frequency 500 # Adjust the sample frequency as needed