#!/bin/zsh

# The following variables must be set before running the script
input_directory_path='./ECG_samples' # Path to the directory containing the ECG data files
output_directory_path='./ecg_features' # Path to the directory where the extracted features will be saved
voltage_unit=4.88 # The voltage unit of the ECG data in microvolts (e.g. 4.88 for our original dataset)
sampling_frequency=500 # The sampling frequency of the ECG data in Hz (e.g. 500 for our original dataset)

# Run the Python script for ECG feature extraction
python ecg_feature_extraction.py \
    --input_directory_path ${input_directory_path} \
    --output_directory_path ${output_directory_path} \
    --voltage_unit ${voltage_unit} \
    --sampling_frequency ${sampling_frequency}