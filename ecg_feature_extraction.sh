#!/bin/zsh

# The following variables must be set before running the script
input_directory_path='' # Path to the directory containing the ECG data files
output_directory_path='' # Path to the directory where the extracted features will be saved
voltage_unit= # The voltage unit of the ECG data in microvolts
sampling_frequency= # The sampling frequency of the ECG data in Hz

# Run the Python script for ECG feature extraction
python ecg_feature_extraction.py \
    --input_directory_path ${input_directory_path} \
    --output_directory_path ${output_directory_path} \
    --voltage_unit ${voltage_unit} \
    --sampling_frequency ${sampling_frequency}