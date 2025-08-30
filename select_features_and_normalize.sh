#!/bin/zsh

# Set the file and directory paths
test_input_csv_directory='' # Path to the directory containing the CSV file of extracted ECG features
test_input_csv_filename='' # Filename of the CSV file of extracted ECG features
number_of_leads=12 # Number of leads used (1 or 12)
output_csv_filename='ecg_features_filled_std.csv' # Filename of the CSV file of selected features and normalized features. Stored in the same directory as the input CSV file.
median_mean_std_csv='./median_mean_std_dev_dataset.csv' # Path to the CSV file of median, mean, and standard deviation of the development data

if [ $number_of_leads -eq 12 ]; then
    feature_list_csv='./feature_list_269_12-lead.csv'
elif [ $number_of_leads -eq 1 ]; then
    feature_list_csv='./feature_list_28_1-lead.csv'
else
    echo "Invalid number of leads: $number_of_leads. Please set 1 or 12."
    exit 1
fi

python select_features_and_normalize.py \
    --input_csv ${test_input_csv_directory}/${test_input_csv_filename} \
    --feature_list_csv ${feature_list_csv} \
    --output_csv ${test_input_csv_directory}/${output_csv_filename} \
    --median_mean_std_csv ${median_mean_std_csv}