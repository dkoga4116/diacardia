#!/bin/bash

path_to_input_csv='input_ecg_features_1-lead_test.csv' # input csv file
path_to_label_csv='label_test.csv' # label csv file
path_to_output_directory='results_241127' # output directory
path_to_model_directory='models' # Directory to save models and the classificaiton threshold
thread_count=2 # number of threads
number_of_trials=1000 # number of trials for hyperparameter optimization (1000 by default)

pred_label='prediabetes_diabetes' # column name in label csv file as label for prediction

python train.py \
        --input_csv ${path_to_input_csv} \
        --label_csv ${path_to_label_csv} \
        --pred_label ${pred_label} \
        --output_dir ${path_to_output_directory} \
        --model_dir ${path_to_model_directory} \
        --thread_count ${thread_count} \
        --n_trials ${number_of_trials} \