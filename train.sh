#!/bin/bash

## THE FULL TRAINING MAY NEED A SUPERCOMPUTER OR A SERVER WITH HIGH COMPUTATIONAL POWER
## WITH THE SCRIPT BELOW, A SMALL TRAINING WILL BE DONE AS A DEMONSTRATION, WHICH CAN BE DONE ON A LAPTOP OR A DESKTOP COMPUTER

path_to_input_csv='./input_ecg_features_12-lead.csv' # input csv file
path_to_label_csv='./class_labels.csv' # label csv file
pred_label='prediabetes_diabetes' # column name in label csv file as label for prediction
path_to_output_directory='./training_results' # output directory
path_to_model_directory='./trained_models' # Directory to save models and the classificaiton threshold
thread_count=2 # number of threads
number_of_trials=10 # number of trials for hyperparameter optimization (set 1000 in our original study)

python train.py \
        --input_csv ${path_to_input_csv} \
        --label_csv ${path_to_label_csv} \
        --pred_label ${pred_label} \
        --output_dir ${path_to_output_directory} \
        --model_dir ${path_to_model_directory} \
        --thread_count ${thread_count} \
        --n_trials ${number_of_trials} \
        --n_splits 10 \
        --num_boost_rounds 100 # Set 10000 in our original study