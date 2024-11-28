#!/bin/bash

path_to_input_csv='' # input csv file
path_to_label_csv='' # label csv file
pred_label='' # column name in label csv file as label for prediction
path_to_output_directory='' # output directory
path_to_model_directory='' # Directory to save models and the classificaiton threshold
thread_count=2 # number of threads
number_of_trials=1000 # number of trials for hyperparameter optimization (1000 by default)

python train.py \
        --input_csv ${path_to_input_csv} \
        --label_csv ${path_to_label_csv} \
        --pred_label ${pred_label} \
        --output_dir ${path_to_output_directory} \
        --model_dir ${path_to_model_directory} \
        --thread_count ${thread_count} \
        --n_trials ${number_of_trials} \
        --n_splits 10 \
        --num_boost_rounds 10000