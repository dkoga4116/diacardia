# LightGBM for detection of prediabetes/diabetes from ECG features
![image](https://github.com/user-attachments/assets/54fb7c7c-bcb0-417b-8001-9faf2503fdaa)

# Contents
* Codes
  * For ECG feature extraction (this will need original ECG signal data, that are not provided here)
    * ecg_feature_extraction.sh
    * ecg_feature_extraction.py
    * feature_extractor.py
  * For feature selection, filling of the null data and standard scaling (when applying data to our trained model)  
    * select_features_and_normalize.sh
    * select_features_and_normalize.py
  * For training with hyperparameter optimization
    * train.sh
    * train.py
  * For prediction of prediabetes/diabetes from ECG feature
    * test.sh
    * test.py
* Data
  * Sample raw ECG signal data (5 CSV files) in the 'ECG_samples" directory for demo
  * Standarized ECG feature for demo (our test data, N=1676)
    * input_ecg_features_12-lead.csv &emsp;[12-lead ECG]
    * input_ecg_features_1-lead.csv &emsp;[Single-lead (lead I) ECG]
    * input_ecg_features_12-lead_external_cohort.csv  &emsp;[12-lead ECG _for external validation_] 
  * Trained models and the classification for test (directory)
    * model_12-lead &emsp; [for 12-lead ECG]
    * model_1-lead &emsp; [for single-lead (lead I) ECG]
    * model_12-lead_matched &emsp; [for _propensity score–matched_ 12-lead ECG]
  * Classification label for demo
    * class_labels.CSV
    * class_labels_matched.csv &emsp;[_for propensity score–matched analysis_]
    * class_labels_external_cohort.csv &emsp;[_for external validation_]
  * List of ECG features used (to be used for ECG extraction for original data)
    * feature_list_269_12-lead.csv &emsp; [269 features for 12-lead ECG analysis]
    * feature_list_28_1-lead.csv &emsp; [28 features for single-lead (lead I) analysis]
  * Table of median, mean, and standard deviations for filling of null data and normalization
    * median_mean_std_dev_dataset.csv

# Getting Started
## Installation
Clone diabetes_detector with git clone command.  
Run .sh files in the directory. Details for each script is described below.
* ecg_feature_extraction.sh  
* train.sh
* test.sh

## Requirements
### Python
Python version 3.11.4 was used for all analyses.
### Packages
* numpy>=1.23.5
* pandas>=2.0.3
* seaborn>=0.12.2
* matplotlib>=3.7.2
* lightgbm>=4.4.0
* shap>=0.44.0
* joblib>=1.2.0
* scikit-learn>=1.3.0
* optuna>=3.3.0 (if training using original data is performed)
* neurokit2>=0.2.10 (if ECG feature extraction is performed)
* wfdb>=4.1.2 (if ECG feature extraction is performed)

# Demo
## ECG feature extraction
We used ECG-featurizer (https://github.com/Bsingstad/ECG-featurizer), with editing codes to use a CSV file as input.  
### Data preparation
* Each ECG signal data is expected to be timepoint x leads matrix (if an ECG is 10-s and 500 Hz, 5000x12 matrix).    
* A header row, usually indicating the leads, is expected on the top row.  
* Unit of voltage, which means what the value "1" in the data represents, in microvolt is needed to be specfied (e.g. 4.88 in our original data).  
* Sampling frequency (Hz) is needed to be specified (e.g. 500 in our original data).
### Run feature extraction
[Noet] For a demonstration, we provide five sample raw ECG signals in the 'ECG_samples' directory.

Run ecg_feature_extraction.sh.  
This operates ECG feature extraction using feature_extractor.py and ecg_feature_extraction.py, then returns a CSV file of #samples x #ECG-features matrix.  
If you have single-lead (lead I) ECG, it identifies that there is only one column and outputs the features of lead I.  
```sh
./ecg_feature_extraction.sh
```
```sh
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
```
For your own data, please specify the following in the script: 
* input_directory_path: Path to the directory containing the ECG data files  
* output_directory_path: Path to the directory where the extracted features will be saved
  * In the output voltage feature values, "1" equals to 4.88 µV.  
  * In the output time feature values, "1" equals to 1 second.
* volatge_unit: unit of volage in the records in microvolts (e.g. 4.88 in our original data)
* sampling_frequency: Sampling frequency of ECG data (e.g. 500 in our original data)

## Select features and normalize data (when applying our trained model)  
When applying our trained model to an original dataset to make predictions, you need to perform feature selection and standard normalization.  
### Run feature selection and normalization
In our model, we used 269 selected features for 12-lead analysis and 28 selected features for single-lead analysis.   
The 269 and 28 features are listed in feature_list_269_12-lead.csv and feature_list_28_1-lead.csv, respectively.   
We filled null data with medians from our development dataset and performed standard scaling using means and standard deviations from our development dataset.
  
Run select_features_and_normalize.sh.  
Set the directory and filename for the CSV file of ECG features to process, and the number of the leads used (1 or 12).

```sh
./select_features_and_normalize.sh
```
This operates ECG feature extraction using select_features_and_normalize.py, a feature list file (feature_list_269_12-lead.csv or feature_list_28_1-lead.csv), and median_mean_std_dev_dataset.csv to generate the feature-selected and normalized input data ecg_features_filled_std.csv.

## Training
[Note] As a demonstration, the numbers of both boosting rounds and optimizaion trials are set much less than those of our original analyses, because full training process may be too heavy for a local environment.  

Run train.sh.  
This uses train.py and operates the training with hyperparameter optimization for 10 trials.   
```sh
./train.sh
```
```sh
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
```

For your own data, please specify the following in the script: 
  * path_to_input_csv: Path to the CSV file of input ECG features
  * path_to_label_csv: Path to the CSV file of classification labels
  * pred_label: The column name in the CSV file for the classification label
  * path_to_output_directory: Path to the output directory
  * path_to_model_directory: Path to the directory in which optimized models and the classification threshold will be stored
  * thread_count: Number of the CPU cores to use (set 2 in our original analyses)
  * number_of_trials: The number of trials for hyperparameter optimization (set 1000 in our original analyses)  

\*You can change the fold of cross-validation, and the number of boosting of LightGBM with adjusting the n_splits (set 10 in our original analyses) and the num_boost_rounds (set 10000 in our original analyses).

## Prediction
Run test.sh. as it is.   
This operates test.py and outputs predictive values and predicted classes in a CSV file.  
If the classification labels for test data are available, the evaluation of predictive performance including classification performance report, ROC curve, SHAP summary plot and feature importance can be performed.  
```sh
./test.sh
```
```sh
#!/bin/bash

# Set the file and directory paths
path_to_test_input_csv='./input_ecg_features_12-lead.csv' # Path to the input CSV file for ECG features
path_to_test_label_csv='./class_labels.csv' # Path to the input CSV file for classificaiton labels. Set to 'None' if the labels are not available.
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
```

For your own data, please specify the following in the script: 
* path_to_test_input_csv: Path to the CSV file of input ECG features for the test
* path_to_test_label_csv: Path to the CSV file of classification labels for the test (set 'None' needed if unavailable)
* label_name: The column name in the CSV file for the classification label (set 'None' needed if unavailable)
* model_dir: Path to the directory where the models and the classification threshold are saved
* output_dir: Path to the output directory
* Whether evaluation of predictive performance is needed (uncomment the far bottom line of the script to enable evaluation)

To reproduce the results of our propensity score–matched analysis, use _input_ecg_features_12-lead.csv_ (same as for all samples) as input, _class_labels_matched.csv_ as label data, and models in _model_12-lead_matched_ directory.  
To reproduce our external validation results, use _input_ecg_features_12-lead_external_cohort.csv_ as input, _class_labels_external_cohort.csv_ as label data, and models in _model_12-lead_ directory (same as for internal validation).