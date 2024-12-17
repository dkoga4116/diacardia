# LightGBM for detection of prediabetes/diabetes from ECG features
![image](https://github.com/user-attachments/assets/54fb7c7c-bcb0-417b-8001-9faf2503fdaa)

# Contents
* Codes
  * For ECG feature extraction (this will need original ECG signal data, that are not provided here)
    * ecg_feature_extraction.sh
    * ecg_feature_extraction.py
    * feature_extractor.py
  * For training with hyperparameter optimization
    * train.sh
    * train.py
  * For prediction of prediabetes/diabetes from ECG feature
    * test.sh
    * test.py
* Data
  * Standarized ECG feature for demo (our test data, N=1676)
    * input_ecg_features_12-lead_test.csv &emsp;[12-lead ECG]
    * input_ecg_features_1-lead_test.csv &emsp;[Single-lead (lead I) ECG]
  * Trained models and the classification for test (directory)
    * model_12-lead &emsp; [for 12-lead ECG]
    * model_1-lead &emsp; [for single-lead (lead I) ECG]
  * Classification label for demo
    * label_test.csv
  * List of ECG features used (to be used for ECG extraction for original data)
    * feature_list_269_12-lead.csv &emsp; [269 features for 12-lead ECG analysis]
    * feature_list_28_1-lead.csv &emsp; [28 features for single-lead (lead I) analysis]

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
Run ecg_feature_extraction.sh.  
```sh
./ecg_feature_extraction.sh
```
This operates ECG feature extraction using feature_extractor.py and ecg_feature_extraction.py, then returns a CSV file of #samples x #ECG-features matrix.  
If you have single-lead (lead I) ECG, it identifies the number of columns and outputs the features of lead I.  

Specify the following in the script: 
* input_directory_path: Path to the directory containing the ECG data files  
* output_directory_path: Path to the directory where the extracted features will be saved
* volatge_unit: unit of volage in the records in microvolts (e.g. 4.88 in our original data)
* sampling_frequency: Sampling frequency of ECG data (e.g. 500 in our original data)

## Training (if necessary)  
[[Warning]] Training may be too heavy for a local environment, depending on the sample size and number of ECG features used.  
Run train.sh.  
This uses train.py and operates the training with hyperparameter optimization for 1000 trials by default.   
```sh
./train.sh
```

Specify the following in the script: 
  * path_to_input_csv: Path to the CSV file of input ECG features
  * path_to_label_csv: Path to the CSV file of classification labels
  * pred_label: The column name in the CSV file for the classification label
  * path_to_output_directory: Path to the output directory
  * path_to_model_directory: Path to the directory in which optimized models and the classification threshold will be stored
  * thread_count: Number of the CPU cores to use (set 1 by default)
  * number_of_trials: The number of trials for hyperparameter optimization (set 1000 by default)  

\*You can change the fold of cross-validation, and the number of boosting of LightGBM with adjusting the n_splits (set 10 by default) and the num_boost_rounds (set 10000 by default), if necessary.

## Prediction
Run test.sh.
This operates test.py and outputs predictive values and predicted classes in a CSV file.  
If the classification labels for test data are available, the evaluation of predictive performance including classification performance report, ROC curve, SHAP summary plot and feature importance can be performed.  
```sh
./test.sh
```

Specify the following parameters
* path_to_test_input_csv: Path to the CSV file of input ECG features for the test
* path_to_test_label_csv: Path to the CSV file of classification labels for the test (set 'None' needed if unavailable)
* label_name: The column name in the CSV file for the classification label (set 'None' needed if unavailable)
* model_dir: Path to the directory where the models and the classification threshold are saved
* output_dir: Path to the output directory
* Whether evaluation of predictive performance is needed (uncomment the far bottom line of the script to enable evaluation)

To run test using the provided test data, modify test.sh as below and run.
```sh
#!/bin/bash

# Set the file and directory paths
path_to_test_input_csv='input_ecg_features_12-lead_test.csv' # Path to the input CSV file for ECG features
path_to_test_label_csv='label_test.csv' # Path to the input CSV file for classificaiton labels. Set to 'None' if the labels are not available.
label_name='prediabetes_diabetes' # Name of the column in the label CSV file that contains the labels. Set to 'None' if the labels are not available.
model_dir='./model_12-lead/' # Directory where the saved models are stored.
output_dir='./test_results/' # Directory to save the output  

# Assume the code for testing and the directory of the saved models are in the current directory
python test.py \
    --input_test_csv ${path_to_test_input_csv} \
    --label_test_csv ${path_to_test_label_csv} \
    --label_name ${label_name} \
    --output_dir ${output_dir} \
    --model_dir ${model_dir} \
    --evaluation # Uncomment this line if you want to evaluate the model
```
