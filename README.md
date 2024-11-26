# LightGBM for detection of prediabetes/diabetes from ECG features
<img width="700" alt="image" src="https://github.com/user-attachments/assets/8301ebfa-5699-45bd-9c7b-ffe864f963a9">

## Demo
### ECG feature extraction
We used ECG-featurizer (https://github.com/Bsingstad/ECG-featurizer), with editing codes to take a CSV file as input.  
* ecg_feature_extraction.sh:  
  Operates ECG feature extraction and returns matrix of #samples x #ECG-features.  
  Specify the path for input and output directories, and sampling rate of the ECGs (Hz).  
  It uses feature_extractor.py and ecg_feature_extraction.py.

### Training (if necessary)  
Training may be too heavy for a local environment, depending on the sample size and number of ECG features used.
* train.sh:  
  Runs the training with hyperparameter optimization for 1000 trials by default.
  It uses train.py.  
  Specify the following parameters  
  * Path to the CSV file of input ECG features
  * Path to the CSV file of classification labels
  * The column name in the CSV file for the classification label
  * Path to the output directory
  * Number of the CPU cores to use (set 1 by default)

### Prediction
If the classification labels for test data are available, the evaluation of prediction performance will be performed.
It uses test.py.   
* test.sh:  
  Uses test.py to run the test with trained models provided.  
  If necessary, returns the results of prediction, classification performance report, ROC curve, SHAP summary plot and feature importance.  
  Specify the following parameters
  * Path to the CSV file of input ECG features for the test
  * Path to the CSV file of classification labels for the test (not needed if 
  * The column name in the CSV file for the classification label
  * Path to the directory where the models and the classification threshold are saved
  * Path to the output directory
  * Whether evaluation of predictive performance is needed (uncomment the far bottom line of the script to enable evaluation)
