import optuna
import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import argparse
from warnings import warn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import datetime
import joblib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_csv', type=str, required=True,help='Path to the input csv file')
    parser.add_argument('--label_csv', type=str, required=True,help='Path to the label csv file')
    parser.add_argument('--pred_label', type=str, required=True,help='Name of the label column')
    parser.add_argument('--n_trials', type=int, default=1000, help='Number of trials for the optimization')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of splits for the cross-validation')
    parser.add_argument('--num_boost_rounds', type=int, default=10000, help='Number of boosting rounds')
    parser.add_argument('--output_dir', type=str, required=True,help='Path to the output directory')
    parser.add_argument('--model_dir', type=str, required=True,help='Path to the directory to save the models and the threshold')
    parser.add_argument('--thread_count', type=int, default=1, help='Number of threads to use')
    
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    
    # Create the output directories if it does not exist
    output_dir = args.output_dir
    model_dir = args.model_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load input and label data
    print("LightGBM, Start training...")
    
    input_df = pd.read_csv(args.input_csv, sep=',', header=0, index_col='ecg_id', encoding='cp932')
    label_df = pd.read_csv(args.label_csv, sep=',', header=0, index_col='ecg_id', encoding='cp932')
    
    # merge input and label files to match the indices
    label_df_matched = label_df[label_df.index.isin(input_df.index)]

    # get the label column
    label = label_df_matched[[args.pred_label]]
    
    print("pred label:", label.columns[0])

    # Store input and label data =============================
    X_all = []
    y_all = []

    for index, row in input_df.iterrows():
        # Store input values
        X = row[:].values
        # Store label values
        y = label.loc[index][args.pred_label]
        X_all.append(X)
        y_all.append(y)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    print("Shape of input data:", X_all.shape)
    print("Shape of label data: ", y_all.shape)
    if X_all.shape[0] != input_df.shape[0]:
        raise ValueError("Number of loaded input samples does not match the number of the original one.")
    if y_all.shape[0] != label.shape[0]:
        raise ValueError("Number of loaded label samples does not match the number of the original one.")
    
    # Instantiate the cross-validation object
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        
    def objective(trial):
        # Set hyperparameters ========================================
        # Set initial scale_pos_weight to 0 and overwrite it later
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting': 'gbdt',
            'verbosity': -1,
            'nthread': args.thread_count,
            'random_state': 42,
            'verbose_eval':100,
            'bagging_seed': 11,
            'feature_pre_filter': False,
            'scale_pos_weight': 0,
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 100.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 300.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
            'max_bin': trial.suggest_int('max_bin', 15, 255)
            }
        
        num_stopping_rounds = trial.suggest_int('num_stopping_rounds', 5, 50)
        callbacks=[lgb.early_stopping(stopping_rounds=num_stopping_rounds, verbose=False)]
        #  ===================================================================
        
        # Initialize the list to store the AUROC scores
        auroc_scores = []
        
        # Split the data into training and validation sets
        for train_index, val_index in kf.split(X_all, y_all):
            X_train, X_val = X_all[train_index], X_all[val_index]
            y_train, y_val = y_all[train_index], y_all[val_index]
            
            # Generate data sets
            lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            lgb_eval = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
            
            # get scale_pos_weight
            n_positives = sum(y_train)
            n_negatives = len(y_train) - n_positives
            scale_pos_weight = n_negatives / n_positives
            
            # Overwrite scale_pos_weight
            params['scale_pos_weight'] = scale_pos_weight

            # Train the model
            gbm = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_eval],
                valid_names=['valid'],
                num_boost_round=args.num_boost_rounds,
                callbacks=callbacks)

            # Calculate AUROC for the validation set
            y_pred_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
            fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)

            auc_val= auc(fpr_val, tpr_val)
            auroc_scores.append(auc_val)
            
        # Return the average score
        return np.mean(auroc_scores)
    
    # Run the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    # Save the best hyperparameters to a file
    filename = f'best_params_{args.pred_label}.txt'
    output_file = os.path.join(output_dir, filename)
    with open(output_file, 'w') as file:
        # get the date of the optimization and write it to the file
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Date of analysis: {date}\n")
        file.write(f"input_csv: {args.input_csv}\n")
        file.write(f"label_csv: {args.label_csv}\n")
        file.write(f"n_trials: {args.n_trials}\n")
        file.write(f"label: {args.pred_label}\n")  # Write the label to the file
        file.write(f"Best score: {study.best_value}\n")  # We write the best score to the file
        file.write(f"optimiszed hyperparameters: {study.best_params}\n")  # We write the best hyperparameters to the file
    
    # Re-train the model with the best hyperparameters and save the model and the classification threshold
    def get_models_and_threshold(params, X_all, y_all, output_dir, model_dir):
        # set parameters
        params = {'objective': 'binary', 'metric':'binary_logloss'} | params
        params['verbosity'] = -1
        params['random_state'] = 42
        params['verbose_eval'] = 100
        params['bagging_seed'] = 11
        params['feature_pre_filter'] = False
        params['scale_pos_weight'] = 0
        params['nthread'] = args.thread_count

        # Initialize the cross-validation object
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        # Initialize the list to store the models
        models = []
        # Initialize the list to store the thresholds
        thresholds = []
        # Initialize the list to store the AUROC scores for each fold
        aurocs = []
        
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        
        count = 0
        # Split the data into training and validation sets
        for train_index, val_index in kf.split(X_all, y_all):
            count += 1
            X_train, X_val = X_all[train_index], X_all[val_index]
            y_train, y_val = y_all[train_index], y_all[val_index]
            
            # Generate data sets
            lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            lgb_eval = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
            
            # get scale_pos_weight
            n_positives = sum(y_train)
            n_negatives = len(y_train) - n_positives
            scale_pos_weight = n_negatives / n_positives
            
            # Overwrite scale_pos_weight
            params['scale_pos_weight'] = scale_pos_weight
            
            evals_result = {}

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=['train', 'valid'],
                num_boost_round=10000,
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False), lgb.record_evaluation(evals_result)])
            
            models.append(model)
            
            # Draw learning curve
            ax = axs[(count-1)//5][(count-1)%5]
            ax.set_title(f'Fold {count}')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Loss')
            ax.grid()
            ax.plot(evals_result['train']['binary_logloss'], label='train loss', color='blue')
            ax.plot(evals_result['valid']['binary_logloss'], label='validation loss', color='darkred')
            ax.legend()

            # Perform predictions on the validation data
            y_pred_val = model.predict(X_val)
            
            # Calculate the Youden index threshold
            fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_pred_val)       
            auroc = auc(fpr_val, tpr_val)
            aurocs.append(auroc)
            
            # Determine the best threshold using Youden index
            youden_threshold = thresholds_val[np.argmax(tpr_val - fpr_val)]
            thresholds.append(youden_threshold)

            # Save the model for each fold
            joblib.dump(model, os.path.join(model_dir, f'model_fold_{count}.pkl'))
        
        # Save the learning curve
        plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
        plt.clf()
        plt.close()

        # Calculate the average threshold
        average_threshold_youden = np.mean(thresholds)
        # Calculate the average auroc
        average_auroc = np.mean(aurocs)
        # Save the average threshold
        joblib.dump(average_threshold_youden, os.path.join(model_dir, 'average_threshold.pkl'))
        
        with open(output_file, 'a') as file:
            # get the date of the optimization and write it to the file
            file.write(f"Classificaiton threshold: {average_threshold_youden}\n")

    get_models_and_threshold(study.best_params, X_all, y_all, output_dir, model_dir)

print(f"Hyperparameter optimization done! Models and threshold saved in {model_dir}.")