import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, matthews_corrcoef
import argparse
from warnings import warn

# Make predictions using the saved models and the classification threshold ================================
def load_models_and_threshold(model_dir):
    # Load the saved models
    models = []
    for i in range(1, 11):
        model_path = os.path.join(model_dir, f'model_fold_{i}.pkl')
        model = joblib.load(model_path)
        models.append(model)
    
    # Load the classification threshold
    threshold_path = os.path.join(model_dir, 'average_threshold.pkl')
    threshold = joblib.load(threshold_path)
    
    return models, threshold

# Make predictions and save the results ================================
def make_predictions_and_save_results(models, threshold, X_test, input_df_test, output_dir):
    
    # Make predictions
    y_pred_test = np.mean([model.predict(X_test) for model in models], axis=0)
    y_pred_test_class = [1 if y_pred >= threshold else 0 for y_pred in y_pred_test]
    
    # Save the predictions ================================
    output_csv = os.path.join(output_dir, 'predictions.csv')
    df_pred = pd.DataFrame({
        'ecg_id': input_df_test.index,
        'pred': y_pred_test,
        'pred_class': y_pred_test_class
    })
    df_pred.to_csv(output_csv, sep=',', header=True, index=False, encoding='cp932')
    print("Predictions saved to:", output_csv)
    
    return y_pred_test, y_pred_test_class

# Visualize the predictions ================================
def visualize_predictions(models, threshold, X_test, y_test, y_pred_test, y_pred_test_class, output_dir, input_df_test):
    
    # Write the classification report to a .txt file ------------------------------------------------
    pred_results_file = os.path.join(output_dir, f"test_results.txt")
    with open(pred_results_file, "w") as file:
        # Performance evaluation for the test data
        fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
        auc_test = auc(fpr_test, tpr_test)
        file.write("\nTest results\n")
        file.write(f"AUROC: {auc_test}\n")
        file.write(f"Threshold: {threshold}\n")
        file.write(f"{confusion_matrix(y_test, y_pred_test_class)}\n")
        file.write(f"{classification_report(y_test, y_pred_test_class)}\n")
        file.write(f"MCC_test: {matthews_corrcoef(y_test, y_pred_test_class)}\n")
    print("Classification report have been saved to:", pred_results_file)
    
    # Plot the ROC curve for test data ------------------------------------------------
    fig = plt.figure(figsize=(8, 7))
    auc_score = auc(fpr_test, tpr_test)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.plot(fpr_test, tpr_test, label=f'Train ROC (area = {auc_score:.3f})', linewidth=2, color='darkred')
    plt.fill_between(fpr_test, tpr_test, alpha=0.1, color='grey')
    # plt.legend()
    plt.title('ROC')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    # Increase the font size of the tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_test.png'), dpi=300)
    plt.clf()
    plt.close()
    print("ROC curve for test data have been saved to:", os.path.join(output_dir, 'roc_test.png'))
    
    # Calculate SHAP values and make a plot ------------------------------------------------
    print("Calculating SHAP values. This may take a while...")
    # Initialize the list to store SHAP values
    shap_values_all = []

    # Get SHAP values for each model
    for model in models:
        # Initialize the explainer
        explainer = shap.TreeExplainer(model)
        # Calculate SHAP values
        # shap_values = explainer.shap_values(X_test)
        shap_values = explainer.shap_values(X_test)
        # Add SHAP values to the list
        shap_values_all.append(shap_values)
    # Calculate the mean SHAP values (When using LightGBM, shap values are calculated for each class)
    mean_shap_values = np.mean(shap_values_all, axis=0) # (folds, n_samples, n_features) -> (n_samples, n_features)

    ## Calculate the importance of SHAP values ================================================================
    # Get the names of features
    col_names = input_df_test.columns
    # Calculate the mean absolute value of SHAP values for each feature
    shap_importance = np.abs(mean_shap_values).mean(axis=0) # (n_features,)
    # Convert SHAP values to percentage
    shap_importance_percent = (shap_importance / shap_importance.sum()) * 100
    # Create a dataframe and save it to a CSV file
    shap_df = pd.DataFrame(shap_importance_percent, index=col_names, columns=['importance(%)'])
    shap_df = shap_df.sort_values(by='importance(%)', ascending=False)
    shap_df.to_csv(os.path.join(output_dir, 'shap_importance.csv'), index=True, header=True)
    print("SHAP importance have been saved to:", os.path.join(output_dir, 'shap_importance.csv'))
    
    ## Create a plot of SHAP values ================================================================
    # Create and save a plot of SHAP values
    shap.summary_plot(mean_shap_values, X_test, feature_names=col_names, show=False, max_display=20)
    plt.title("SHAP summary plot")
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300)
    plt.close()
    print("SHAP values have been saved to:", os.path.join(output_dir, 'shap_summaryplot.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model_dir', type=str, required=True, help='path to the directory containing the saved models')
    parser.add_argument('--input_test_csv', type=str, required=True, help='path to the test data CSV file')
    parser.add_argument('--label_test_csv', type=str, required=False, default=None, help='path to the label CSV file')
    parser.add_argument('--label_name', type=str, required=False, default=None, help='name of the label column')
    parser.add_argument('--output_dir', type=str, required=True, help='path to the output directory')
    parser.add_argument('--evaluation', action='store_true', help='whether to perform evaluation or not')
    args, unk = parser.parse_known_args()
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")     
        
    # Set the output directory ================================
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the evaluation flag ================================
    evaluation = args.evaluation
    print("Evaluation:", evaluation)
    
    # Set the model directory and the input CSV file ================================
    model_dir = args.model_dir
    input_test_csv = args.input_test_csv
    label_test_csv = args.label_test_csv
    label = args.label_name
    
    # Load the test data ================================
    input_test = pd.read_csv(input_test_csv, sep=',', header=0, index_col='ecg_id', encoding='cp932')
    input_test.sort_index(inplace=True)
    ## Load the label data if evaluation is True ------------------------------
    if evaluation == True:
        label_test = pd.read_csv(label_test_csv, sep=',', header=0, index_col='ecg_id', encoding='cp932')
        label_test.sort_index(inplace=True)
        label_test_matched = label_test[label_test.index.isin(input_test.index)].copy()
        
        # Store test data =======================================
        print("Loading test data...")
        X_test = []
        y_test = []

        for index, row in input_test.iterrows():
            # Store input values
            X = row[:].values
            # Store target values
            y = label_test_matched.loc[index, label]
            X_test.append(X)
            y_test.append(y)

        X_test = np.array(X_test)
        y_test = np.array(y_test)
    ## Skip the label data if evaluation is False ------------------------------
    elif evaluation == False:
        X_test = input_test.values
        
    # Perform test ================================
    ## Get input and label data ------------------------------
    models, threshold = load_models_and_threshold(model_dir)
    ## Gat predictions and save the results ------------------------------
    y_pred_test, y_pred_test_class = make_predictions_and_save_results(models, threshold, X_test, input_test, output_dir)
    
    ## Visualize the predictions ------------------------------
    if evaluation == True:
        visualize_predictions(models,threshold, X_test, y_test, y_pred_test, y_pred_test_class, output_dir, input_test)
    else:
        print("Evaluation is skipped.")
    
    print("Test done!")