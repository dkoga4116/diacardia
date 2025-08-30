import pandas as pd
import argparse

def select_features(input_csv, feature_list_csv):
    # load the feature list
    feature_df = pd.read_csv(feature_list_csv)
    feature_list = feature_df.iloc[:, 0].tolist()

    # load the input CSV
    df = pd.read_csv(input_csv)

    # always keep the ecg_id column
    columns_to_keep = ['ecg_id'] if 'ecg_id' in df.columns else []

    # add only the columns that exist in the input CSV
    existing_features = [f for f in feature_list if f in df.columns]
    columns_to_keep.extend(existing_features)

    # extract the columns
    df_selected = df[columns_to_keep]
    df_selected.set_index('ecg_id', inplace=True)
    print(f'Number of features in df_selected: {len(df_selected.columns)}')

    return df_selected


def fill_and_normalize(input_df, median_mean_std_csv):
    # load the median, mean, and standard deviation
    df_median_mean_std_12lead = pd.read_csv(median_mean_std_csv, index_col=0, header=0)

    # fill the missing values with the median of the training data and standard scaling
    def fill_and_standard_scaling(df_test, df_median_mean_std):
        # fill the missing values with the median of the training data
        df_test_filled = df_test.fillna(df_median_mean_std.loc['median'])
        print("Null values in test data",df_test.isnull().sum().sum(), '->', df_test_filled.isnull().sum().sum()) 
        
        # get the mean and standard deviation of the training data
        mean = df_median_mean_std.loc['mean']
        std = df_median_mean_std.loc['std']

        # standard scaling
        df_test_filled_std = df_test_filled.copy()
        for col in df_test_filled.columns:
            if col == 'ecg_id': continue
            else:
                df_test_filled_std[col] = (df_test_filled[col] - mean[col]) / std[col]
        return df_test_filled_std

    df_features_12lead_filled_std = fill_and_standard_scaling(input_df, df_median_mean_std_12lead)
    
    return df_features_12lead_filled_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specified features from a CSV file and normalize the features")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--feature_list_csv", required=True, help="Path to the CSV file of feature list (feature_list_269_12-lead.csv or feature_list_28_1-lead.csv)")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file")
    parser.add_argument("--median_mean_std_csv", required=True, help="Path to the CSV file of median, mean, and standard deviation")

    args = parser.parse_args()

    # select features
    df_selected = select_features(args.input_csv, args.feature_list_csv)

    # fill the missing values with the median of the training data and standard scaling
    df_features_12lead_filled_std = fill_and_normalize(df_selected, args.median_mean_std_csv)

    # save the output CSV file
    df_features_12lead_filled_std.to_csv(args.output_csv, index=True, header=True)
    
    print(f"Extracted features and normalized features are saved to {args.output_csv}.")


