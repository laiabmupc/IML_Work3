import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def preprocess_data(dataset, plot = False):
    """
    Parameters:
        dataset (str): Dataset name ('hypothyroid', 'hepatitis', 'heart-statlog')
        plot (bool): If True, shows histograms for numeric features with NaNs

    Preprocessing steps:
    1. Load the ARFF file.
    2. Apply MinMax scaling to numerical features.
    3. Handle missing values (mean and median for numeric values and 'missing' for categorical features).
    4. Apply OneHot Encoding to categorical features.
    5. Save preprocessed file into csv.
    """
    # Obtain the path, load the data and store it into a dataframe
    data_path = os.path.join('.', 'data', dataset+'.arff')
    data, _ = arff.loadarff(data_path)
    df = pd.DataFrame(data)

    # Convert byte strings to normal strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Extract the names of numerical and categorical features
    if dataset!='heart-statlog':
        # Remove the Class for hepatitis and hypothyroid
        class_label = df['Class']
        df.drop(columns=['Class'], inplace=True)
    else:
        # Remove the class for heart-statlog
        class_label = df['class']
        df.drop(columns=['class'], inplace=True)

    features_num = df.select_dtypes(include=[np.number]).columns.to_numpy()
    features_cat = df.select_dtypes(exclude=[np.number]).columns.to_numpy()

    # MinMax scalers
    for var in features_num:
        scaler = MinMaxScaler().fit(df[[var]])
        df[var] = scaler.transform(df[[var]])

        # Treat the NaNs
        if df[var].isna().sum() != 0:
            if dataset=='hypothyroid':
                df[var] = df[var].fillna(df[var].mean())
            elif dataset=='hepatitis':
                df[var] = df[var].fillna(df[var].mean())

    # One Hot Encoding
    for var in features_cat:
        # Treat the NaNs
        df[[var]] = df[[var]].replace({'?': "missing"})
        df[[var]] = df[[var]].replace({np.nan: "missing"})
        df[[var]] = df[[var]].fillna('missing')

        # Create the OneHotEncoder
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df[[var]])
        encoded = enc.transform(df[[var]])
        encoded_df = pd.DataFrame(encoded,
                                  columns=[f"{var}_{cat}" for cat in enc.categories_[0]],
                                  index=df.index)
        # Remove the original column once the encoding has been performed
        df = df.drop(columns=[var])
        df = pd.concat([df, encoded_df], axis=1)

    class_df = pd.DataFrame()
    class_df['Class'] = class_label

    save_path = os.path.join('.', 'data', dataset + '.preprocessed.csv')
    save_path_class = os.path.join('.', 'data', dataset + '_class' + '.preprocessed.csv')

    # Store the preprocessed data and the class in a separate file
    df.to_csv(save_path, index=False)
    class_df.to_csv(save_path_class, index=False)


def load_data(dataset):
    dataset_path = os.path.join('.', 'data', dataset + '.preprocessed.csv')
    save_path_class = os.path.join('.', 'data', dataset + '_class' + '.preprocessed.csv')

    df = pd.read_csv(dataset_path)
    class_df = pd.read_csv(save_path_class)
    return df, class_df

