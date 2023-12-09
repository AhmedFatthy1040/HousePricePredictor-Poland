import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load the dataset from the 'data' folder
    try:
        dataset = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If utf-8 fails, try 'latin1' encoding
        dataset = pd.read_csv(file_path, encoding='latin1')

    # Drop unnecessary columns for now
    dataset = dataset.drop(['id', 'address', 'latitude', 'longitude'], axis=1)

    # Separate features (X) and target variable (y)
    X = dataset.drop('price', axis=1)
    y = dataset['price']

    missing_data = dataset.isnull()
    missing_data_count_per_column = missing_data.sum()

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])] , remainder='passthrough')
    X = ct.fit_transform(X)

    # train_test_split() takes numpy arrays only as arguments
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train_scaled = X_train
    X_test_scaled = X_test
    X_train_scaled[:, 3:] = sc.fit_transform(X_train[:, 3:])
    X_test_scaled[:, 3:] = sc.transform(X_test[:, 3:])

    return X_train_scaled, X_test_scaled, y_train, y_test