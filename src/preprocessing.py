import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, save_path=None):
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
    X = np.array(ct.fit_transform(X))
    X = np.delete(X, 3, axis=1)

    # train_test_split() takes numpy arrays only as arguments
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # sc = StandardScaler()
    # X_train_scaled = X_train
    # X_test_scaled = X_test
    # X_train_scaled[:, 3:] = sc.fit_transform(X_train[:, 3:])
    # X_test_scaled[:, 3:] = sc.transform(X_test[:, 3:])

    
    column_names = ["Kraków_City", "Warszawa_City", "Poznañ_City", "floor", "rooms", "sq", "year"]

    # Convert back to DataFrame for easy visualization (optional)
    X_train_df = pd.DataFrame(X_train, columns=column_names).apply(lambda x: x.map('{:.2f}'.format))
    X_test_df = pd.DataFrame(X_test, columns=column_names).apply(lambda x: x.map('{:.2f}'.format))


    # Save the preprocessed data to a new CSV file
    if save_path:
        preprocessed_data = pd.concat([X_train_df, X_test_df], ignore_index=True)
        preprocessed_data.to_csv(save_path, index=False)

    return X_train, X_test, y_train, y_test


file_path = 'data/Houses.csv'

save_path = 'data/preprocessed_data.csv'

X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, save_path)