import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, save_path_prefix=''):
    try:
        dataset = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        dataset = pd.read_csv(file_path, encoding='latin1')

    # Drop unnecessary columns for now
    dataset = dataset.drop(['id', 'address', 'latitude', 'longitude'], axis=1)

    # Separate features (X) and target variable (y)
    X = dataset.drop('price', axis=1)
    y = dataset['price']

    missing_data = dataset.isnull()
    missing_data_count_per_column = missing_data.sum()

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    X = np.delete(X, 3, axis=1)

    # train_test_split() takes numpy arrays only as arguments
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert back to DataFrame for easy visualization
    column_names = ["Kraków_City", "Warszawa_City", "Poznañ_City", "floor", "rooms", "sq", "year"]
    X_train_df = pd.DataFrame(X_train, columns=column_names).apply(lambda x: x.map('{:.2f}'.format))
    X_test_df = pd.DataFrame(X_test, columns=column_names).apply(lambda x: x.map('{:.2f}'.format))

    # Save the preprocessed data to separate CSV files
    X_train_df.to_csv(f'{save_path_prefix}X_train.csv', index=False)
    X_test_df.to_csv(f'{save_path_prefix}X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['price']).to_csv(f'{save_path_prefix}y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['price']).to_csv(f'{save_path_prefix}y_test.csv', index=False)

file_path = 'data/Houses.csv'
save_path_prefix = 'data/'

load_and_preprocess_data(file_path, save_path_prefix)
