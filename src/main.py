# main.py
import pandas as pd
from preprocessing import load_and_preprocess_data
from linear_regression import train_linear_regression, evaluate_linear_regression
from knn import train_knn, evaluate_knn

def main():
    
    # File paths
    file_path = 'data/Houses.csv'
    save_path_prefix = 'data/'

    # Load and preprocess data
    load_and_preprocess_data(file_path, save_path_prefix)

    # Load preprocessed data
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')['price']
    y_test = pd.read_csv('data/y_test.csv')['price']

    # Train and evaluate Linear Regression
    linear_regression_model = train_linear_regression(X_train, y_train)
    evaluate_linear_regression(linear_regression_model, X_test, y_test, X_train, y_train)

    # Train and evaluate KNN
    knn_model = train_knn(X_train, y_train)
    evaluate_knn(knn_model, X_test, y_test, X_train, y_train)


if __name__ == "__main__":
    main()
