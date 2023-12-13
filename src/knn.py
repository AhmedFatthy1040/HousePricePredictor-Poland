from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import pandas as pd

def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def evaluate_knn(model, X_test, y_test, X_train, y_train):
    # Evaluate on the test set
    mse = mean_squared_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))

    # Cross-validation scores
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    print("KNN Mean Squared Error on Test Set:", mse)
    print("KNN R2 Score on Test Set:", r2)
    print("Cross-Validation Scores for KNN:", cross_val_scores)

    # Save the trained model
    save_model(model, 'models/knn_model.pkl')

def save_model(model, model_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)