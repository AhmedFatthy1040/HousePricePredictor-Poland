# House Price Predictor - Poland

## Overview

The House Price Predictor for Poland is a machine learning project designed to predict house prices based on various features. This repository contains the source code, data files, and trained models used in the project.

## Project Structure

- **data/:** Contains the raw dataset (`Houses.csv`) and preprocessed data files (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`).
  
- **models/:** Holds the trained models saved as pickle files (`linear_regression_model.pkl`, `knn_model.pkl`).

- **src/:** Contains the source code for data preprocessing, model training, and evaluation (`preprocessing.py`, `linear_regression.py`, `knn.py`, `main.py`).

- **notebooks/:** Includes Jupyter notebooks for Exploratory Data Analysis (`EDA.ipynb`) and modeling (`Modeling.ipynb`).

## Usage

1. **Dependencies:**
   - Install project dependencies using `pip install -r requirements.txt`.

2. **Run the Project:**
   - Execute `src/main.py` to run the entire pipeline.

## Models

1. **Linear Regression:**
   - Trained using scikit-learn's `LinearRegression`.
   - Model saved as `models/linear_regression_model.pkl`.
   - Evaluation metrics include Mean Squared Error, R2 Score, and Cross-Validation Scores.

2. **K-Nearest Neighbors (KNN):**
   - Trained using scikit-learn's `KNeighborsRegressor`.
   - Model saved as `models/knn_model.pkl`.
   - Evaluation metrics include Mean Squared Error, R2 Score, and Cross-Validation Scores.

## Future Improvements

- Hyperparameter Tuning: Experiment with different hyperparameter configurations for improved model performance.
  
- Feature Engineering: Explore additional features or transformations to enhance model predictions.

- Visualizations: Enhance visualizations for better interpretation of results.

## Contribution

Feel free to contribute to the project by opening issues, suggesting improvements, or submitting pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
