import argparse
import numpy as np
import logging
import gc
import joblib  # For saving models and hyperparameters
import json
from scipy import sparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score, roc_auc_score,
                             mean_absolute_error, r2_score)
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from my_package.data_preprocessing import load_data, get_common_columns
from my_package.regression_estimators import CustomRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    # Define file paths and columns to load
    eye_data_path = args.eye_data_path
    demo_data_path = args.demo_data_path
    columns_to_load = args.columns_to_load.split(',')

    # Load data
    eye_data, demo_data = load_data(eye_data_path, demo_data_path, columns_to_load, nrows=args.nrows)

    # Preview the loaded data
    logging.info("Eye Data Preview:")
    logging.info("\n" + eye_data.head().to_string())

    logging.info("Demographic Data Preview:")
    logging.info("\n" + demo_data.head().to_string())

    # Identify and log common columns
    common_columns = get_common_columns(eye_data, demo_data)
    logging.info(f"Common Columns: {common_columns}")

    # Separate features and target variable
    X = eye_data.drop(columns=[args.target_label])
    y_class = eye_data[args.target_label]  # For classification
    y_reg = eye_data[args.target_regression] if args.target_regression else None  # For regression (assuming 'Target_Ave' is the regression target)

    # Preprocessing
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])  # Ensure sparse output

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Preprocess the data
    X_transformed = preprocessor.fit_transform(X)

    # Convert sparse matrix to dense
    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    # Check for negative values and apply Min-Max Scaling if necessary
    if np.any(X_transformed < 0):
        logging.info("Negative values found in feature matrix X. Applying Min-Max Scaling.")
        scaler = MinMaxScaler()
        X_transformed = scaler.fit_transform(X_transformed)

    # Train-test split for classification
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_transformed, y_class, test_size=0.2, random_state=42)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_class, y_train_class = smote.fit_resample(X_train_class, y_train_class)

    # Train-test split for regression (if applicable)
    if y_reg is not None:
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_transformed, y_reg, test_size=0.2, random_state=42)

    # Classification models and their parameter grids
    classification_models = {
        'Logistic Regression': {
            'model': LogisticRegression(),
            'param_grid': {
                'penalty': ['l2'],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs']
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        },
        'Gaussian Naive Bayes': {
            'model': GaussianNB(),
            'param_grid': {}
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }

    # Regression models
    regression_models = {
        'Linear Regression': CustomRegressor(model=LinearRegression()),
        'K-Nearest Neighbors': CustomRegressor(model=KNeighborsRegressor()),
        'Random Forest': CustomRegressor(model=RandomForestRegressor()),
        'Gradient Boosting': CustomRegressor(model=GradientBoostingRegressor()),
        'AdaBoost': CustomRegressor(model=AdaBoostRegressor()),
        'Multi-Layer Perceptron': CustomRegressor(model=MLPRegressor(max_iter=500))
    }

    # Functions to calculate metrics
    def calculate_classification_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tnr = tn / (tn + fp)  # True Negative Rate
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) == 2 else None
        return precision, recall, f1, roc_auc, tnr

    def calculate_regression_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mrae_val = np.mean(np.abs((y_true - y_pred) / np.mean(y_true)))
        return mae, mrae_val, r2

    # Initialize result dictionaries
    results = {
        'classification': {},
        'regression': {}
    }

    # Hyperparameter tuning and evaluation for classification models
    for clf_name, clf_info in classification_models.items():
        try:
            logging.info(f"Starting hyperparameter tuning for {clf_name}...")
            
            grid_search = GridSearchCV(estimator=clf_info['model'],
                                       param_grid=clf_info['param_grid'],
                                       cv=5,
                                       scoring='accuracy',
                                       verbose=1,
                                       n_jobs=-1)
            
            # Fit the GridSearchCV
            grid_search.fit(X_train_class, y_train_class)
            
            # Save the best hyperparameters
            best_params = grid_search.best_params_
            joblib.dump(best_params, f"{clf_name}_best_params.pkl")
            logging.info(f"Best Hyperparameters for {clf_name}: {best_params}")
            
            # Apply the best model found by GridSearchCV
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_class)
            
            # Calculate and save metrics
            precision, recall, f1, roc_auc, tnr = calculate_classification_metrics(y_test_class, y_pred)
            results['classification'][clf_name] = {
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC-AUC': roc_auc,
                'TNR': tnr,
                'Classification Report': classification_report(y_test_class, y_pred, output_dict=True),
                'Confusion Matrix': confusion_matrix(y_test_class, y_pred).tolist()
            }
            
            # Save the best model
            joblib.dump(best_model, f"{clf_name}_best_model.pkl")
            
        except Exception as e:
            logging.error(f"Error with {clf_name}: {e}")
            results['classification'][clf_name] = {'Error': str(e)}

    # Hyperparameter tuning and evaluation for regression models
    if y_reg is not None:
        for reg_name, custom_model in regression_models.items():
            try:
                logging.info(f"Starting training for {reg_name}...")
                
                # Fit the regression model
                custom_model.fit(X_train_reg, y_train_reg)
                
                # Predict and evaluate
                y_pred = custom_model.predict(X_test_reg)
                mae, mrae, r2 = calculate_regression_metrics(y_test_reg, y_pred)
                
                # Save metrics
                results['regression'][reg_name] = {
                    'MAE': mae,
                    'MRAE': mrae,
                    'R2 Score': r2
                }
                
                # Save the model
                joblib.dump(custom_model, f"{reg_name}_best_model.pkl")
                
            except Exception as e:
                logging.error(f"Error with {reg_name}: {e}")
                results['regression'][reg_name] = {'Error': str(e)}

    # Write results to JSON file
    results_path = args.results_path
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Free up memory
    del eye_data
    del demo_data
    gc.collect()

    logging.info("Processing completed and results saved to JSON")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classification and regression models with hyperparameter tuning.")
    
    parser.add_argument('--eye_data_path', type=str, required=True, help='Path to the eye movement data CSV file')
    parser.add_argument('--demo_data_path', type=str, required=True, help='Path to the demographic data CSV file')
    parser.add_argument('--columns_to_load', type=str, required=True, help='Comma-separated list of columns to load from the CSV files')
    parser.add_argument('--target_label', type=str, required=True, help='Name of the target column for classification')
    parser.add_argument('--target_regression', type=str, help='Name of the target column for regression (optional)')
    parser.add_argument('--nrows', type=int, default=10000, help='Number of rows to load from the CSV files')
    parser.add_argument('--results_path', type=str, required=True, help='Path to save the results JSON file')
    
    args = parser.parse_args()
    main(args)
