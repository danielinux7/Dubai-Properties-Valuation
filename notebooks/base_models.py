import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib, random
from evaluation import *
from config import Config
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

def predict_base_models(data, base_models=None, scale=True):
    config = Config()
    if base_models is None:
        base_models = config.models['base_models']
        
    # Make predictions using the base models
    xgb_pred = base_models['xgb'].predict(data)
    rf_pred = base_models['rf'].predict(data)
    svr_pred = base_models['svr'].predict(data)

    if scale:
        xgb_pred = np.expm1(xgb_pred)
        rf_pred = np.expm1(rf_pred)
        svr_pred = np.expm1(svr_pred)

    return {'xgb':xgb_pred, 'rf':rf_pred, 'svr':svr_pred}
    
def train_base_models(data):
    print('*'*50)
    print('Training Base Models...')

    config = Config()

    # Generate sample dataset (replace this with your actual data)
    X = data.drop('amount',axis=1)
    y = data['amount']

    # First split: separate test set (10% of data)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Second split: separate train and validation sets (90% train, 10% validation of remaining data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # - XGBoost
    # - Use Bayesian Optimization for hyperparameter tuning

    # Define the objective function for Bayesian optimization
    def xgb_train_eval(
        max_depth,
        learning_rate, 
        n_estimators,
        min_child_weight,
        gamma,
        subsample,
        colsample_bytree
    ):
        # Convert hyperparameters to expected types
        max_depth = int(max_depth)
        n_estimators = int(n_estimators)
        
        # Initialize and train XGBoost model
        xgb_model = XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=10,
            random_state=42
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Evaluate model on validation set
        val_pred = xgb_model.predict(X_val)
        rmse = calculate_rmse(y_val, val_pred)
        
        # Return the negative RMSE as the objective to maximize
        return -rmse

    # Run Bayesian optimization
    xgb_bo = BayesianOptimization(
        xgb_train_eval,
        config.bo_xgb,
        random_state=42
    )

    print('Bayesian Optimization for XGB Model...')
    # Optimize the hyperparameters
    xgb_bo.maximize(init_points=config.bo['init_points'], n_iter=config.bo['n_iter'])

    # Get the best hyperparameters
    best_params = xgb_bo.max['params']

    # Train the final model with the best hyperparameters
    xgb_model = XGBRegressor(
        max_depth=int(best_params['max_depth']),
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        min_child_weight=best_params['min_child_weight'],
        gamma=best_params['gamma'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        random_state=42
    )

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # - Random Forest
    # - Use Bayesian Optimization for hyperparameter tuning

    def rf_train_eval(
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features
        ):
        # Convert hyperparameters to expected types
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)
        min_samples_split = int(min_samples_split)
        min_samples_leaf = int(min_samples_leaf)

        # Initialize and train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        rf_model.fit(X_train, y_train)

        # Evaluate model on validation set
        val_pred = rf_model.predict(X_val)
        rmse = calculate_rmse(y_val, val_pred)

        # Return the negative RMSE as the objective to maximize
        return -rmse

    # Run Bayesian optimization
    rf_bo = BayesianOptimization(
        rf_train_eval,
        config.bo_rf,
        random_state=42
    )

    print('Bayesian Optimization for RF Model...')
    # Optimize the hyperparameters
    rf_bo.maximize(init_points=config.bo['init_points'], n_iter=config.bo['n_iter'])

    # Get the best hyperparameters
    best_params = rf_bo.max['params']

    rf_model = RandomForestRegressor(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        min_samples_split=int(best_params['min_samples_split']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        max_features=best_params['max_features'],
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    # - Support Vector Regression
    # - Use Bayesian Optimization for hyperparameter tuning
    def svr_train_eval(
        C,
        epsilon,
        gamma
        ):
        indeces = random.sample(range(len(X_train)), config.SVR['n_samples'])
        X = X_train.loc[indeces]
        y = y_train[indeces]
        # Initialize and train Support Vector Regression model
        svr_model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')
        svr_model.fit(X, y)

        # Evaluate model on validation set
        val_pred = svr_model.predict(X_val)
        rmse = calculate_rmse(y_val, val_pred)

        # Return the negative RMSE as the objective to maximize
        return -rmse

    svr_bo = BayesianOptimization(
        svr_train_eval,
        config.bo_svr,
        random_state=42
    )

    print('Bayesian Optimization for SVR Model...')
    # Optimize the hyperparameters
    svr_bo.maximize(init_points=config.bo['init_points'], n_iter=config.bo['n_iter'])

    # Get the best hyperparameters
    best_params = svr_bo.max['params']

    svr_model = SVR(
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        gamma=best_params['gamma'],
        kernel='rbf'
    )

    random.seed(42)
    indeces = random.sample(range(len(X_train)), config.SVR['n_samples'])
    X = X_train.loc[indeces]
    y = y_train[indeces]
    svr_model.fit(X, y)
    
    return {'xgb':xgb_model, 'rf':rf_model, 'svr':svr_model}, {'x_test':X_test, 'y_test':y_test}

