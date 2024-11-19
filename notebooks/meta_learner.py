import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from config import Config
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K, gc
from bayes_opt import BayesianOptimization

def predict_with_meta_learner(data, base_models=None, meta_model=None, scale=True):
    config = Config()

    if base_models is None:
        base_models = config.models['base_models']
        
    if meta_model is None:
        meta_model = config.models['meta_model']

    # Make predictions using the base models
    xgb_pred = base_models['xgb'].predict(data)
    rf_pred = base_models['rf'].predict(data)
    svr_pred = base_models['svr'].predict(data)
    
    # Make final predictions using the meta-learner
    base_models_pred = pd.DataFrame({'xgb': xgb_pred, 'rf': rf_pred, 'svr': svr_pred})
    
    # StandardScaler
    mean = config.temp['meta_mean']
    std = config.temp['meta_std']
    base_models_pred = (base_models_pred - mean) / std
    meta_pred = meta_model['meta_model'].predict(base_models_pred)

    if scale:
        meta_pred = np.expm1(meta_pred)

    return meta_pred

def train_meta_learner(data, base_models):
    print('*'*50)
    print('Training Meta Model...')
    config = Config()
    xgb_model, rf_model, svr_model = base_models['xgb'], base_models['rf'], base_models['svr']
    
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

    # Make predictions on all sets
    print('XGB predictions...')
    train_pred_xgb = xgb_model.predict(X_train)
    val_pred_xgb = xgb_model.predict(X_val)
    test_pred_xgb = xgb_model.predict(X_test)

    print('RF predictions...')
    train_pred_rf = rf_model.predict(X_train)
    val_pred_rf = rf_model.predict(X_val)
    test_pred_rf = rf_model.predict(X_test)

    print('SVR predictions...')
    train_pred_svr = []
    for i in tqdm(range(0, len(X_train), 10000)):
        train_pred_svr.extend(svr_model.predict(X_train.iloc[i:i+10000]))
    train_pred_svr = np.array(train_pred_svr)
    val_pred_svr = svr_model.predict(X_val)
    test_pred_svr = svr_model.predict(X_test)

    X_train = np.concatenate((train_pred_xgb.reshape(-1,1), 
                        train_pred_rf.reshape(-1,1),
                        train_pred_svr.reshape(-1,1)), axis=-1)

    X_val = np.concatenate((val_pred_xgb.reshape(-1,1), 
                        val_pred_rf.reshape(-1,1),
                        val_pred_svr.reshape(-1,1)), axis=-1)

    X_test = np.concatenate((test_pred_xgb.reshape(-1,1), 
                        test_pred_rf.reshape(-1,1),
                        test_pred_svr.reshape(-1,1)), axis=-1)

    # StandardScaler
    mean = .9 *(.9 *X_train.mean(axis=0) + .1 *X_val.mean(axis=0)) + .1*X_test.mean(axis=0)
    std = .9 *(.9 *X_train.std(axis=0) + .1 *X_val.std(axis=0)) + .1*X_test.std(axis=0)
    config.temp['meta_mean'] = mean
    config.temp['meta_std'] = std
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    # Define the objective function for Bayesian optimization
    def meta_model_train_eval(
        units1,
        units2,
        dropout1,
        dropout2,
        learning_rate,
        batch_size
        ):
        # Convert hyperparameters to expected types
        units1 = int(units1)
        units2 = int(units2)
        dropout1 = float(dropout1)
        dropout2 = float(dropout2)
        batch_size = int(batch_size)
        # Initialize and train the meta learner model
        meta_model = Sequential()
        meta_model.add(Input(shape=(X_train.shape[1],)))
        meta_model.add(Dense(units1, activation='relu'))
        meta_model.add(Dropout(dropout1))
        meta_model.add(Dense(units2, activation='relu'))
        meta_model.add(Dropout(dropout2))
        meta_model.add(Dense(1))
        
        meta_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        meta_model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_val, y_val), 
                       callbacks=[early_stopping], verbose=0)
        
        # Evaluate the model on the validation set
        val_loss = meta_model.evaluate(X_val, y_val, verbose=0)
        
        del meta_model
        gc.collect()
        
        # Return the negative validation loss as the objective to maximize
        return -np.sqrt(val_loss)

    # Run Bayesian optimization
    meta_bo = BayesianOptimization(
        meta_model_train_eval,
        config.bo_meta,
        random_state=42
    )

    print('Bayesian Optimization for Meta Model...')
    meta_bo.maximize(init_points=config.bo['init_points'], n_iter=config.bo['n_iter'])

    # Get the best hyperparameters
    best_params = meta_bo.max['params']

    # Train the final meta learner model with the best hyperparameters
    meta_model = Sequential()
    meta_model.add(Input(shape=(X_train.shape[1],)))
    meta_model.add(Dense(int(best_params['units1']), activation='relu'))
    meta_model.add(Dropout(best_params['dropout1']))
    meta_model.add(Dense(int(best_params['units2']), activation='relu'))
    meta_model.add(Dropout(best_params['dropout2']))
    meta_model.add(Dense(1))

    meta_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    meta_model.fit(X_train, y_train, epochs=50, batch_size=int(best_params['batch_size']), 
                         validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

    return {'meta_model':meta_model}

