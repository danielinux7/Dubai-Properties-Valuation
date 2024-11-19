from preprocess import *
from feature_selection import *
from base_models import *
from meta_learner import *

import pytest
import pandas as pd
from config import Config
from evaluation import *

config = Config()

def sample_data():
    config.mode = 'train'
    data = pd.read_csv(config.repo_root / 'data/snp_dld_2024_transactions.csv')
    return data.sample(1000).to_dict()

def test_data_loading_processing():
    data = load_process_data(sample_data())
    config.temp['data'] = data
    assert isinstance(data, pd.DataFrame)
    assert not data.isnull().any().any()
    assert len(data) == 1000

def test_feature_selection():
    config.RFE['n_samples'] = 200
    selected = select_features(config.temp['data'])
    config.temp['data'] = selected
    assert isinstance(selected, pd.DataFrame)
    assert len(selected) == 1000
    assert len(config.temp['selected_features']) == len(selected.columns)-1

def test_base_model_training():
    config.bo['n_iter'] = 0
    config.SVR['n_samples'] = 500
    base_models, test_data = train_base_models(config.temp['data'])
    config.models['base_models'] = base_models
    config.temp['test_data'] = test_data
    assert isinstance(base_models, dict)
    assert len(base_models) == 3
    assert len(test_data['x_test']) == int(len(config.temp['data']) * .1) # Testset should be 10% of data

def test_meta_model_training():
    meta_model = train_meta_learner(config.temp['data'], config.models['base_models'])
    config.models['meta_model'] = meta_model
    assert isinstance(meta_model, dict)
    assert len(meta_model) == 1
    assert len(config.temp['data']) == 1000

def test_models_prediction():
    base_models = config.models['base_models']
    test_data = config.temp['test_data']
    meta_model = config.models['meta_model']
    base_models_preds = predict_base_models(test_data['x_test'], base_models, scale=False)
    calculate_metrics(test_data['y_test'], base_models_preds['xgb'], set_name="Test XGB Model")
    calculate_metrics(test_data['y_test'], base_models_preds['rf'], set_name="Test RF Model")
    calculate_metrics(test_data['y_test'], base_models_preds['svr'], set_name="Test SVR Model")
    meta_preds = predict_with_meta_learner(test_data['x_test'], base_models, meta_model, scale=False)
    calculate_metrics(test_data['y_test'], meta_preds, set_name="Test Meta Model")
    assert len(base_models_preds['xgb']) == len(test_data['y_test'])
    assert len(base_models_preds['rf']) == len(test_data['y_test'])
    assert len(base_models_preds['svr']) == len(test_data['y_test'])
    assert len(meta_preds) == len(test_data['y_test'])

def test_models_inference():
    config.mode = 'inference'
    data = load_process_data(config.data_point)
    selected = select_features(data)
    meta_preds = predict_with_meta_learner(selected, scale=True)
    assert len(meta_preds) == len(selected)
