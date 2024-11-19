#!/usr/bin/env python3
"""
Main orchestration script for a machine learning pipeline that implements stacked modeling
for house price prediction. Supports training, inference, and testing modes.

The pipeline includes:
- Data preprocessing
- Feature selection
- Base model training (XGBoost, Random Forest, SVR)
- Meta-learner training and prediction
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Union

import joblib
import pytest
import tensorflow as tf
from preprocess import load_process_data
from feature_selection import select_features
from base_models import train_base_models, predict_base_models
from meta_learner import train_meta_learner, predict_with_meta_learner
from evaluation import calculate_metrics
from config import Config

class Pipeline:
    def __init__(self, config: Config):
        """Initialize the ML pipeline with configuration."""
        self.config = config
        self.models: Dict[str, Any] = {}
    
    def save_data(self, name: str, data) -> None:
        """Save processed data to CSV file.
        
        Args:
            name: Output filename
            data: DataFrame to save
        """
        output_path = self.config.repo_root / 'data' / name
        data.to_csv(output_path, index=False)
        print(f'Data saved to {output_path}')

    def save_models(self, models: Dict[str, Any]) -> None:
        """Save trained models to disk.
        
        Args:
            models: Dictionary of model names and their corresponding objects
        """
        for name, model in models.items():
            if name == 'meta_model':
                path = self.config.repo_root / "models/meta_model.keras"
                model.save(path)
            else:
                path = self.config.repo_root / f"models/{name}_model.joblib"
                joblib.dump(model, path)
        print('Models saved successfully')

    def load_models(self) -> None:
        """Load trained models from disk."""
        try:
            self.config.models['base_models'] = {
                'xgb': joblib.load(self.config.repo_root / "models/xgb_model.joblib"),
                'rf': joblib.load(self.config.repo_root / "models/rf_model.joblib"),
                'svr': joblib.load(self.config.repo_root / "models/svr_model.joblib")
            }
            self.config.models['meta_model'] = {
                'meta_model': tf.keras.models.load_model(
                    self.config.repo_root / "models/meta_model.keras"
                )
            }
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def save_metadata(self) -> None:
        """Save pipeline metadata."""
        with open(self.config.repo_root / "data/meta_data.pkl", 'wb') as f:
            pickle.dump(self.config.temp, f)

    def load_metadata(self) -> None:
        """Load pipeline metadata."""
        try:
            with open(self.config.repo_root / "data/meta_data.pkl", 'rb') as f:
                self.config.temp = pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError("Metadata file not found")

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration JSON file
        """
        try:
            with open(f'{self.config.repo_root / "data"}/{config_path}', 'r') as file:
                conf = json.load(file)

            # Convert lists to tuples in configuration
            def list2tuple(data: Dict) -> Dict:
                return {
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in data.items()
                }

            # Update configuration attributes
            self.config.RFE = conf['RFE']
            self.config.CHI = conf['CHI']
            self.config.bo = conf['bo']
            self.config.bo_xgb = list2tuple(conf['bo_xgb'])
            self.config.bo_rf = list2tuple(conf['bo_rf'])
            self.config.SVR = conf['SVR']
            self.config.bo_svr = list2tuple(conf['bo_svr'])
            self.config.bo_meta = list2tuple(conf['bo_meta'])
            print('Configuration loaded successfully')
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load configuration: {str(e)}")

    def train(self) -> None:
        """Execute the training pipeline."""
        # Load and preprocess data
        data = load_process_data()
        data = select_features(data)
        self.save_data('processed_data.csv', data)

        # Train models
        base_models, test_data = train_base_models(data)
        meta_model = train_meta_learner(data, base_models)
        self.save_models({**base_models, **meta_model})
        self.save_metadata()

        # Evaluate models
        base_models_preds = predict_base_models(
            test_data['x_test'], base_models, scale=False
        )
        for model_name, predictions in base_models_preds.items():
            calculate_metrics(
                test_data['y_test'], 
                predictions, 
                set_name=f"Test {model_name.upper()} Model"
            )

        meta_preds = predict_with_meta_learner(
            test_data['x_test'], 
            base_models, 
            meta_model, 
            scale=False
        )
        calculate_metrics(test_data['y_test'], meta_preds, set_name="Test Meta Model")

    def inference(self, json_file: str) -> None:
        """Run inference on new data.
        
        Args:
            json_file: Path to JSON file containing inference data
        """
        try:
            self.load_models()
            self.load_metadata()
            
            json_path = self.config.repo_root / "data" / json_file
            with open(json_path, 'r') as file:
                data = json.load(file)
                
            actual_amounts = data[-1]['actual_amounts']
            processed_data = load_process_data(data[:-1])
            processed_data = select_features(processed_data)
            
            meta_preds = predict_with_meta_learner(processed_data, scale=True)
            
            # Print predictions with actual values
            for i, (pred, actual) in enumerate(zip(meta_preds, actual_amounts)):
                print(f'House {i+1}: Predicted=${pred.squeeze():.2f} | Actual=${actual}')
                
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")

    def test(self) -> None:
        """Run tests."""
        pytest.main([self.config.repo_root / 'notebooks/test.py'])

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Machine Learning Pipeline for House Price Prediction'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'inference', 'test'],
        default='inference',
        help='Operation mode: train|inference|test'
    )
    parser.add_argument(
        '--json_file',
        default='sample.json',
        help='Path to the JSON file for inference'
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to the JSON config file for training or testing'
    )
    return parser.parse_args()

def main() -> None:
    """Main entry point for the pipeline."""
    args = parse_arguments()
    config = Config()
    pipeline = Pipeline(config)
    
    try:
        config.mode = args.mode
        if args.config:
            pipeline.load_config(args.config)

        if args.mode == 'train':
            pipeline.train()
        elif args.mode == 'inference':
            pipeline.inference(args.json_file)
        elif args.mode == 'test':
            pipeline.test()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
