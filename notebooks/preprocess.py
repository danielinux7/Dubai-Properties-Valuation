import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
from config import Config

class DataPreprocessor:
    """Handles data preprocessing for real estate transaction data."""
    
    def __init__(self, config: Config):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.numerical_features = [
            'amount', 'total_buyer', 'total_seller',
            'transaction_size_sqm', 'property_size_sqm', 'rooms_en'
        ]
        self.categorical_features = [
            'transaction_type_en', 'transaction_subtype_en',
            'registration_type_en', 'is_freehold_text',
            'property_usage_en', 'property_type_en',
            'property_subtype_en', 'project_name_en', 'area_en',
            'nearest_landmark_en', 'nearest_metro_en', 'nearest_mall_en'
        ]
        self.rooms_mapping = {
            '1 B/R': 1, '2 B/R': 2, '3 B/R': 3, '4 B/R': 4,
            '5 B/R': 5, '6 B/R': 6, '7 B/R': 7, 'Studio': 0.5,
            'Single Room': 0, 'NA': -1, 'Office': -1,
            'Shop': -1, 'Hotel': 100, 'PENTHOUSE': -1
        }

    def load_data(self, data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Load data either from a file or provided dictionary.
        
        Args:
            data: Optional dictionary containing data to process
            
        Returns:
            Loaded DataFrame with selected features
        """
        required_columns = self.numerical_features + self.categorical_features
        
        if data is None:
            file_path = self.config.repo_root / 'data/snp_dld_2024_transactions.csv'
            return pd.read_csv(file_path, usecols=required_columns)
        
        df = pd.DataFrame(data)
        return df[required_columns]

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        # Fill missing values in categorical columns
        df.loc[:, 'property_subtype_en'] = df.property_subtype_en.fillna(df.property_type_en)
        df.loc[:, self.categorical_features] = df[self.categorical_features].fillna('NA')
        
        # Fill missing values in numerical columns
        df.loc[:, 'transaction_size_sqm'] = df.transaction_size_sqm.fillna(df.property_size_sqm)
        df.loc[:, 'rooms_en'] = df.rooms_en.fillna('NA')
        
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Map room values to numerical representations
        self.config.temp['preprocess_rooms'] = self.rooms_mapping
        df['rooms_en'] = df['rooms_en'].map(self.rooms_mapping)
        
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        df[self.categorical_features] = df[self.categorical_features].astype('category')
        
        if self.config.mode == 'train':
            self.config.temp['preprocess_categories'] = df.iloc[0:0]
        elif self.config.mode == 'inference':
            df_empty = self.config.temp['preprocess_categories'].copy()
            for col in self.categorical_features:
                df[col] = df[col].cat.set_categories(df_empty[col].cat.categories)
        
        return pd.get_dummies(df, columns=self.categorical_features)

    def scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using log transformation and standardization.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled numerical features
        """
        df = df.copy()
        
        # Log transformation
        df[self.numerical_features] = np.log1p(df[self.numerical_features] + 1e-15)
        
        # Standardization (excluding 'amount' feature)
        features_to_scale = self.numerical_features[1:]
        
        if self.config.mode == 'train':
            mean = df[features_to_scale].mean(axis=0)
            std = df[features_to_scale].std(axis=0)
            self.config.temp['preprocess_mean'] = mean
            self.config.temp['preprocess_std'] = std
        else:
            mean = self.config.temp['preprocess_mean']
            std = self.config.temp['preprocess_std']
            
        df[features_to_scale] = (df[features_to_scale] - mean) / std
        return df

    def process(self, data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            data: Optional dictionary containing data to process
            
        Returns:
            Fully processed DataFrame
        """
        print('='*50)
        print('Starting data preprocessing pipeline...')
        
        df = self.load_data(data)
        print('Data loaded successfully')
        
        df = self.handle_missing_values(df)
        print('Missing values handled')
        
        df = self.engineer_features(df)
        print('Feature engineering completed')
        
        df = self.encode_categorical_features(df)
        print('Categorical features encoded')
        
        df = self.scale_numerical_features(df)
        print('Numerical features scaled')
        
        print('Preprocessing pipeline completed')
        print('='*50)
        
        return df

def load_process_data(data: Optional[Dict] = None) -> pd.DataFrame:
    """
    Main function to load and process data.
    
    Args:
        data: Optional dictionary containing data to process
        
    Returns:
        Processed DataFrame ready for modeling
    """
    preprocessor = DataPreprocessor(Config())
    return preprocessor.process(data)
