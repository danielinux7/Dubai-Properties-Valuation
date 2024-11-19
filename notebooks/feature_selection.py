import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from tqdm import tqdm
from typing import List, Tuple
from config import Config


class FeatureSelector:
    """A class to handle feature selection for numerical and categorical variables."""
    
    def __init__(self):
        self.config = Config()
        self.numerical_features = [
            'amount', 'total_buyer', 'total_seller', 'transaction_size_sqm',
            'property_size_sqm', 'rooms_en'
        ]
        self.categorical_features = [
            'transaction_type_en', 'transaction_subtype_en', 'registration_type_en',
            'is_freehold_text', 'property_usage_en', 'property_type_en',
            'property_subtype_en', 'project_name_en', 'area_en',
            'nearest_landmark_en', 'nearest_metro_en', 'nearest_mall_en'
        ]

    def select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to perform feature selection on the dataset.
        
        Args:
            data: Input DataFrame containing all features
            
        Returns:
            DataFrame with selected features only
        """
        if self.config.mode == 'inference':
            return data[self.config.temp['selected_features']]

        print('='*50)
        print('Starting Feature Selection Process')
        print('='*50)
        
        data = data.reset_index(drop=True)
        
        # Step 1: Select numerical features
        selected_numerical = self._select_numerical_features(data)
        
        # Step 2: Select categorical features
        initial_categorical = self._perform_rfe_selection(data)
        final_categorical = self._perform_chi_squared_selection(data, initial_categorical)
        
        # Combine and store selected features
        selected_features = selected_numerical + final_categorical
        
        if self.config.mode == 'train':
            self.config.temp['selected_features'] = selected_features
            
        return data[selected_features + ['amount']]

    def _select_numerical_features(self, data: pd.DataFrame) -> List[str]:
        """
        Select numerical features based on correlation analysis.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of selected numerical feature names
        """
        print('Performing correlation analysis for numerical features...')
        corr_matrix = data[self.numerical_features].corr()
        corr_with_target = corr_matrix['amount'].drop('amount')
        sorted_corr = corr_with_target.sort_values(ascending=False)
        
        selected_features = []
        i = 0
        while True:
            if i > len(sorted_corr)-1:
                break
            feature = sorted_corr.iloc[i:i+1].index[0]
            corr_value = sorted_corr.iloc[i]
            # Check if the feature has a moderate/high correlation with the amount
            if abs(corr_value) > 0.3:
                selected_features.append(feature)
            
                # Remove any features that have a correlation greater than 0.9 with the selected feature
                for other_feature in sorted_corr.index:
                    if other_feature != feature and abs(corr_matrix.loc[feature, other_feature]) > 0.9:
                        sorted_corr = sorted_corr.drop(other_feature)
            i+=1
            
        return selected_features

    def _perform_rfe_selection(self, data: pd.DataFrame) -> List[str]:
        """
        Perform Random Forest Feature Importance and RFE selection.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of selected categorical feature names
        """
        print('Performing Random Forest Feature Importance and RFE...')
        X_all = data.drop(self.numerical_features, axis=1)
        y_all = data['amount']
        selected_features = []
        
        for i in tqdm(range(0, len(X_all.columns), self.config.RFE['n_columns'])):
            # Sample subset of data
            indices = np.random.choice(
                len(X_all), 
                size=self.config.RFE['n_samples'], 
                replace=False
            )
            X = X_all.iloc[indices, i:i + self.config.RFE['n_columns']]
            y = y_all.iloc[indices]
            
            # Train Random Forest and perform RFE
            rf_model = RandomForestRegressor(
                n_estimators=self.config.RFE['RF_n_estimators'],
                random_state=42
            )
            rfe = RFE(
                rf_model,
                n_features_to_select=self.config.RFE['n_features']
            )
            rfe.fit(X, y)
            
            selected_features.extend(list(X.columns[rfe.support_]))
            
        return selected_features

    def _perform_chi_squared_selection(
        self, 
        data: pd.DataFrame, 
        categorical_features: List[str]
    ) -> List[str]:
        """
        Perform Chi-squared test for categorical feature selection.
        
        Args:
            data: Input DataFrame
            categorical_features: List of categorical features to evaluate
            
        Returns:
            List of selected categorical feature names
        """
        print('Performing Chi-squared test for categorical features...')
        X = data[categorical_features]
        y = pd.qcut(data['amount'], q=3)
        
        chi2_results = []
        for column in tqdm(X.columns):
            contingency_table = pd.crosstab(X[column], y)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            chi2_results.append({
                'feature': column,
                'chi2': chi2,
                'p_value': p_value
            })
        
        chi2_df = pd.DataFrame(chi2_results)
        selected_features = (chi2_df
            .query(f'p_value < {self.config.CHI["p_value"]}')
            .nlargest(self.config.CHI['n_features'], 'chi2')
            .feature.tolist()
        )
        
        return selected_features


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper function for backwards compatibility.
    
    Args:
        data: Input DataFrame
        
    Returns:
        DataFrame with selected features only
    """
    selector = FeatureSelector()
    return selector.select_features(data)
