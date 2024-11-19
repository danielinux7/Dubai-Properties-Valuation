from pathlib import Path

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    """Project Configuration"""

    def __init__(self):
        # temp meta data
        self.temp = {}
        
        # Train mode
        self.mode = 'train' # 'train', 'inference'
        
        # Trained models
        self.models = {}
        
        # File Paths
        self.repo_root = self._get_repo_root()
        
        # Feature Selection Parameters for Categorical
        self.RFE = {'n_samples':10000,
                    'RF_n_estimators':10,
                    'n_features':2, 
                    'n_columns':50}
        self.CHI = {'n_features':100,
                    'p_value':0.05}
        
        # Bayesian Optimization and Models Parameters
        self.bo = {'init_points':1,'n_iter':5} 
        self.bo_xgb = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 500),
            'min_child_weight': (1, 10),
            'gamma': (0, 1),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.5, 1)
            }
        self.bo_rf = {
            'n_estimators': (10, 50), 
            'max_depth': (5, 50),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 20),
            'max_features': (0.5, 1)
            }
        self.SVR = {'n_samples':10000}
        self.bo_svr = {
            'C': (1, 100), 
            'epsilon': (0.01, 0.5),
            'gamma': (0.001, 0.1)
            }
        self.bo_meta = {
            'units1': (32, 128),
            'units2': (16, 64),
            'dropout1': (0.1, 0.5),
            'dropout2': (0.1, 0.5),
            'learning_rate': (1e-4, 1e-2),
            'batch_size': (8, 128)
            }

        # Inference Test Sample
        self.data_point = [{'transaction_type_en': 'Sales',
                           'transaction_subtype_en': 'Sell - Pre registration',
                           'registration_type_en': 'Off-Plan',
                          'is_freehold_text': 'Free Hold',
                          'property_usage_en': 'Residential',
                          'amount': 0,
                          'total_buyer': 2,
                          'total_seller': 1,
                          'transaction_size_sqm': 66.47,
                          'property_size_sqm': 66.47,
                          'property_type_en': 'Unit',
                          'property_subtype_en': 'Flat',
                          'rooms_en': '1 B/R',
                          'project_name_en': 'THE VYBE',
                          'area_en': 'JUMEIRAH VILLAGE CIRCLE',
                          'nearest_landmark_en': 'Sports City Swimming Academy',
                          'nearest_metro_en': 'Dubai Internet City',
                          'nearest_mall_en': 'Mall of the Emirates'}]
        
    def _get_repo_root(self):
        current = Path.cwd()
        while current != current.parent:
            if (current / 'requirements.txt').exists(): break
            else: current = current.parent
        return current