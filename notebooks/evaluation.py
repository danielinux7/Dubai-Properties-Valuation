from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Calculate rmse,r2,mae performance metrics
def calculate_metrics(y_true, y_pred, set_name=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    if set_name:
        print(f"\n{set_name} Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        return {'rmse':rmse,'r2':r2,'mae':mae}
    
# Calculate rmse performance metrics
def calculate_rmse(y_true, y_pred, set_name=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if set_name:
        print(f"\n{set_name} Metrics:")
        print(f"RMSE: {rmse:.4f}")
    else:
        return rmse
        
# Calculate r2 performance metrics
def calculate_r2(y_true, y_pred, set_name=None):
    r2 = r2_score(y_true, y_pred)
    if set_name:
        print(f"\n{set_name} Metrics:")
        print(f"R²: {r2:.4f}")
    else:
        return r2
        
# Calculate mae performance metrics
def calculate_mae(y_true, y_pred, set_name=None):
    mae = mean_absolute_error(y_true, y_pred)
    if set_name:
        print(f"\n{set_name} Metrics:")
        print(f"MAE: {mae:.4f}")
    else:
        return mae
