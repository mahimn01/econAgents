"""
Vector Autoregression (VAR) model for comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleVARModel:
    """
    Simple Vector Autoregression model for economic forecasting.
    """
    
    def __init__(self, lags: int = 4):
        """
        Initialize VAR model.
        
        Args:
            lags: Number of lags to include
        """
        self.lags = lags
        self.coefficients = None
        self.variable_names = None
        self.is_fitted = False
        
        logger.info(f"Initialized VAR model with {lags} lags")
    
    def calibrate(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate VAR model to historical data."""
        try:
            # Select key variables for VAR
            var_variables = []
            
            # Add GDP growth if available
            if 'GDP' in historical_data.columns:
                gdp_growth = historical_data['GDP'].pct_change().dropna()
                var_variables.append(gdp_growth.rename('gdp_growth'))
            elif 'gdp' in historical_data.columns:
                gdp_growth = historical_data['gdp'].pct_change().dropna()
                var_variables.append(gdp_growth.rename('gdp_growth'))
            
            # Add inflation if available
            if 'inflation' in historical_data.columns:
                var_variables.append(historical_data['inflation'].dropna())
            
            # Add unemployment if available
            if 'unemployment' in historical_data.columns:
                var_variables.append(historical_data['unemployment'].dropna())
            
            if not var_variables:
                raise ValueError("No suitable variables found for VAR model")
            
            # Combine variables
            var_data = pd.concat(var_variables, axis=1).dropna()
            
            if len(var_data) < self.lags + 5:
                raise ValueError("Insufficient data for VAR estimation")
            
            # Estimate VAR coefficients
            self.coefficients, self.variable_names = self._estimate_var(var_data)
            self.is_fitted = True
            
            # Compute fit statistics
            fitted_values = self._compute_fitted_values(var_data)
            residuals = var_data.iloc[self.lags:] - fitted_values
            
            rmse = np.sqrt((residuals ** 2).mean().mean())
            
            return {
                'calibration_success': True,
                'n_variables': len(self.variable_names),
                'n_observations': len(var_data) - self.lags,
                'rmse': rmse,
                'lags': self.lags
            }
            
        except Exception as e:
            logger.error(f"VAR calibration failed: {e}")
            return {
                'calibration_success': False,
                'error': str(e)
            }
    
    def _estimate_var(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Estimate VAR coefficients using OLS."""
        n_vars = data.shape[1]
        n_obs = data.shape[0] - self.lags
        
        # Construct lagged variables
        Y = data.iloc[self.lags:].values  # Dependent variables
        X = np.ones((n_obs, 1))  # Constant term
        
        # Add lagged variables
        for lag in range(1, self.lags + 1):
            lagged_data = data.iloc[self.lags - lag:-lag].values
            X = np.hstack([X, lagged_data])
        
        # OLS estimation: β = (X'X)^(-1)X'Y
        try:
            coefficients = np.linalg.solve(X.T @ X, X.T @ Y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            coefficients = np.linalg.pinv(X.T @ X) @ X.T @ Y
        
        return coefficients, data.columns.tolist()
    
    def _compute_fitted_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute fitted values for VAR model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing predictions")
        
        n_obs = data.shape[0] - self.lags
        fitted = np.zeros((n_obs, len(self.variable_names)))
        
        for i in range(n_obs):
            # Construct X vector for observation i
            x = np.array([1])  # Constant
            
            for lag in range(1, self.lags + 1):
                x = np.hstack([x, data.iloc[self.lags + i - lag].values])
            
            # Predict
            fitted[i] = x @ self.coefficients
        
        return pd.DataFrame(
            fitted, 
            columns=self.variable_names,
            index=data.index[self.lags:]
        )
    
    def predict(self, initial_state: Dict[str, Any], 
               time_horizon: int = 20) -> pd.DataFrame:
        """Generate VAR predictions."""
        if not self.is_fitted:
            logger.warning("VAR model not fitted, using simple trend extrapolation")
            return self._simple_trend_prediction(initial_state, time_horizon)
        
        try:
            # Initialize with last known values (simplified)
            current_values = np.array([0.02, 0.02, 0.05])  # Default: 2% GDP growth, 2% inflation, 5% unemployment
            
            if len(self.variable_names) > 0:
                current_values = current_values[:len(self.variable_names)]
            
            predictions = []
            
            for t in range(time_horizon):
                # Predict next period
                x = np.array([1])  # Constant
                
                # Add lagged values
                for lag in range(1, self.lags + 1):
                    if lag <= len(predictions):
                        x = np.hstack([x, predictions[-lag]])
                    else:
                        x = np.hstack([x, current_values])
                
                # Generate prediction
                if x.shape[0] == self.coefficients.shape[0]:
                    pred = x @ self.coefficients
                else:
                    pred = current_values  # Fallback
                
                predictions.append(pred)
            
            # Convert to DataFrame
            result_df = pd.DataFrame(predictions, columns=self.variable_names)
            result_df.index.name = 'period'
            
            # Convert growth rates back to levels if needed
            if 'gdp_growth' in result_df.columns:
                initial_gdp = initial_state.get('gdp', 1000.0)
                gdp_levels = [initial_gdp]
                
                for growth in result_df['gdp_growth']:
                    gdp_levels.append(gdp_levels[-1] * (1 + growth))
                
                result_df['gdp'] = gdp_levels[1:]  # Exclude initial level
            
            # Add constant values for variables not in VAR
            result_df['gini'] = 0.3  # Assume constant inequality
            result_df['system_health'] = 1.0  # VAR doesn't model constraints
            result_df['equilibrium_status'] = True
            
            return result_df
            
        except Exception as e:
            logger.error(f"VAR prediction failed: {e}")
            return self._simple_trend_prediction(initial_state, time_horizon)
    
    def _simple_trend_prediction(self, initial_state: Dict[str, Any], 
                                time_horizon: int) -> pd.DataFrame:
        """Simple trend extrapolation fallback."""
        results = []
        initial_gdp = initial_state.get('gdp', 1000.0)
        growth_rate = 0.02  # 2% default growth
        
        for t in range(time_horizon):
            gdp = initial_gdp * ((1 + growth_rate) ** t)
            results.append({
                'period': t,
                'gdp': gdp,
                'gdp_growth': growth_rate,
                'gini': 0.3,
                'system_health': 1.0,
                'equilibrium_status': True
            })
        
        return pd.DataFrame(results).set_index('period')