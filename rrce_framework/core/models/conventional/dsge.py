"""
Simple DSGE model for comparison with RRCE.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class SimpleDSGEModel:
    """
    Simplified Dynamic Stochastic General Equilibrium model for comparison.
    
    This implements a basic RBC-style model with:
    - Cobb-Douglas production function
    - CRRA utility
    - Capital accumulation
    - Technology shocks
    """
    
    def __init__(self, parameters: Optional[Dict[str, float]] = None):
        """Initialize DSGE model with parameters."""
        # Default parameters
        self.params = {
            'alpha': 0.33,      # Capital share
            'beta': 0.96,       # Discount factor
            'delta': 0.1,       # Depreciation rate
            'sigma': 2.0,       # Risk aversion
            'rho_z': 0.95,      # Technology persistence
            'sigma_z': 0.007,   # Technology shock std
            'steady_state_growth': 0.02  # Long-run growth rate
        }
        
        if parameters:
            self.params.update(parameters)
        
        logger.info("Initialized simple DSGE model")
    
    def calibrate(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate model to historical data."""
        try:
            # Simple calibration - match GDP growth
            gdp_data = historical_data.get('GDP', historical_data.get('gdp'))
            
            if gdp_data is not None:
                growth_rate = gdp_data.pct_change().mean()
                self.params['steady_state_growth'] = max(min(growth_rate, 0.1), -0.05)
            
            # Match volatility
            if gdp_data is not None:
                gdp_volatility = gdp_data.pct_change().std()
                self.params['sigma_z'] = max(min(gdp_volatility, 0.05), 0.001)
            
            return {
                'calibrated_growth': self.params['steady_state_growth'],
                'calibrated_volatility': self.params['sigma_z'],
                'calibration_success': True
            }
            
        except Exception as e:
            logger.warning(f"DSGE calibration failed: {e}")
            return {'calibration_success': False, 'error': str(e)}
    
    def predict(self, initial_state: Dict[str, Any], 
               time_horizon: int = 20) -> pd.DataFrame:
        """Generate DSGE predictions."""
        try:
            # Initialize variables
            periods = range(time_horizon)
            results = []
            
            # Initial values
            current_gdp = initial_state.get('gdp', 1000.0)
            current_capital = current_gdp / 0.33  # Approximate K/Y ratio
            current_tfp = 1.0
            
            for t in periods:
                # Technology shock
                shock = np.random.normal(0, self.params['sigma_z'])
                current_tfp = (current_tfp ** self.params['rho_z']) * np.exp(shock)
                
                # Production function: Y = A * K^α * L^(1-α)
                labor = 1.0  # Normalized
                output = current_tfp * (current_capital ** self.params['alpha']) * (labor ** (1 - self.params['alpha']))
                
                # Capital accumulation: K' = (1-δ)K + I
                investment = 0.2 * output  # Simple investment rule
                current_capital = (1 - self.params['delta']) * current_capital + investment
                
                # Consumption
                consumption = output - investment
                
                # Apply steady-state growth
                growth_factor = (1 + self.params['steady_state_growth']) ** t
                
                results.append({
                    'period': t,
                    'gdp': output * growth_factor,
                    'consumption': consumption * growth_factor,
                    'investment': investment * growth_factor,
                    'capital': current_capital * growth_factor,
                    'tfp': current_tfp,
                    'gini': 0.3,  # Assume constant inequality
                    'system_health': 1.0,  # DSGE doesn't model system constraints
                    'equilibrium_status': True,  # Always in equilibrium
                })
            
            return pd.DataFrame(results).set_index('period')
            
        except Exception as e:
            logger.error(f"DSGE prediction failed: {e}")
            # Return simple growth projection
            results = []
            initial_gdp = initial_state.get('gdp', 1000.0)
            
            for t in range(time_horizon):
                gdp = initial_gdp * ((1 + self.params['steady_state_growth']) ** t)
                results.append({
                    'period': t,
                    'gdp': gdp,
                    'gini': 0.3,
                    'system_health': 1.0,
                    'equilibrium_status': True
                })
            
            return pd.DataFrame(results).set_index('period')