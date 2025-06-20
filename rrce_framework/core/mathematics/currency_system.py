"""
Currency system implementation based on Step 7 of mathematical foundation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CurrencyParameters:
    """Parameters for currency system."""
    # Base elasticity parameters
    gamma_0: float = 0.1  # Base monetary elasticity
    gamma_1: float = 0.2  # Resource stress elasticity
    gamma_2: float = 0.3  # Speculation elasticity
    
    # Sustainability parameters
    alpha_max: float = 0.1  # Maximum speculative excess
    epsilon: float = 2.0    # Sustainability elasticity
    
    # Social parameters
    beta: float = 2.0  # Social coordination exponent

class CurrencySystem:
    """
    Implementation of Common-Wealth Units (CWU) from Step 7:
    - Wealth Creation Index (WCI)
    - Dynamic Elasticity Coefficient
    - Multi-dimensional productive capacity
    """
    
    def __init__(self, currency_params: CurrencyParameters):
        """
        Initialize currency system.
        
        Args:
            currency_params: Currency system parameters
        """
        self.params = currency_params
        
        logger.info("Initialized Common-Wealth Units (CWU) currency system")
    
    def compute_wci(self, gdp: float, speculative_excess: float = 0.0,
                   social_factor: float = 1.0) -> float:
        """
        Implementation of Theorem 24: Wealth Creation Index
        WCI(t) = GDP(t) / (1+α)^ε * Φ_social(t)
        
        Args:
            gdp: Gross Domestic Product
            speculative_excess: Measure of speculative excess (α)
            social_factor: Social coordination factor
            
        Returns:
            Wealth Creation Index
        """
        # Sustainability adjustment
        alpha = min(speculative_excess, self.params.alpha_max)
        sustainability_factor = (1 + alpha) ** (-self.params.epsilon)
        
        # Wealth Creation Index
        wci = gdp * sustainability_factor * social_factor
        
        return max(wci, 0.0)  # Ensure non-negative
    
    def compute_gamma(self, resource_factor: float, productive_capacity: float,
                     total_claims: float) -> float:
        """
        Implementation of Theorem 25: Dynamic Elasticity Coefficient
        γ(t) = γ_0 + γ_1 * (1 - Ψ_resource(t)) + γ_2 * (V(t)/(PC(t)*α) - 1)^+
        
        Args:
            resource_factor: Resource availability factor (Ψ_resource)
            productive_capacity: Total productive capacity
            total_claims: Total value claims in the economy
            
        Returns:
            Dynamic elasticity coefficient
        """
        gamma = self.params.gamma_0
        
        # Resource stress effect
        resource_stress = 1 - resource_factor
        gamma += self.params.gamma_1 * resource_stress
        
        # Speculative excess effect
        if productive_capacity > 0:
            leverage_ratio = total_claims / productive_capacity
            excess = max(0, leverage_ratio - 1.0)
            gamma += self.params.gamma_2 * excess
        
        return max(gamma, 0.01)  # Minimum elasticity
    
    def compute_social_factor(self, gini: float, g_max: float = 0.6) -> float:
        """
        Compute social coordination factor Φ_social(t).
        
        Args:
            gini: Current Gini coefficient
            g_max: Maximum sustainable Gini coefficient
            
        Returns:
            Social coordination factor
        """
        if g_max <= 0:
            return 1.0
        
        # Social coordination decreases with inequality
        # Φ_social(t) = 1 - (Gini(t)/G^max)^β
        normalized_gini = min(gini / g_max, 1.0)
        social_factor = 1.0 - (normalized_gini ** self.params.beta)
        
        return max(social_factor, 0.1)  # Minimum coordination level
    
    def compute_resource_factor(self, resource_levels: Dict[str, float],
                               critical_thresholds: Dict[str, float],
                               weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute resource availability factor Ψ_resource(t).
        
        Args:
            resource_levels: Current resource availability levels
            critical_thresholds: Critical thresholds for each resource
            weights: Importance weights for each resource
            
        Returns:
            Resource availability factor
        """
        if not resource_levels or not critical_thresholds:
            return 1.0
        
        # Default equal weights
        if weights is None:
            weights = {name: 1.0 for name in resource_levels.keys()}
        
        # Compute weighted average of resource ratios
        total_weight = 0.0
        weighted_sum = 0.0
        
        for resource_name, level in resource_levels.items():
            if resource_name in critical_thresholds and resource_name in weights:
                threshold = critical_thresholds[resource_name]
                weight = weights[resource_name]
                
                if threshold > 0:
                    ratio = min(level / threshold, 2.0)  # Cap at 2x threshold
                    weighted_sum += weight * ratio
                    total_weight += weight
        
        if total_weight > 0:
            resource_factor = weighted_sum / total_weight
        else:
            resource_factor = 1.0
        
        return max(resource_factor, 0.01)  # Minimum resource factor
    
    def compute_cwu(self, gdp: float, gini: float,
                   resource_levels: Dict[str, float],
                   critical_thresholds: Dict[str, float],
                   productive_capacity: float = None,
                   total_claims: float = None,
                   speculative_excess: float = 0.0,
                   g_max: float = 0.6,
                   resource_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Implementation of Theorem 26: Common-Wealth Units (CWU) Derivation
        CWU(t) = WCI(t)/(1 + γ(t)) * Φ_social(t) * Ψ_resource(t)
        
        Args:
            gdp: Gross Domestic Product
            gini: Gini coefficient
            resource_levels: Current resource availability levels
            critical_thresholds: Critical thresholds for resources
            productive_capacity: Total productive capacity
            total_claims: Total value claims
            speculative_excess: Speculative excess parameter
            g_max: Maximum sustainable Gini coefficient
            resource_weights: Importance weights for resources
            
        Returns:
            Dictionary with CWU components and final value
        """
        # Compute social coordination factor
        social_factor = self.compute_social_factor(gini, g_max)
        
        # Compute resource availability factor
        resource_factor = self.compute_resource_factor(
            resource_levels, critical_thresholds, resource_weights
        )
        
        # Compute Wealth Creation Index
        wci = self.compute_wci(gdp, speculative_excess, social_factor)
        
        # Compute dynamic elasticity coefficient
        if productive_capacity is None:
            productive_capacity = gdp  # Default assumption
        if total_claims is None:
            total_claims = gdp  # Default assumption
        
        gamma = self.compute_gamma(resource_factor, productive_capacity, total_claims)
        
        # Compute final CWU value
        cwu_base = wci / (1 + gamma)
        cwu_final = cwu_base * social_factor * resource_factor
        
        return {
            'cwu_value': cwu_final,
            'wci': wci,
            'gamma': gamma,
            'social_factor': social_factor,
            'resource_factor': resource_factor,
            'gdp': gdp,
            'gini': gini,
            'productive_capacity': productive_capacity,
            'sustainability_ratio': cwu_final / gdp if gdp > 0 else 0.0
        }
    
    def simulate_currency_trajectory(self, economic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate CWU trajectory over time given economic data.
        
        Args:
            economic_data: DataFrame with required economic indicators
            
        Returns:
            DataFrame with CWU trajectory and components
        """
        results = []
        
        for idx, row in economic_data.iterrows():
            try:
                # Extract data with defaults
                gdp = row.get('GDP', row.get('gdp', 1000.0))
                gini = row.get('gini_coefficient', row.get('gini', 0.3))
                
                # Resource data (simplified)
                resource_levels = {
                    'energy': row.get('energy_use_per_capita', 100.0),
                    'land': row.get('arable_land_per_capita', 50.0),
                }
                
                critical_thresholds = {
                    'energy': 50.0,  # Default threshold
                    'land': 25.0,    # Default threshold
                }
                
                # Compute CWU
                cwu_result = self.compute_cwu(
                    gdp=gdp,
                    gini=gini,
                    resource_levels=resource_levels,
                    critical_thresholds=critical_thresholds,
                    speculative_excess=row.get('speculative_excess', 0.0)
                )
                
                # Add time information
                cwu_result['date'] = idx
                cwu_result['time_index'] = idx
                
                results.append(cwu_result)
                
            except Exception as e:
                logger.warning(f"Failed to compute CWU for time {idx}: {e}")
                # Add default result
                results.append({
                    'date': idx,
                    'time_index': idx,
                    'cwu_value': 1.0,
                    'wci': 1000.0,
                    'gamma': 0.1,
                    'social_factor': 1.0,
                    'resource_factor': 1.0,
                    'gdp': row.get('GDP', row.get('gdp', 1000.0)),
                    'gini': row.get('gini_coefficient', row.get('gini', 0.3)),
                    'productive_capacity': 1000.0,
                    'sustainability_ratio': 1.0
                })
        
        result_df = pd.DataFrame(results)
        if 'date' in result_df.columns:
            result_df.set_index('date', inplace=True)
        
        logger.info(f"Simulated CWU trajectory for {len(result_df)} time periods")
        return result_df
