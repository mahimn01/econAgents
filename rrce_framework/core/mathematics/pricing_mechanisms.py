"""
Pricing mechanisms implementation based on Step 6 of mathematical foundation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PricingParameters:
    """Parameters for pricing mechanisms."""
    # Resource Criticality Multiplier parameters
    kappa: float = 2.0  # Intensity parameter
    eta: float = 3.0    # Shape parameter
    
    # System Impact Factor parameters
    sigma: Dict[str, float] = None  # Impact coefficients
    
    # Social Stability Coefficient parameters
    delta: float = 1.0  # Inequality amplification factor
    beta: float = 0.5   # Essential goods factor

class StabilityWeightedPricing:
    """
    Implementation of Stability-Weighted Pricing Framework from Step 6:
    - Resource Criticality Multiplier (RCM)
    - System Impact Factor (SIF)
    - Social Stability Coefficient (SSC)
    """
    
    def __init__(self, pricing_params: Dict[str, PricingParameters]):
        """
        Initialize pricing mechanisms.
        
        Args:
            pricing_params: Dictionary mapping resource names to parameters
        """
        self.pricing_params = pricing_params
        
        logger.info("Initialized Stability-Weighted Pricing Framework")
    
    def compute_rcm(self, resource_name: str, A_i: float, C_i: float) -> float:
        """
        Implementation of Theorem 17: Resource Criticality Multiplier
        RCM_i(t) = 1 + κ_i * exp(η_i * (1 - A_i(t)/C_i(t)))
        
        Args:
            resource_name: Name of resource
            A_i: Current resource availability
            C_i: Critical threshold
            
        Returns:
            Resource Criticality Multiplier
        """
        if resource_name not in self.pricing_params:
            return 1.0
        
        params = self.pricing_params[resource_name]
        
        # Avoid division by zero
        if C_i <= 0:
            return 1.0 + params.kappa
        
        # Calculate RCM from Theorem 17
        ratio = A_i / C_i
        exponent = params.eta * (1 - ratio)
        
        # Cap the exponential to prevent overflow
        exponent = min(exponent, 10.0)
        
        rcm = 1.0 + params.kappa * np.exp(exponent)
        
        # Ensure RCM is always >= 1
        return max(rcm, 1.0)
    
    def compute_sif(self, resource_name: str, system_states: Dict[str, float],
                   reference_states: Dict[str, float]) -> float:
        """
        Implementation of Theorem 18: System Impact Factor
        SIF_{i,k}(t) = 1 + Σ σ_{i,k,m} * (1 - S_m(t)/S_m^ref)
        
        Args:
            resource_name: Name of resource
            system_states: Current system states
            reference_states: Reference system states
            
        Returns:
            System Impact Factor
        """
        if resource_name not in self.pricing_params:
            return 1.0
        
        params = self.pricing_params[resource_name]
        
        if params.sigma is None:
            return 1.0
        
        sif = 1.0
        
        for system_name, current_state in system_states.items():
            if system_name in reference_states and system_name in params.sigma:
                ref_state = reference_states[system_name]
                sigma_coefficient = params.sigma[system_name]
                
                if ref_state > 0:
                    impact = sigma_coefficient * (1 - current_state / ref_state)
                    sif += impact
        
        # Ensure SIF is always >= 0.1 (prevent negative prices)
        return max(sif, 0.1)
    
    def compute_ssc(self, resource_name: str, gini: float, g_max: float,
                   essential_consumption: float, basic_consumption: float) -> float:
        """
        Implementation of Theorem 19: Social Stability Coefficient
        SSC_{i,k}(t) = 1 + δ_{i,k} * Gini(t)/G^max - β_{i,k} * E_k(t)/E_k^basic
        
        Args:
            resource_name: Name of resource
            gini: Current Gini coefficient
            g_max: Maximum sustainable Gini coefficient
            essential_consumption: Current essential consumption
            basic_consumption: Basic needs consumption level
            
        Returns:
            Social Stability Coefficient
        """
        if resource_name not in self.pricing_params:
            return 1.0
        
        params = self.pricing_params[resource_name]
        
        # Inequality amplification term
        inequality_term = 0.0
        if g_max > 0:
            inequality_term = params.delta * (gini / g_max)
        
        # Essential needs term
        essential_term = 0.0
        if basic_consumption > 0:
            essential_term = params.beta * (essential_consumption / basic_consumption)
        
        ssc = 1.0 + inequality_term - essential_term
        
        # Ensure SSC is always >= 0.1 (prevent negative prices)
        return max(ssc, 0.1)
    
    def compute_swpf_price(self, resource_name: str, base_price: float,
                          A_i: float, C_i: float,
                          system_states: Dict[str, float],
                          reference_states: Dict[str, float],
                          gini: float, g_max: float,
                          essential_consumption: float = 1.0,
                          basic_consumption: float = 1.0) -> Dict[str, float]:
        """
        Implementation of Theorem 20: Stability-Weighted Pricing Framework
        p_{i,k}(t) = p_{i,k}^base(t) * RCM_i(t) * SIF_{i,k}(t) * SSC_{i,k}(t)
        
        Args:
            resource_name: Name of resource
            base_price: Base marginal cost price
            A_i: Current resource availability
            C_i: Critical threshold
            system_states: Current system states
            reference_states: Reference system states
            gini: Current Gini coefficient
            g_max: Maximum sustainable Gini coefficient
            essential_consumption: Essential consumption level
            basic_consumption: Basic consumption level
            
        Returns:
            Dictionary with price components and final price
        """
        # Compute all pricing factors
        rcm = self.compute_rcm(resource_name, A_i, C_i)
        sif = self.compute_sif(resource_name, system_states, reference_states)
        ssc = self.compute_ssc(resource_name, gini, g_max, 
                              essential_consumption, basic_consumption)
        
        # Compute final SWPF price
        swpf_price = base_price * rcm * sif * ssc
        
        return {
            'base_price': base_price,
            'rcm': rcm,
            'sif': sif,
            'ssc': ssc,
            'final_price': swpf_price,
            'resource_premium': (rcm - 1) * base_price,
            'system_adjustment': (sif - 1) * base_price * rcm,
            'social_adjustment': (ssc - 1) * base_price * rcm * sif
        }
    
    def batch_compute_prices(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SWPF prices for multiple resources and time periods.
        
        Args:
            price_data: DataFrame with columns for all required inputs
            
        Returns:
            DataFrame with computed pricing factors and final prices
        """
        results = []
        
        for idx, row in price_data.iterrows():
            try:
                resource_name = row.get('resource_name', 'default')
                
                price_result = self.compute_swpf_price(
                    resource_name=resource_name,
                    base_price=row.get('base_price', 1.0),
                    A_i=row.get('resource_availability', 100.0),
                    C_i=row.get('critical_threshold', 50.0),
                    system_states=row.get('system_states', {}),
                    reference_states=row.get('reference_states', {}),
                    gini=row.get('gini', 0.3),
                    g_max=row.get('g_max', 0.6),
                    essential_consumption=row.get('essential_consumption', 1.0),
                    basic_consumption=row.get('basic_consumption', 1.0)
                )
                
                # Add index information
                price_result['index'] = idx
                price_result['resource_name'] = resource_name
                
                results.append(price_result)
                
            except Exception as e:
                logger.warning(f"Failed to compute price for row {idx}: {e}")
                # Add default result
                results.append({
                    'index': idx,
                    'resource_name': resource_name,
                    'base_price': row.get('base_price', 1.0),
                    'rcm': 1.0,
                    'sif': 1.0,
                    'ssc': 1.0,
                    'final_price': row.get('base_price', 1.0),
                    'resource_premium': 0.0,
                    'system_adjustment': 0.0,
                    'social_adjustment': 0.0
                })
        
        result_df = pd.DataFrame(results)
        result_df.set_index('index', inplace=True)
        
        logger.info(f"Computed SWPF prices for {len(result_df)} data points")
        return result_df