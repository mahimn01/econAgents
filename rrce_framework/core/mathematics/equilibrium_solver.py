"""
Equilibrium solver implementation based on Step 5 of mathematical foundation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import fsolve, minimize
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EquilibriumParameters:
    """Parameters for equilibrium conditions."""
    # Resource balance parameters
    resource_tolerance: float = 0.01  # Tolerance for resource balance
    
    # Productive capacity parameters
    max_leverage_ratio: float = 1.5  # Maximum α parameter
    capacity_tolerance: float = 0.05  # Tolerance for capacity constraint
    
    # Social stability parameters
    max_gini: float = 0.6  # Maximum sustainable Gini coefficient
    social_tolerance: float = 0.02  # Tolerance for social constraint

class EquilibriumSolver:
    """
    Implementation of system equilibrium from Step 5:
    - Resource balance constraint
    - Productive capacity constraint
    - Social stability constraint
    """
    
    def __init__(self, equilibrium_params: EquilibriumParameters):
        """
        Initialize equilibrium solver.
        
        Args:
            equilibrium_params: Equilibrium parameters
        """
        self.params = equilibrium_params
        
        logger.info("Initialized equilibrium solver")
    
    def check_resource_balance(self, resource_flows: Dict[str, float],
                              resource_levels: Dict[str, float],
                              critical_thresholds: Dict[str, float]) -> Dict[str, bool]:
        """
        Check Theorem 12: Physical Resource Balance Constraint
        dA_i/dt >= 0 for sustainability
        
        Args:
            resource_flows: Current resource flow rates
            resource_levels: Current resource levels
            critical_thresholds: Critical thresholds
            
        Returns:
            Dictionary indicating which resources satisfy balance constraint
        """
        balance_status = {}
        
        for resource_name in resource_levels.keys():
            flow_rate = resource_flows.get(resource_name, 0.0)
            current_level = resource_levels.get(resource_name, 0.0)
            critical_level = critical_thresholds.get(resource_name, 0.0)
            
            # Resource balance criteria:
            # 1. Flow rate is non-negative (dA/dt >= 0)
            # 2. Current level is above critical threshold
            
            flow_sustainable = flow_rate >= -self.params.resource_tolerance
            level_sustainable = current_level >= critical_level
            
            balance_status[resource_name] = flow_sustainable and level_sustainable
        
        return balance_status
    
    def check_capacity_constraint(self, total_claims: float,
                                 productive_capacity: float) -> bool:
        """
        Check Theorem 13: Productive Capacity Constraint
        V(t) <= α * PC(t)
        
        Args:
            total_claims: Total value claims in economy
            productive_capacity: Total productive capacity
            
        Returns:
            True if constraint is satisfied
        """
        if productive_capacity <= 0:
            return False
        
        # Maximum allowed claims
        max_claims = self.params.max_leverage_ratio * productive_capacity
        
        # Check if current claims are within limit
        capacity_satisfied = total_claims <= max_claims + self.params.capacity_tolerance
        
        return capacity_satisfied
    
    def check_social_stability(self, gini_coefficient: float) -> bool:
        """
        Check Theorem 14: Social Stability Constraint
        Gini(t) <= G^max
        
        Args:
            gini_coefficient: Current Gini coefficient
            
        Returns:
            True if constraint is satisfied
        """
        # Check if inequality is within sustainable bounds
        max_allowed_gini = self.params.max_gini + self.params.social_tolerance
        
        social_stable = gini_coefficient <= max_allowed_gini
        
        return social_stable
    
    def assess_system_equilibrium(self, system_state: Dict[str, any]) -> Dict[str, any]:
        """
        Implementation of Theorem 15: System Equilibrium
        Check all three equilibrium conditions simultaneously.
        
        Args:
            system_state: Complete system state dictionary
            
        Returns:
            Dictionary with equilibrium assessment
        """
        # Extract required data with defaults
        resource_flows = system_state.get('resource_flows', {})
        resource_levels = system_state.get('resource_levels', {})
        critical_thresholds = system_state.get('critical_thresholds', {})
        total_claims = system_state.get('total_claims', 1000.0)
        productive_capacity = system_state.get('productive_capacity', 1000.0)
        gini_coefficient = system_state.get('gini_coefficient', 0.3)
        
        # Check individual constraints
        resource_balance = self.check_resource_balance(
            resource_flows, resource_levels, critical_thresholds
        )
        
        capacity_constraint = self.check_capacity_constraint(
            total_claims, productive_capacity
        )
        
        social_stability = self.check_social_stability(gini_coefficient)
        
        # Overall equilibrium status
        all_resources_balanced = all(resource_balance.values()) if resource_balance else True
        overall_equilibrium = all_resources_balanced and capacity_constraint and social_stability
        
        # Compute stability metrics
        resource_stability_ratio = sum(resource_balance.values()) / len(resource_balance) if resource_balance else 1.0
        
        capacity_utilization = total_claims / (productive_capacity * self.params.max_leverage_ratio) if productive_capacity > 0 else 0.0
        
        social_stress = gini_coefficient / self.params.max_gini if self.params.max_gini > 0 else 0.0
        
        return {
            'overall_equilibrium': overall_equilibrium,
            'resource_balance': resource_balance,
            'capacity_constraint_satisfied': capacity_constraint,
            'social_stability_satisfied': social_stability,
            'resource_stability_ratio': resource_stability_ratio,
            'capacity_utilization': capacity_utilization,
            'social_stress_ratio': social_stress,
            'constraint_violations': {
                'resource': not all_resources_balanced,
                'capacity': not capacity_constraint,
                'social': not social_stability
            },
            'system_health_score': (
                resource_stability_ratio * 0.4 +
                (1.0 if capacity_constraint else 0.0) * 0.3 +
                (1.0 if social_stability else 0.0) * 0.3
            )
        }
    
    def find_equilibrium_prices(self, base_prices: Dict[str, float],
                               system_state: Dict[str, any],
                               max_iterations: int = 100) -> Dict[str, float]:
        """
        Find equilibrium prices that satisfy all constraints.
        
        Args:
            base_prices: Initial/base prices for goods
            system_state: Current system state
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with equilibrium prices
        """
        def equilibrium_conditions(price_vector):
            """Define equilibrium conditions as optimization target."""
            prices = {name: price for name, price in zip(base_prices.keys(), price_vector)}
            
            # Update system state with new prices
            updated_state = system_state.copy()
            updated_state['prices'] = prices
            
            # Assess equilibrium
            equilibrium_status = self.assess_system_equilibrium(updated_state)
            
            # Create penalty for constraint violations
            penalty = 0.0
            
            if not equilibrium_status['capacity_constraint_satisfied']:
                penalty += (equilibrium_status['capacity_utilization'] - 1.0) ** 2
            
            if not equilibrium_status['social_stability_satisfied']:
                penalty += (equilibrium_status['social_stress_ratio'] - 1.0) ** 2
            
            resource_penalty = (1.0 - equilibrium_status['resource_stability_ratio']) ** 2
            penalty += resource_penalty
            
            return penalty
        
        # Initial price vector
        initial_prices = np.array(list(base_prices.values()))
        
        try:
            # Find equilibrium prices using optimization
            result = minimize(
                equilibrium_conditions,
                initial_prices,
                method='L-BFGS-B',
                bounds=[(0.01, 1000.0) for _ in initial_prices],  # Positive prices
                options={'maxiter': max_iterations}
            )
            
            if result.success:
                equilibrium_prices = {
                    name: price for name, price in zip(base_prices.keys(), result.x)
                }
                logger.info("Found equilibrium prices successfully")
            else:
                logger.warning("Equilibrium price optimization did not converge")
                equilibrium_prices = base_prices
                
        except Exception as e:
            logger.error(f"Equilibrium price optimization failed: {e}")
            equilibrium_prices = base_prices
        
        return equilibrium_prices
    
    def simulate_equilibrium_trajectory(self, initial_state: Dict[str, any],
                                       time_points: np.ndarray,
                                       external_shocks: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Simulate equilibrium trajectory over time.
        
        Args:
            initial_state: Initial system state
            time_points: Time points for simulation
            external_shocks: List of external shock events
            
        Returns:
            DataFrame with equilibrium status over time
        """
        results = []
        current_state = initial_state.copy()
        
        for t in time_points:
            try:
                # Apply external shocks if any
                if external_shocks:
                    for shock in external_shocks:
                        if shock.get('time') == t:
                            for key, value in shock.get('changes', {}).items():
                                current_state[key] = value
                
                # Assess equilibrium at current time
                equilibrium_status = self.assess_system_equilibrium(current_state)
                
                # Add time information
                equilibrium_status['time'] = t
                equilibrium_status['time_index'] = t
                
                results.append(equilibrium_status)
                
                # Simple state evolution (placeholder)
                # In a full implementation, this would be more sophisticated
                current_state['gini_coefficient'] *= 0.999  # Slight improvement over time
                
            except Exception as e:
                logger.warning(f"Equilibrium assessment failed at time {t}: {e}")
                results.append({
                    'time': t,
                    'overall_equilibrium': False,
                    'system_health_score': 0.5
                })
        
        result_df = pd.DataFrame(results)
        if 'time' in result_df.columns:
            result_df.set_index('time', inplace=True)
        
        logger.info(f"Simulated equilibrium trajectory for {len(result_df)} time periods")
        return result_df