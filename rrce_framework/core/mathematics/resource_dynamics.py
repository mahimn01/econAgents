"""
Resource dynamics implementation based on Steps 1-3 of mathematical foundation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy.integrate import odeint
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ResourceParameters:
    """Parameters for resource dynamics."""
    # Resource conservation parameters
    regeneration_rate: float = 0.05
    degradation_rate: float = 0.02
    extraction_efficiency: float = 0.8
    
    # Carrying capacity parameters
    carrying_capacity: float = 1000.0
    growth_rate: float = 0.1
    critical_threshold: float = 100.0
    
    # Thermodynamic parameters
    energy_content: float = 1.0  # Energy per unit resource
    conversion_efficiency: float = 0.3  # Thermodynamic efficiency
    
    # Environmental parameters
    environmental_impact: float = 0.1  # Impact per unit extraction

class ResourceDynamics:
    """
    Implementation of resource dynamics from Steps 1-3:
    - Conservation of mass-energy (Step 1)
    - Thermodynamic constraints (Step 2) 
    - Ecological carrying capacity (Step 3)
    """
    
    def __init__(self, resource_params: Dict[str, ResourceParameters]):
        """
        Initialize resource dynamics.
        
        Args:
            resource_params: Dictionary mapping resource names to parameters
        """
        self.resource_params = resource_params
        self.resource_names = list(resource_params.keys())
        
        logger.info(f"Initialized resource dynamics for {len(self.resource_names)} resources")
    
    def conservation_equation(self, A: np.ndarray, t: float, 
                            inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        """
        Implementation of Theorem 1: Resource Balance Constraint
        dA_i/dt = I_i(t) - O_i(t) - D_i(t)
        
        Args:
            A: Resource availability vector
            t: Time
            inputs: Input rates vector
            outputs: Output rates vector
            
        Returns:
            Derivative vector dA/dt
        """
        n_resources = len(A)
        dA_dt = np.zeros(n_resources)
        
        for i, resource_name in enumerate(self.resource_names):
            params = self.resource_params[resource_name]
            
            # Conservation equation components
            I_i = inputs[i]  # Input rate
            O_i = outputs[i]  # Output rate
            D_i = params.degradation_rate * A[i]  # Degradation rate
            
            # Apply conservation constraint
            dA_dt[i] = I_i - O_i - D_i
        
        return dA_dt
    
    def logistic_growth(self, A: np.ndarray, t: float) -> np.ndarray:
        """
        Implementation of Theorem 5: Logistic Carrying Capacity
        dA_i/dt = r_i * A_i * (1 - A_i/K_i) - H_i
        
        Args:
            A: Resource availability vector
            t: Time
            
        Returns:
            Growth rate vector
        """
        n_resources = len(A)
        growth = np.zeros(n_resources)
        
        for i, resource_name in enumerate(self.resource_names):
            params = self.resource_params[resource_name]
            
            # Logistic growth with carrying capacity
            r_i = params.growth_rate
            K_i = params.carrying_capacity
            A_i = max(A[i], 0.01)  # Avoid division by zero
            
            growth[i] = r_i * A_i * (1 - A_i / K_i)
        
        return growth
    
    def thermodynamic_efficiency(self, resource_name: str, 
                                T_hot: float = 400.0, T_cold: float = 300.0) -> float:
        """
        Implementation of Theorem 2: Thermodynamic Production Constraint
        η_max = 1 - T_cold/T_hot
        
        Args:
            resource_name: Name of resource
            T_hot: Hot reservoir temperature (K)
            T_cold: Cold reservoir temperature (K)
            
        Returns:
            Maximum thermodynamic efficiency
        """
        if resource_name not in self.resource_params:
            return 0.3  # Default efficiency
        
        # Carnot efficiency (theoretical maximum)
        carnot_efficiency = 1 - T_cold / T_hot
        
        # Actual efficiency is lower due to irreversibilities
        params = self.resource_params[resource_name]
        actual_efficiency = params.conversion_efficiency * carnot_efficiency
        
        return min(actual_efficiency, 0.95)  # Cap at 95%
    
    def compute_critical_thresholds(self) -> Dict[str, float]:
        """
        Compute critical thresholds C_i for each resource.
        From Corollary 5.1: C_i = max(H_i/r_i, α_safety * K_i)
        
        Returns:
            Dictionary mapping resource names to critical thresholds
        """
        thresholds = {}
        
        for resource_name, params in self.resource_params.items():
            # Safety factor approach
            alpha_safety = 0.2  # 20% safety margin
            threshold = alpha_safety * params.carrying_capacity
            
            # Use the provided critical threshold if available
            if hasattr(params, 'critical_threshold'):
                threshold = max(threshold, params.critical_threshold)
            
            thresholds[resource_name] = threshold
        
        return thresholds
    
    def simulate_dynamics(self, initial_state: np.ndarray, 
                         time_points: np.ndarray,
                         input_func: callable = None,
                         output_func: callable = None) -> np.ndarray:
        """
        Simulate resource dynamics over time.
        
        Args:
            initial_state: Initial resource availability vector
            time_points: Time points for simulation
            input_func: Function returning input rates over time
            output_func: Function returning output rates over time
            
        Returns:
            Array of resource levels over time
        """
        n_resources = len(initial_state)
        n_timesteps = len(time_points)
        
        if input_func is None:
            input_func = lambda t: np.zeros(n_resources)
        if output_func is None:
            output_func = lambda t: np.ones(n_resources) * 0.1  # Small default extraction
        
        def system_dynamics(A, t):
            """Combined dynamics function."""
            inputs = input_func(t)
            outputs = output_func(t)
            
            # Conservation equation
            conservation_term = self.conservation_equation(A, t, inputs, outputs)
            
            # Logistic growth term
            growth_term = self.logistic_growth(A, t)
            
            # Combined dynamics
            dA_dt = conservation_term + growth_term
            
            # Ensure non-negative resources
            for i in range(len(A)):
                if A[i] <= 0 and dA_dt[i] < 0:
                    dA_dt[i] = 0
            
            return dA_dt
        
        # Solve ODE system
        try:
            solution = odeint(system_dynamics, initial_state, time_points)
            
            # Ensure non-negative values
            solution = np.maximum(solution, 0.0)
            
            logger.info(f"Simulated resource dynamics for {len(time_points)} time steps")
            return solution
            
        except Exception as e:
            logger.error(f"Resource dynamics simulation failed: {e}")
            # Return simple exponential decay as fallback
            fallback = np.zeros((n_timesteps, n_resources))
            for i in range(n_resources):
                fallback[:, i] = initial_state[i] * np.exp(-0.01 * time_points)
            return fallback
    
    def assess_sustainability(self, resource_trajectory: np.ndarray) -> Dict[str, bool]:
        """
        Assess sustainability of resource trajectory.
        
        Args:
            resource_trajectory: Array of resource levels over time
            
        Returns:
            Dictionary indicating sustainability for each resource
        """
        sustainability = {}
        critical_thresholds = self.compute_critical_thresholds()
        
        for i, resource_name in enumerate(self.resource_names):
            trajectory = resource_trajectory[:, i]
            threshold = critical_thresholds[resource_name]
            
            # Check if resource stays above critical threshold
            min_level = np.min(trajectory)
            final_level = trajectory[-1]
            
            # Sustainability criteria:
            # 1. Never falls below critical threshold
            # 2. Final level is stable or increasing
            is_sustainable = (min_level >= threshold) and (final_level >= min_level * 0.95)
            
            sustainability[resource_name] = is_sustainable
        
        return sustainability