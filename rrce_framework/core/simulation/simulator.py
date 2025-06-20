"""
Main simulation engine for RRCE Framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime, timedelta

from rrce_framework.core.utils.config import SimulationConfig
from rrce_framework.core.models.rrce_model import RRCEModel
from rrce_framework.core.mathematics.resource_dynamics import ResourceParameters

logger = logging.getLogger(__name__)

class RRCESimulator:
    """
    Main simulation engine that orchestrates RRCE model execution.
    """
    
    def __init__(self, model: RRCEModel, simulation_config: Dict[str, Any]):
        """
        Initialize RRCE simulator.
        
        Args:
            model: RRCE model instance
            simulation_config: Simulation configuration
        """
        self.model = model
        self.config = simulation_config
        
        # Simulation parameters
        self.time_step = simulation_config.get('time_step', 0.25)
        self.max_iterations = simulation_config.get('max_iterations', 1000)
        self.convergence_tolerance = simulation_config.get('convergence_tolerance', 1e-6)
        
        logger.info("Initialized RRCE simulator")
    
    def simulate(self, input_data: pd.DataFrame, 
                scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run RRCE simulation.
        
        Args:
            input_data: Input economic and resource data
            scenario: Optional scenario configuration
            
        Returns:
            Simulation results dictionary
        """
        logger.info("Starting RRCE simulation")
        
        try:
            # Prepare initial state from input data
            initial_state = self._prepare_initial_state(input_data)
            
            # Determine simulation horizon
            if scenario and 'time_horizon' in scenario:
                time_horizon = scenario['time_horizon']
            else:
                time_horizon = 20  # Default 20 periods (5 years if quarterly)
            
            # Run simulation
            if scenario and scenario.get('type') == 'monte_carlo':
                results = self._run_monte_carlo_simulation(
                    initial_state, time_horizon, scenario
                )
            else:
                results = self._run_deterministic_simulation(
                    initial_state, time_horizon, scenario
                )
            
            # Add metadata
            results['metadata'] = {
                'simulation_type': scenario.get('type', 'deterministic') if scenario else 'deterministic',
                'time_horizon': time_horizon,
                'time_step': self.time_step,
                'model_calibrated': self.model.is_calibrated,
                'simulation_date': datetime.now().isoformat()
            }
            
            logger.info("RRCE simulation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"RRCE simulation failed: {e}")
            return {
                'error': str(e),
                'predictions': pd.DataFrame(),
                'diagnostics': {'simulation_failed': True}
            }
    
    def _prepare_initial_state(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare initial state from input data."""
        # Get the most recent data point
        if len(input_data) == 0:
            # Default initial state
            return {
                'gdp': 1000.0,
                'gini': 0.35,
                'resource_levels': {
                    'energy': 500.0,
                    'agricultural': 400.0,
                    'mineral': 250.0
                }
            }
        
        latest_data = input_data.iloc[-1]
        
        # Extract key variables
        initial_state = {
            'gdp': latest_data.get('GDP', latest_data.get('gdp', 1000.0)),
            'gini': latest_data.get('gini_coefficient', latest_data.get('gini', 0.35)),
            'population': latest_data.get('population', 1000000),
            'resource_levels': {
                'energy': latest_data.get('energy_use_per_capita', 500.0),
                'agricultural': latest_data.get('arable_land_per_capita', 400.0),
                'mineral': latest_data.get('mineral_extraction', 250.0)
            },
            'environmental_quality': {
                'co2_emissions': latest_data.get('co2_emissions_per_capita', 10.0),
                'pollution_level': latest_data.get('pollution_index', 0.1)
            }
        }
        
        return initial_state
    
    def _run_deterministic_simulation(self, initial_state: Dict[str, Any],
                                    time_horizon: int,
                                    scenario: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run deterministic simulation."""
        # Generate predictions using RRCE model
        predictions = self.model.predict(
            initial_state=initial_state,
            time_horizon=time_horizon,
            external_scenario=scenario.get('external_shocks') if scenario else None
        )
        
        # Compute diagnostics
        diagnostics = self._compute_diagnostics(predictions, initial_state)
        
        return {
            'predictions': predictions,
            'diagnostics': diagnostics,
            'initial_state': initial_state,
            'scenario': scenario
        }
    
    def _run_monte_carlo_simulation(self, initial_state: Dict[str, Any],
                                  time_horizon: int,
                                  scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation with uncertainty."""
        n_simulations = scenario.get('n_simulations', 100)
        
        all_predictions = []
        
        for i in range(n_simulations):
            # Add random perturbations to initial state
            perturbed_state = self._add_uncertainty(initial_state, scenario)
            
            # Run single simulation
            predictions = self.model.predict(
                initial_state=perturbed_state,
                time_horizon=time_horizon,
                external_scenario=scenario.get('external_shocks')
            )
            
            predictions['simulation_id'] = i
            all_predictions.append(predictions)
        
        # Combine results
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Compute statistics
        prediction_stats = self._compute_monte_carlo_stats(combined_predictions)
        
        # Compute diagnostics
        diagnostics = self._compute_diagnostics(prediction_stats['mean'], initial_state)
        diagnostics['monte_carlo_stats'] = prediction_stats
        
        return {
            'predictions': prediction_stats['mean'],
            'prediction_bands': prediction_stats,
            'all_simulations': combined_predictions,
            'diagnostics': diagnostics,
            'initial_state': initial_state,
            'scenario': scenario
        }
    
    def _add_uncertainty(self, initial_state: Dict[str, Any], 
                        scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Add random uncertainty to initial state."""
        perturbed_state = initial_state.copy()
        
        # Get uncertainty parameters
        uncertainty = scenario.get('uncertainty', {})
        
        # Add GDP uncertainty
        gdp_std = uncertainty.get('gdp_std', 0.02)
        perturbed_state['gdp'] *= (1 + np.random.normal(0, gdp_std))
        
        # Add Gini uncertainty
        gini_std = uncertainty.get('gini_std', 0.01)
        perturbed_state['gini'] += np.random.normal(0, gini_std)
        perturbed_state['gini'] = np.clip(perturbed_state['gini'], 0, 1)
        
        # Add resource uncertainty
        resource_std = uncertainty.get('resource_std', 0.05)
        for resource in perturbed_state['resource_levels']:
            perturbed_state['resource_levels'][resource] *= (1 + np.random.normal(0, resource_std))
        
        return perturbed_state
    
    def _compute_monte_carlo_stats(self, combined_predictions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute statistics from Monte Carlo simulations."""
        # Group by period
        grouped = combined_predictions.groupby('period')
        
        # Compute percentiles
        percentiles = [5, 25, 50, 75, 95]
        stats = {}
        
        for percentile in percentiles:
            stats[f'p{percentile}'] = grouped.quantile(percentile / 100.0)
        
        stats['mean'] = grouped.mean()
        stats['std'] = grouped.std()
        
        return stats
    
    def _compute_diagnostics(self, predictions: pd.DataFrame, 
                           initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute simulation diagnostics."""
        diagnostics = {}
        
        try:
            # Growth rate analysis
            if 'gdp' in predictions.columns:
                gdp_growth = predictions['gdp'].pct_change().mean()
                diagnostics['average_gdp_growth'] = gdp_growth
                diagnostics['gdp_volatility'] = predictions['gdp'].pct_change().std()
            
            # Inequality analysis
            if 'gini' in predictions.columns:
                diagnostics['average_gini'] = predictions['gini'].mean()
                diagnostics['gini_trend'] = predictions['gini'].iloc[-1] - predictions['gini'].iloc[0]
            
            # System health analysis
            if 'system_health' in predictions.columns:
                diagnostics['average_system_health'] = predictions['system_health'].mean()
                diagnostics['health_deterioration'] = (
                    predictions['system_health'].iloc[0] - predictions['system_health'].iloc[-1]
                )
            
            # Currency analysis
            if 'cwu_value' in predictions.columns:
                diagnostics['currency_stability'] = predictions['cwu_value'].std()
                diagnostics['currency_trend'] = (
                    predictions['cwu_value'].iloc[-1] / predictions['cwu_value'].iloc[0] - 1
                )
            
            # Equilibrium analysis
            if 'equilibrium_status' in predictions.columns:
                diagnostics['equilibrium_periods'] = predictions['equilibrium_status'].sum()
                diagnostics['equilibrium_ratio'] = predictions['equilibrium_status'].mean()
            
            # Sustainability indicators
            diagnostics['sustainability_score'] = self._compute_sustainability_score(predictions)
            
        except Exception as e:
            logger.warning(f"Failed to compute some diagnostics: {e}")
            diagnostics['computation_error'] = str(e)
        
        return diagnostics
    
    def _compute_sustainability_score(self, predictions: pd.DataFrame) -> float:
        """Compute overall sustainability score."""
        try:
            score = 0.0
            weights = 0.0
            
            # Economic sustainability (stable growth)
            if 'gdp' in predictions.columns:
                gdp_stability = 1.0 / (1.0 + predictions['gdp'].pct_change().std())
                score += 0.3 * gdp_stability
                weights += 0.3
            
            # Social sustainability (stable inequality)
            if 'gini' in predictions.columns:
                gini_stability = 1.0 / (1.0 + predictions['gini'].std())
                score += 0.3 * gini_stability
                weights += 0.3
            
            # System health
            if 'system_health' in predictions.columns:
                avg_health = predictions['system_health'].mean()
                score += 0.4 * avg_health
                weights += 0.4
            
            return score / weights if weights > 0 else 0.5
            
        except Exception:
            return 0.5

    def run(self, data: pd.DataFrame, num_periods: int) -> pd.DataFrame:
        logging.info("Starting RRCE simulation")
        
        # Convert resource parameter dicts to ResourceParameters objects
        resource_params_dict = self.config.get('resource_params', {})
        resource_params = {
            name: ResourceParameters(**params) 
            for name, params in resource_params_dict.items()
        }
        
        initial_state = self.model.prepare_initial_state(data)
        
        predictions = self.model.predict(
            initial_state=initial_state,
            num_periods=num_periods,
            resource_params=resource_params
        )
        
        logging.info("RRCE simulation completed successfully")
        return predictions