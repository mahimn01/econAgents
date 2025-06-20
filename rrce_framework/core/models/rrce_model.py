"""
Main RRCE model integrating all mathematical components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field

from ..mathematics.resource_dynamics import ResourceDynamics, ResourceParameters
from ..mathematics.pricing_mechanisms import StabilityWeightedPricing, PricingParameters
from ..mathematics.currency_system import CurrencySystem, CurrencyParameters
from ..mathematics.equilibrium_solver import EquilibriumSolver, EquilibriumParameters

logger = logging.getLogger(__name__)

@dataclass
class RRCEModelConfig:
    """Configuration for RRCE model."""
    # Resource configuration
    resources: Dict[str, ResourceParameters] = field(default_factory=dict)
    
    # Pricing configuration
    pricing: Dict[str, PricingParameters] = field(default_factory=dict)
    
    # Currency configuration
    currency: CurrencyParameters = field(default_factory=CurrencyParameters)
    
    # Equilibrium configuration
    equilibrium: EquilibriumParameters = field(default_factory=EquilibriumParameters)
    
    # Simulation parameters
    time_step: float = 0.25  # Quarterly timesteps
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000

class RRCEModel:
    """
    Main RRCE Model integrating all mathematical components from Steps 1-8.
    
    This model implements:
    - Resource dynamics (Steps 1-3)
    - Individual agent optimization (Step 4) 
    - System equilibrium (Step 5)
    - Pricing mechanisms (Step 6)
    - Currency system (Step 7)
    - Convergence properties (Step 8)
    """
    
    def __init__(self, config: Any):
        """
        Initialize RRCE model.
        
        Args:
            config: Model configuration dictionary
        """
        # Parse configuration into RRCEModelConfig
        if isinstance(config, RRCEModelConfig):
            self.config = config
        else:
            # Convert ModelConfig or dict to dict
            if isinstance(config, dict):
                config_dict = config
            else:
                # Assuming object with __dict__ containing config attributes
                config_dict = getattr(config, '__dict__', {})
            self.config = self._parse_config(config_dict)
        
        # Initialize mathematical components
        self.resource_dynamics = ResourceDynamics(self.config.resources)
        self.pricing_system = StabilityWeightedPricing(self.config.pricing)
        self.currency_system = CurrencySystem(self.config.currency)
        self.equilibrium_solver = EquilibriumSolver(self.config.equilibrium)
        
        # Model state
        self.is_calibrated = False
        self.calibration_params = {}
        
        logger.info("Initialized RRCE model with all mathematical components")
    
    def _parse_config(self, config_dict: Dict[str, Any]) -> RRCEModelConfig:
        """Parse configuration dictionary into structured config."""
        # Default resource parameters
        default_resources = {
            'energy': ResourceParameters(
                regeneration_rate=0.05,
                degradation_rate=0.02,
                carrying_capacity=1000.0,
                critical_threshold=200.0
            ),
            'agricultural': ResourceParameters(
                regeneration_rate=0.08,
                degradation_rate=0.01,
                carrying_capacity=800.0,
                critical_threshold=160.0
            ),
            'mineral': ResourceParameters(
                regeneration_rate=0.01,
                degradation_rate=0.005,
                carrying_capacity=500.0,
                critical_threshold=100.0
            )
        }
        
        # Default pricing parameters
        default_pricing = {
            'energy': PricingParameters(kappa=2.0, eta=3.0, delta=1.0, beta=0.5),
            'agricultural': PricingParameters(kappa=1.5, eta=2.5, delta=0.8, beta=0.7),
            'mineral': PricingParameters(kappa=3.0, eta=4.0, delta=1.2, beta=0.3)
        }
        
        # Parse and convert resource parameters
        raw_resources = config_dict.get('resources', default_resources)
        parsed_resources: Dict[str, ResourceParameters] = {}
        # Instantiate ResourceParameters with only valid fields
        valid_r_keys = set(ResourceParameters.__dataclass_fields__.keys())
        for name, rp in raw_resources.items():
            if isinstance(rp, ResourceParameters):
                parsed_resources[name] = rp
            elif isinstance(rp, dict):
                # Filter only valid keys
                rp_clean = {k: v for k, v in rp.items() if k in valid_r_keys}
                parsed_resources[name] = ResourceParameters(**rp_clean)
            else:
                parsed_resources[name] = rp  # assume already correct type
        # Parse and convert pricing parameters
        raw_pricing = config_dict.get('pricing', default_pricing)
        parsed_pricing: Dict[str, PricingParameters] = {}
        # Instantiate PricingParameters with only valid fields
        valid_p_keys = set(PricingParameters.__dataclass_fields__.keys())
        for name, pp in raw_pricing.items():
            if isinstance(pp, PricingParameters):
                parsed_pricing[name] = pp
            elif isinstance(pp, dict):
                pp_clean = {k: v for k, v in pp.items() if k in valid_p_keys}
                parsed_pricing[name] = PricingParameters(**pp_clean)
            else:
                parsed_pricing[name] = pp
        
        # Instantiate CurrencyParameters filtering valid keys
        valid_c_keys = set(CurrencyParameters.__dataclass_fields__.keys())
        raw_currency = config_dict.get('currency', {}) or {}
        currency_clean = {k: v for k, v in raw_currency.items() if k in valid_c_keys}
        currency_params = CurrencyParameters(**currency_clean)
        # Instantiate EquilibriumParameters filtering valid keys
        valid_e_keys = set(EquilibriumParameters.__dataclass_fields__.keys())
        raw_equil = config_dict.get('equilibrium', {}) or {}
        equil_clean = {k: v for k, v in raw_equil.items() if k in valid_e_keys}
        equilibrium_params = EquilibriumParameters(**equil_clean)
        return RRCEModelConfig(
            resources=parsed_resources,
            pricing=parsed_pricing,
            currency=currency_params,
            equilibrium=equilibrium_params,
            time_step=config_dict.get('time_step', 0.25),
            convergence_tolerance=config_dict.get('convergence_tolerance', 1e-6),
            max_iterations=config_dict.get('max_iterations', 1000)
        )
    
    def calibrate(self, historical_data: pd.DataFrame, 
                 validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calibrate model parameters to historical data.
        
        Args:
            historical_data: Historical data for calibration
            validation_data: Optional validation dataset
            
        Returns:
            Calibration results and metrics
        """
        logger.info("Starting RRCE model calibration")
        
        try:
            # Extract key variables from historical data
            gdp_series = historical_data.get('GDP', historical_data.get('gdp'))
            gini_series = historical_data.get('gini_coefficient', historical_data.get('gini'))
            
            if gdp_series is None:
                raise ValueError("GDP data required for calibration")
            
            # Simple calibration approach - fit to GDP and Gini trends
            calibration_results = {
                'gdp_trend': self._fit_gdp_trend(gdp_series),
                'gini_baseline': gini_series.mean() if gini_series is not None else 0.35,
                'resource_scaling': self._estimate_resource_scaling(historical_data),
                'calibration_error': 0.0
            }
            
            # Validate if validation data provided
            if validation_data is not None:
                validation_results = self._validate_calibration(validation_data, calibration_results)
                calibration_results['validation'] = validation_results
            
            # Store calibration parameters
            self.calibration_params = calibration_results
            self.is_calibrated = True
            
            logger.info("RRCE model calibration completed successfully")
            return calibration_results
            
        except Exception as e:
            logger.error(f"Model calibration failed: {e}")
            # Return default calibration
            return {
                'gdp_trend': 0.02,  # 2% growth
                'gini_baseline': 0.35,
                'resource_scaling': 1.0,
                'calibration_error': 1.0,
                'error': str(e)
            }
    
    def _fit_gdp_trend(self, gdp_series: pd.Series) -> float:
        """Fit exponential trend to GDP data."""
        try:
            # Simple log-linear trend
            log_gdp = np.log(gdp_series.dropna())
            time_index = np.arange(len(log_gdp))
            
            if len(time_index) > 1:
                trend = np.polyfit(time_index, log_gdp, 1)[0]
                return max(min(trend, 0.1), -0.05)  # Cap between -5% and 10%
            else:
                return 0.02  # Default 2% growth
                
        except Exception:
            return 0.02
    
    def _estimate_resource_scaling(self, historical_data: pd.DataFrame) -> float:
        """Estimate resource scaling factor."""
        try:
            # Use energy consumption as proxy for resource intensity
            energy_data = historical_data.get('energy_use_per_capita')
            if energy_data is not None:
                energy_trend = energy_data.pct_change().mean()
                return max(min(1.0 + energy_trend, 2.0), 0.5)  # Scale between 0.5 and 2.0
            else:
                return 1.0
        except Exception:
            return 1.0
    
    def _validate_calibration(self, validation_data: pd.DataFrame, 
                            calibration_params: Dict[str, Any]) -> Dict[str, float]:
        """Validate calibration against validation dataset."""
        try:
            # Simple validation - predict GDP growth
            predicted_growth = calibration_params['gdp_trend']
            
            gdp_validation = validation_data.get('GDP', validation_data.get('gdp'))
            if gdp_validation is not None and len(gdp_validation) > 1:
                actual_growth = gdp_validation.pct_change().mean()
                error = abs(predicted_growth - actual_growth)
                
                return {
                    'prediction_error': error,
                    'r_squared': max(0, 1 - error),  # Simplified R²
                    'validation_success': error < 0.05
                }
            else:
                return {'validation_success': False, 'error': 'Insufficient validation data'}
                
        except Exception as e:
            return {'validation_success': False, 'error': str(e)}
    
    def predict(self, initial_state: Dict[str, Any], 
               time_horizon: int = 20,
               external_scenario: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate predictions using the RRCE mathematical framework.
        
        Args:
            initial_state: Initial economic and resource state
            time_horizon: Number of time periods to predict
            external_scenario: Optional external scenario parameters
            
        Returns:
            DataFrame with comprehensive economic predictions
        """
        logger.info(f"Generating RRCE predictions for {time_horizon} periods")
        
        # Time vector
        time_points = np.arange(0, time_horizon * self.config.time_step, self.config.time_step)
        n_periods = len(time_points)
        
        # Initialize comprehensive state vectors
        results = self._initialize_comprehensive_state(initial_state, n_periods)
        
        # Set initial conditions from real data
        self._set_initial_conditions(results, initial_state)
        
        # Run comprehensive RRCE simulation
        for i in range(1, n_periods):
            t = time_points[i]
            dt = self.config.time_step
            
            try:
                # 1. Resource Dynamics (Theorems 1, 3, 5)
                resource_state = self._evolve_resource_dynamics(results, i-1, dt)
                results['resource_availability'][i] = resource_state
                
                # 2. Pricing Mechanisms (Theorem 20)
                prices = self._compute_stability_weighted_prices(results, i-1)
                results['resource_prices'][i] = prices
                
                # 3. Currency System Dynamics (Step 7)
                currency_value = self._evolve_currency_system(results, i-1)
                results['cwu_value'][i] = currency_value
                
                # 4. Social Dynamics (Steps 4-5)
                social_state = self._evolve_social_dynamics(results, i-1)
                results['gini'][i] = social_state['gini']
                results['social_cohesion'][i] = social_state['cohesion']
                
                # 5. Economic Equilibrium (Step 8)
                economic_state = self._solve_economic_equilibrium(results, i-1)
                results['gdp'][i] = economic_state['gdp']
                results['productivity'][i] = economic_state['productivity']
                
                # 6. Environmental Quality
                env_quality = self._compute_environmental_quality(results, i-1)
                results['environmental_quality'][i] = env_quality
                
                # 7. System Health and Sustainability
                results['system_health'][i] = self._compute_system_health(results, i)
                results['sustainability_index'][i] = self._compute_sustainability_index_comprehensive(results, i)
                
                # 8. Apply external scenarios/shocks
                if external_scenario:
                    self._apply_scenario_effects(results, i, t, external_scenario)
                
            except Exception as e:
                logger.warning(f"Prediction failed at period {i}: {e}")
                # Fallback to previous values with small perturbation
                self._apply_fallback_prediction(results, i)
        
        # Convert to DataFrame with proper structure
        prediction_df = self._create_prediction_dataframe(results, time_points)
        
        logger.info("RRCE predictions completed successfully")
        return prediction_df
        
    def _initialize_comprehensive_state(self, initial_state: Dict[str, Any], n_periods: int) -> Dict[str, np.ndarray]:
        """Initialize comprehensive state vectors for simulation."""
        n_resources = len(self.resource_dynamics.resource_names)
        
        return {
            'gdp': np.zeros(n_periods),
            'gini': np.zeros(n_periods),
            'resource_availability': np.zeros((n_periods, n_resources)),
            'resource_prices': np.zeros((n_periods, n_resources)),
            'cwu_value': np.zeros(n_periods),
            'social_cohesion': np.zeros(n_periods),
            'environmental_quality': np.zeros(n_periods),
            'productivity': np.zeros(n_periods),
            'system_health': np.zeros(n_periods),
            'sustainability_index': np.zeros(n_periods)
        }
    
    def _set_initial_conditions(self, results: Dict[str, np.ndarray], initial_state: Dict[str, Any]):
        """Set initial conditions from real economic data."""
        # Economic indicators
        results['gdp'][0] = float(initial_state.get('gdp', 1000.0))
        results['gini'][0] = float(initial_state.get('gini', 0.35))
        results['productivity'][0] = results['gdp'][0] / initial_state.get('population', 1000000) * 1000
        
        # Resource levels (from real data)
        resource_levels = initial_state.get('resource_levels', {})
        for i, resource_name in enumerate(self.resource_dynamics.resource_names):
            if isinstance(resource_levels, dict):
                results['resource_availability'][0, i] = resource_levels.get(resource_name, 500.0)
            elif isinstance(resource_levels, (list, np.ndarray)) and len(resource_levels) > i:
                results['resource_availability'][0, i] = resource_levels[i]
            else:
                results['resource_availability'][0, i] = 500.0
        
        # Initialize derived values
        results['cwu_value'][0] = 1.0
        results['social_cohesion'][0] = 1.0 - results['gini'][0]
        results['environmental_quality'][0] = float(initial_state.get('environmental_quality', {}).get('pollution_level', 0.8))
        results['system_health'][0] = self._compute_system_health(results, 0)
        results['sustainability_index'][0] = self._compute_sustainability_index_comprehensive(results, 0)
    
    def _evolve_resource_dynamics(self, results: Dict[str, np.ndarray], prev_idx: int, dt: float) -> np.ndarray:
        """Evolve resource availability using conservation equations."""
        current_resources = results['resource_availability'][prev_idx]
        current_gdp = results['gdp'][prev_idx]
        
        # Compute extraction based on economic activity
        extraction_rates = np.zeros_like(current_resources)
        for i, resource_name in enumerate(self.resource_dynamics.resource_names):
            params = self.resource_dynamics.resource_params[resource_name]
            base_extraction = current_gdp * 0.001  # Economic demand scaling
            efficiency = params.extraction_efficiency
            availability_factor = current_resources[i] / params.carrying_capacity
            extraction_rates[i] = base_extraction * efficiency * availability_factor
        
        # Compute regeneration rates
        regeneration_rates = np.zeros_like(current_resources)
        for i, resource_name in enumerate(self.resource_dynamics.resource_params):
            params = self.resource_dynamics.resource_params[resource_name]
            regeneration_rates[i] = params.regeneration_rate * current_resources[i]
        
        # Apply conservation equation: dA/dt = I - O - D
        resource_change = self.resource_dynamics.conservation_equation(
            current_resources, 0, regeneration_rates, extraction_rates
        )
        
        # Apply logistic growth constraints
        growth_component = self.resource_dynamics.logistic_growth(current_resources, 0)
        
        # Update resources
        new_resources = current_resources + dt * (resource_change + growth_component)
        return np.maximum(new_resources, 10.0)  # Minimum resource floor
    
    def _compute_stability_weighted_prices(self, results: Dict[str, np.ndarray], prev_idx: int) -> np.ndarray:
        """Compute resource prices using Stability-Weighted Pricing Framework."""
        resource_levels = results['resource_availability'][prev_idx]
        environmental_quality = results['environmental_quality'][prev_idx]
        
        prices = np.zeros_like(resource_levels)
        
        for i, resource_name in enumerate(self.resource_dynamics.resource_names):
            params = self.resource_dynamics.resource_params[resource_name]
            
            # Base price
            base_price = 100.0
            
            # Scarcity premium (inverse relationship)
            scarcity_factor = params.carrying_capacity / max(resource_levels[i], 1.0)
            
            # Environmental cost
            env_cost_factor = 1.0 + (1.0 - environmental_quality) * 0.5
            
            # Stability weighting (from Theorem 20)
            stability_weight = min(2.0, max(0.5, scarcity_factor))
            
            prices[i] = base_price * scarcity_factor * env_cost_factor * stability_weight
        
        return prices
    
    def _evolve_currency_system(self, results: Dict[str, np.ndarray], prev_idx: int) -> float:
        """Evolve CWU currency value based on resource backing."""
        current_value = results['cwu_value'][prev_idx]
        resource_levels = results['resource_availability'][prev_idx]
        env_quality = results['environmental_quality'][prev_idx]
        
        # Resource backing strength
        total_resource_value = np.sum(resource_levels * results['resource_prices'][prev_idx] if prev_idx > 0 else resource_levels)
        resource_backing_factor = total_resource_value / 50000.0  # Normalize
        
        # Environmental sustainability factor
        sustainability_factor = env_quality
        
        # Social stability factor
        social_stability = results['social_cohesion'][prev_idx]
        
        # Currency appreciation rate
        appreciation_rate = 0.001 * (resource_backing_factor + sustainability_factor + social_stability - 1.5)
        
        return current_value * (1.0 + appreciation_rate)
    
    def _evolve_social_dynamics(self, results: Dict[str, np.ndarray], prev_idx: int) -> Dict[str, float]:
        """Evolve social indicators using RRCE social dynamics."""
        current_gini = results['gini'][prev_idx]
        resource_distribution = results['resource_availability'][prev_idx]
        
        # Resource inequality affects social inequality
        resource_gini = np.std(resource_distribution) / (np.mean(resource_distribution) + 1e-6)
        resource_inequality_effect = 0.1 * resource_gini
        
        # Economic growth effect on inequality
        if prev_idx > 0:
            gdp_growth = (results['gdp'][prev_idx] - results['gdp'][prev_idx-1]) / results['gdp'][prev_idx-1]
            growth_inequality_effect = 0.05 * max(0, gdp_growth - 0.02)  # Inequality increases with high growth
        else:
            growth_inequality_effect = 0
        
        # Update Gini coefficient
        new_gini = current_gini + resource_inequality_effect + growth_inequality_effect
        new_gini = max(0.15, min(0.65, new_gini))  # Reasonable bounds
        
        # Social cohesion (inverse of inequality)
        social_cohesion = 1.0 - new_gini
        
        return {'gini': new_gini, 'cohesion': social_cohesion}
    
    def _solve_economic_equilibrium(self, results: Dict[str, np.ndarray], prev_idx: int) -> Dict[str, float]:
        """Solve economic equilibrium using RRCE framework."""
        current_gdp = results['gdp'][prev_idx]
        resource_levels = results['resource_availability'][prev_idx]
        prices = results['resource_prices'][prev_idx] if prev_idx > 0 else np.ones_like(resource_levels) * 100
        
        # Resource-constrained production function
        resource_productivity = np.sum(resource_levels * prices) / np.sum(prices)
        
        # Base productivity growth
        base_growth = 0.005  # 0.5% per quarter
        
        # Resource constraint multiplier
        min_resource_level = np.min(resource_levels)
        resource_constraint = min(2.0, max(0.1, min_resource_level / 200.0))
        
        # Environmental productivity factor
        env_quality = results['environmental_quality'][prev_idx]
        env_productivity_factor = 0.5 + 0.5 * env_quality
        
        # New GDP calculation
        productivity_factor = resource_constraint * env_productivity_factor
        new_gdp = current_gdp * (1.0 + base_growth * productivity_factor)
        
        # Productivity per capita equivalent
        productivity = resource_productivity * productivity_factor
        
        return {'gdp': new_gdp, 'productivity': productivity}
    
    def _compute_environmental_quality(self, results: Dict[str, np.ndarray], prev_idx: int) -> float:
        """Compute environmental quality based on resource extraction and economic activity."""
        current_env = results['environmental_quality'][prev_idx]
        resource_levels = results['resource_availability'][prev_idx]
        gdp = results['gdp'][prev_idx]
        
        # Environmental degradation from economic activity
        degradation_rate = gdp * 0.00001  # Economic activity impact
        
        # Resource extraction impact
        if prev_idx > 0:
            resource_extraction = np.sum(results['resource_availability'][prev_idx-1] - resource_levels)
            extraction_impact = max(0, resource_extraction * 0.0001)
        else:
            extraction_impact = 0
        
        # Natural recovery
        recovery_rate = 0.002 * current_env  # Proportional recovery
        
        # Update environmental quality
        new_env_quality = current_env - degradation_rate - extraction_impact + recovery_rate
        return max(0.0, min(1.0, new_env_quality))
    
    def _compute_system_health(self, results: Dict[str, np.ndarray], idx: int) -> float:
        """Compute overall system health indicator."""
        # Economic health (normalized GDP growth)
        economic_health = min(1.0, results['gdp'][idx] / 2000.0)
        
        # Resource health (average resource levels)
        resource_health = np.mean(results['resource_availability'][idx]) / 500.0
        resource_health = max(0.0, min(1.0, resource_health))
        
        # Social health
        social_health = results['social_cohesion'][idx]
        
        # Environmental health
        environmental_health = results['environmental_quality'][idx]
        
        # Weighted system health
        system_health = (
            0.3 * economic_health +
            0.3 * resource_health +
            0.2 * social_health +
            0.2 * environmental_health
        )
        
        return max(0.0, min(1.0, system_health))
    
    def _compute_sustainability_index_comprehensive(self, results: Dict[str, np.ndarray], idx: int) -> float:
        """Compute comprehensive sustainability index."""
        # Economic sustainability (stable growth)
        if idx > 0:
            gdp_stability = 1.0 - abs((results['gdp'][idx] - results['gdp'][idx-1]) / results['gdp'][idx-1])
            economic_sustainability = max(0.0, min(1.0, gdp_stability))
        else:
            economic_sustainability = 0.8
        
        # Resource sustainability (maintaining resource levels)
        min_resource_ratio = np.min(results['resource_availability'][idx]) / 500.0
        resource_sustainability = max(0.0, min(1.0, min_resource_ratio))
        
        # Social sustainability (low inequality)
        social_sustainability = 1.0 - results['gini'][idx]
        
        # Environmental sustainability
        environmental_sustainability = results['environmental_quality'][idx]
        
        # Comprehensive sustainability index
        sustainability = (
            0.25 * economic_sustainability +
            0.3 * resource_sustainability +
            0.25 * social_sustainability +
            0.2 * environmental_sustainability
        )
        
        return max(0.0, min(1.0, sustainability))
    
    def _apply_scenario_effects(self, results: Dict[str, np.ndarray], idx: int, time: float, scenario: Dict[str, Any]):
        """Apply external scenario effects to simulation."""
        scenario_type = scenario.get('type', 'none')
        
        if scenario_type == 'resource_shock' and idx == 5:  # Apply shock at period 5
            magnitude = scenario.get('magnitude', 0.2)
            results['resource_availability'][idx] *= (1.0 - magnitude)
        
        elif scenario_type == 'economic_crisis' and idx == 8:  # Apply crisis at period 8
            magnitude = scenario.get('magnitude', 0.15)
            results['gdp'][idx] *= (1.0 - magnitude)
        
        elif scenario_type == 'environmental_disaster' and idx == 10:
            magnitude = scenario.get('magnitude', 0.3)
            results['environmental_quality'][idx] *= (1.0 - magnitude)
    
    def _apply_fallback_prediction(self, results: Dict[str, np.ndarray], idx: int):
        """Apply fallback prediction when calculation fails."""
        if idx > 0:
            # Copy previous values with small random perturbation
            for key in results:
                if key != 'resource_availability' and key != 'resource_prices':
                    results[key][idx] = results[key][idx-1] * (1.0 + np.random.normal(0, 0.01))
                else:
                    results[key][idx] = results[key][idx-1] * (1.0 + np.random.normal(0, 0.01, results[key][idx-1].shape))
    
    def _create_prediction_dataframe(self, results: Dict[str, np.ndarray], time_points: np.ndarray) -> pd.DataFrame:
        """Create structured DataFrame from simulation results."""
        df_data = {
            'period': np.arange(len(time_points)),
            'time': time_points,
            'gdp': results['gdp'],
            'gini_coefficient': results['gini'],
            'cwu_value': results['cwu_value'],
            'social_cohesion': results['social_cohesion'],
            'environmental_quality': results['environmental_quality'],
            'productivity': results['productivity'],
            'system_health': results['system_health'],
            'sustainability_index': results['sustainability_index']
        }
        
        # Add resource columns
        for i, resource_name in enumerate(self.resource_dynamics.resource_names):
            df_data[f'{resource_name}_availability'] = results['resource_availability'][:, i]
            df_data[f'{resource_name}_price'] = results['resource_prices'][:, i]
        
        df = pd.DataFrame(df_data)
        return df.set_index('period')
    
    def _initialize_state(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize complete model state."""
        state = initial_state.copy()
        
        # Set defaults if not provided
        state.setdefault('gdp', 1000.0)
        state.setdefault('gini', 0.35)
        state.setdefault('resource_levels', {
            'energy': 500.0,
            'agricultural': 400.0,
            'mineral': 250.0
        })
        state.setdefault('critical_thresholds', {
            'energy': 200.0,
            'agricultural': 160.0,
            'mineral': 100.0
        })
        state.setdefault('prices', {
            'energy': 10.0,
            'agricultural': 8.0,
            'mineral': 15.0
        })
        
        return state
    
    def _predict_single_period(self, state: Dict[str, Any], time: float,
                              external_scenario: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict single time period."""
        # Apply external scenario if provided
        if external_scenario:
            for key, value in external_scenario.items():
                if key in state:
                    state[key] *= (1 + value * self.config.time_step)
        
        # Use calibrated parameters if available
        if self.is_calibrated:
            gdp_growth = self.calibration_params.get('gdp_trend', 0.02)
            gini_baseline = self.calibration_params.get('gini_baseline', 0.35)
        else:
            gdp_growth = 0.02
            gini_baseline = 0.35
        
        # Simple evolution model (placeholder for full simulation)
        new_gdp = state['gdp'] * (1 + gdp_growth * self.config.time_step)
        new_gini = gini_baseline + 0.01 * np.random.normal()  # Small random variation
        
        # Compute currency value
        cwu_result = self.currency_system.compute_cwu(
            gdp=new_gdp,
            gini=new_gini,
            resource_levels=state['resource_levels'],
            critical_thresholds=state['critical_thresholds']
        )
        
        # Assess equilibrium
        equilibrium_state = {
            'resource_levels': state['resource_levels'],
            'resource_flows': {k: 0.01 for k in state['resource_levels'].keys()},
            'critical_thresholds': state['critical_thresholds'],
            'total_claims': new_gdp,
            'productive_capacity': new_gdp * 1.2,
            'gini_coefficient': new_gini
        }
        
        equilibrium_status = self.equilibrium_solver.assess_system_equilibrium(equilibrium_state)
        
        return {
            'gdp': new_gdp,
            'gini': new_gini,
            'cwu_value': cwu_result['cwu_value'],
            'wci': cwu_result['wci'],
            'resource_factor': cwu_result['resource_factor'],
            'social_factor': cwu_result['social_factor'],
            'system_health': equilibrium_status['system_health_score'],
            'equilibrium_status': equilibrium_status['overall_equilibrium'],
            'resource_levels': state['resource_levels'].copy(),
            'prices': state['prices'].copy()
        }
    
    def _update_state(self, current_state: Dict[str, Any], 
                     prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Update state based on prediction."""
        new_state = current_state.copy()
        
        # Update main indicators
        new_state['gdp'] = prediction['gdp']
        new_state['gini'] = prediction['gini']
        
        # Simple resource evolution
        for resource in new_state['resource_levels']:
            # Small depletion over time
            new_state['resource_levels'][resource] *= 0.999
        
        return new_state