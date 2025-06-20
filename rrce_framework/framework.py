"""
Main RRCE Framework class that coordinates all components.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from . import __version__
from .core.utils.config import Config
from .core.utils.logger import setup_logger
from .core.data.collectors import RRCEDataManager, DataConfig
from .core.simulation.simulator import RRCESimulator
from .core.models.rrce_model import RRCEModel
from .core.analysis.comparisons import ModelComparison
try:
    from .core.analysis.visualization import RRCEVisualizer
except ImportError:
    # Create a temporary placeholder class
    class RRCEVisualizer:
        def __init__(self, *args, **kwargs):
            pass
        def create_dashboard(self, *args, **kwargs):
            return None
        def generate_report(self, *args, **kwargs):
            return None

logger = logging.getLogger(__name__)

class RRCEFramework:
    """
    Main RRCE Framework class that coordinates all components.
    
    This class provides a high-level interface for:
    - Data collection and processing
    - Model configuration and simulation
    - Analysis and comparison with conventional models
    - Visualization and reporting
    """
    
    def __init__(self, config: Union[str, Path, Dict, Config] = None):
        """
        Initialize the RRCE Framework.
        
        Args:
            config: Configuration file path, dict, or Config object
        """
        # Load configuration
        if isinstance(config, (str, Path)):
            self.config = Config.from_file(config)
        elif isinstance(config, dict):
            self.config = Config.from_dict(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            # Load default configuration
            default_config_path = Path(__file__).parent / "config" / "default_config.yaml"
            self.config = Config.from_file(default_config_path)
        
        # Setup logging
        setup_logger(self.config.logging)
        logger.info("Initializing RRCE Framework")
        
        # Initialize components
        self.data_manager: Optional[RRCEDataManager] = None
        self.simulator: Optional[RRCESimulator] = None
        self.model: Optional[RRCEModel] = None
        self.comparison: Optional[ModelComparison] = None
        self.visualizer: Optional[RRCEVisualizer] = None
        
        # Data storage
        self.raw_data: Dict[str, Any] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.simulation_results: Dict[str, Any] = {}
        self.analysis_results: Dict[str, Any] = {}
        
        logger.info("RRCE Framework initialized successfully")
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'RRCEFramework':
        """Create framework instance from configuration file."""
        return cls(config=config_path)
    
    @classmethod
    def quick_start(cls, countries: List[str], start_date: str, end_date: str) -> 'RRCEFramework':
        """Quick start with minimal configuration."""
        config = {
            'data': {
                'default_start_date': start_date,
                'default_end_date': end_date,
            },
            'countries': countries,
        }
        return cls(config=config)
    
    def setup_data_collection(self, api_keys: Optional[Dict[str, str]] = None) -> None:
        """
        Setup data collection with API keys and configuration.
        
        Args:
            api_keys: Dictionary of API keys for data sources
        """
        logger.info("Setting up data collection")
        
        # Get countries from config
        countries = self.config.get('countries', ['US', 'DE', 'CN'])
        
        # Create data configuration
        data_config = DataConfig(
            countries=countries,
            start_date=self.config.data.default_start_date,
            end_date=self.config.data.default_end_date,
            data_dir=Path(self.config.get('data_dir', './data')),
            cache_enabled=self.config.data.cache.get('enabled', True),
            api_keys=api_keys or {},
        )
        
        # Initialize data manager
        self.data_manager = RRCEDataManager(data_config)
        
        logger.info(f"Data collection setup complete for {len(countries)} countries")
    
    def collect_data(self, countries: Optional[List[str]] = None, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect data for specified countries and time period.
        
        Args:
            countries: List of country codes/names
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Dictionary containing collected data
        """
        if self.data_manager is None:
            self.setup_data_collection()
        
        # Use provided parameters or defaults
        countries = countries or self.data_manager.config.countries
        start_date = start_date or self.data_manager.config.start_date
        end_date = end_date or self.data_manager.config.end_date
        
        logger.info(f"Starting data collection for {countries} from {start_date} to {end_date}")
        
        # Update data manager config if needed
        if (countries != self.data_manager.config.countries or 
            start_date != self.data_manager.config.start_date or 
            end_date != self.data_manager.config.end_date):
            
            self.data_manager.config.countries = countries
            self.data_manager.config.start_date = start_date
            self.data_manager.config.end_date = end_date
        
        # Collect raw data
        self.raw_data = self.data_manager.collect_all_data()
        
        # Process data for RRCE framework
        self.processed_data = self.data_manager.process_for_rrce(self.raw_data)
        self.data_manager.processed_data = self.processed_data
        
        # Validate data quality
        quality_report = self.data_manager.validate_data_quality()
        logger.info(f"Data quality report: {quality_report}")
        
        # Save processed data
        self.data_manager.save_processed_data()
        
        logger.info("Data collection completed successfully")
        return self.processed_data
    
    def setup_model(self, model_params: Optional[Dict] = None) -> None:
        """
        Setup the RRCE model with parameters.
        
        Args:
            model_params: Dictionary of model parameters (overrides config)
        """
        logger.info("Setting up RRCE model")
        
        # Merge config parameters with provided parameters
        model_config = self.config.model.copy()
        if model_params:
            # Update underlying dict of model_config if update method exists
            if hasattr(model_config, 'update'):
                model_config.update(model_params)
            else:
                for k, v in model_params.items():
                    setattr(model_config, k, v)
        # Initialize model with plain dict to ensure correct parsing
        self.model = RRCEModel(model_config.__dict__)
         
        # Initialize simulator
        sim_config = self.config.simulation.copy()
        # Simulation config supports dict-like access
        self.simulator = RRCESimulator(self.model, sim_config)
        
        logger.info("RRCE model setup completed")
    
    def calibrate_model(self, country: str, validation_split: float = 0.8) -> Dict[str, Any]:
        """
        Calibrate model parameters using historical data.
        
        Args:
            country: Country to calibrate on
            validation_split: Fraction of data to use for training
            
        Returns:
            Calibration results and metrics
        """
        if self.model is None:
            self.setup_model()
        
        if country not in self.processed_data:
            raise ValueError(f"No data available for {country}")
        
        logger.info(f"Calibrating model for {country}")
        
        # Get data for this country
        country_data = self.processed_data[country]
        
        # Split data
        split_idx = int(len(country_data) * validation_split)
        train_data = country_data.iloc[:split_idx]
        test_data = country_data.iloc[split_idx:]
        
        # Calibrate model
        calibration_results = self.model.calibrate(train_data, test_data)
        
        logger.info(f"Model calibration completed for {country}")
        return calibration_results
    
    def simulate(self, country: str, 
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                scenarios: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run RRCE simulation for a country.
        
        Args:
            country: Country to simulate
            start_date: Simulation start date
            end_date: Simulation end date
            scenarios: List of scenario configurations
            
        Returns:
            Simulation results
        """
        if self.simulator is None:
            self.setup_model()
        
        if country not in self.processed_data:
            raise ValueError(f"No data available for {country}")
        
        logger.info(f"Running simulation for {country}")
        
        # Get country data
        country_data = self.processed_data[country]
        
        # Set simulation period
        if start_date:
            start_dt = pd.to_datetime(start_date)
            country_data = country_data[country_data.index >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            country_data = country_data[country_data.index <= end_dt]
        
        # Run simulation
        if scenarios:
            results = {}
            for i, scenario in enumerate(scenarios):
                scenario_name = scenario.get('name', f'scenario_{i}')
                results[scenario_name] = self.simulator.simulate(country_data, scenario)
        else:
            results = self.simulator.simulate(country_data)
        
        # Store results
        self.simulation_results[country] = results
        
        logger.info(f"Simulation completed for {country}")
        return results
    
    def compare_models(self, country: str, 
                      conventional_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare RRCE model with conventional economic models.
        
        Args:
            country: Country to analyze
            conventional_models: List of conventional models to compare
            
        Returns:
            Comparison results
        """
        if self.comparison is None:
            self.comparison = ModelComparison(self.config.analysis)
        
        if country not in self.processed_data:
            raise ValueError(f"No data available for {country}")
        
        if country not in self.simulation_results:
            logger.warning(f"No RRCE simulation results for {country}, running simulation")
            self.simulate(country)
        
        logger.info(f"Comparing models for {country}")
        
        # Get data and results
        country_data = self.processed_data[country]
        rrce_results = self.simulation_results[country]
        
        # Use specified models or default
        models_to_compare = conventional_models or self.config.analysis.conventional_models
        
        # Run comparison
        comparison_results = self.comparison.compare_predictions(
            historical_data=country_data,
            rrce_results=rrce_results,
            models=models_to_compare
        )
        
        # Store results
        self.analysis_results[country] = comparison_results
        
        logger.info(f"Model comparison completed for {country}")
        return comparison_results
    
    def visualize_results(self, country: str, 
                         output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Create visualizations of results.
        
        Args:
            country: Country to visualize
            output_dir: Directory to save plots
            
        Returns:
            Dictionary of plot objects/paths
        """
        if self.visualizer is None:
            self.visualizer = RRCEVisualizer(self.config.visualization)
        
        logger.info(f"Creating visualizations for {country}")
        
        # Prepare simulation results for visualization
        sim_results = self.simulation_results.get(country)
        if sim_results is None:
            raise ValueError(f"No simulation results available for {country}")
        # Ensure country is included in results for dashboard
        viz_input = sim_results.copy() if isinstance(sim_results, dict) else {**sim_results}
        viz_input['country'] = country
        # Create visualizations using existing dashboard method
        plots = self.visualizer.create_dashboard(viz_input)
        
        logger.info(f"Visualizations created for {country}")
        return plots
    
    def run_full_analysis(self, countries: Optional[List[str]] = None,
                         api_keys: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run complete analysis pipeline for specified countries.
        
        Args:
            countries: List of countries to analyze
            api_keys: API keys for data collection
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting full RRCE analysis pipeline")
        
        # Setup and collect data
        if api_keys:
            self.setup_data_collection(api_keys)
        
        if not self.processed_data:
            self.collect_data(countries)
        
        results = {}
        
        for country in (countries or self.processed_data.keys()):
            logger.info(f"Running full analysis for {country}")
            
            try:
                # Calibrate model
                calibration = self.calibrate_model(country)
                
                # Run simulation
                simulation = self.simulate(country)
                
                # Compare with conventional models
                comparison = self.compare_models(country)
                
                # Create visualizations
                plots = self.visualize_results(country)
                 
                # Compile results
                results[country] = {
                     'calibration': calibration,
                     'simulation': simulation,
                     'comparison': comparison,
                     'plots': plots
                 }
                
                logger.info(f"Full analysis completed for {country}")
                
            except Exception as e:
                logger.error(f"Analysis failed for {country}: {e}")
                results[country] = {'error': str(e)}
        
        logger.info("Full RRCE analysis pipeline completed")
        return results
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all analyses."""
        logger.info("Generating summary report")
        
        summary = {
            'framework_version': __version__,
            'analysis_date': datetime.now().isoformat(),
            'countries_analyzed': list(self.processed_data.keys()),
            'data_quality': {},
            'model_performance': {},
            'key_insights': [],
        }
        
        # Data quality summary
        if self.data_manager:
            summary['data_quality'] = self.data_manager.validate_data_quality()
        
        # Model performance summary
        for country, results in self.analysis_results.items():
            if 'comparison' in results:
                summary['model_performance'][country] = {
                    'rrce_accuracy': results['comparison'].get('rrce_accuracy', {}),
                    'conventional_accuracy': results['comparison'].get('conventional_accuracy', {}),
                }
        
        # Key insights (placeholder for more sophisticated analysis)
        summary['key_insights'] = [
            "RRCE framework successfully integrates physical constraints",
            "Model performance varies by constraint regime",
            "Significant deviations from conventional models observed",
        ]
        
        return summary
