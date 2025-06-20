"""
Model comparison framework for RRCE vs conventional models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

from ..models.conventional.dsge import SimpleDSGEModel
from ..models.conventional.var_model import SimpleVARModel

logger = logging.getLogger(__name__)

class ModelComparison:
    """
    Framework for comparing RRCE model with conventional economic models.
    """
    
    def __init__(self, analysis_config: Dict[str, Any]):
        """
        Initialize model comparison framework.
        
        Args:
            analysis_config: Analysis configuration dictionary
        """
        self.config = analysis_config
        
        # Initialize conventional models
        self.conventional_models = {
            'dsge': SimpleDSGEModel(),
            'var': SimpleVARModel(lags=4)
        }
        
        # Metrics to compute
        self.metrics = analysis_config.get('metrics', ['rmse', 'mae', 'mape', 'directional_accuracy'])
        
        logger.info("Initialized model comparison framework")
    
    def compare_predictions(self, historical_data: pd.DataFrame,
                          rrce_results: Dict[str, Any],
                          models: Optional[List[str]] = None,
                          test_period: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Compare RRCE model predictions with conventional models.
        
        Args:
            historical_data: Historical data for calibration and testing
            rrce_results: Results from RRCE simulation
            models: List of conventional models to compare (default: all)
            test_period: Optional test period (start_date, end_date)
            
        Returns:
            Comprehensive comparison results
        """
        logger.info("Starting model comparison analysis")
        
        if models is None:
            models = list(self.conventional_models.keys())
        
        try:
            # Split data into training and testing
            train_data, test_data = self._split_data(historical_data, test_period)
            
            # Get RRCE predictions
            rrce_predictions = rrce_results.get('predictions', pd.DataFrame())
            
            # Generate conventional model predictions
            conventional_predictions = {}
            calibration_results = {}
            
            for model_name in models:
                if model_name in self.conventional_models:
                    model = self.conventional_models[model_name]
                    
                    # Calibrate model
                    cal_result = model.calibrate(train_data)
                    calibration_results[model_name] = cal_result
                    
                    # Generate predictions
                    if len(test_data) > 0:
                        initial_state = self._extract_initial_state(test_data.iloc[0])
                        predictions = model.predict(initial_state, len(test_data))
                    else:
                        # Use last training observation
                        initial_state = self._extract_initial_state(train_data.iloc[-1])
                        predictions = model.predict(initial_state, len(rrce_predictions))
                    
                    conventional_predictions[model_name] = predictions
            
            # Compute prediction accuracy
            accuracy_results = self._compute_prediction_accuracy(
                rrce_predictions, conventional_predictions, test_data
            )
            
            # Analyze constraint violations
            constraint_analysis = self._analyze_constraint_violations(rrce_predictions)
            
            # Generate comparison summary
            comparison_summary = self._generate_comparison_summary(
                accuracy_results, constraint_analysis, calibration_results
            )
            
            logger.info("Model comparison analysis completed")
            
            return {
                'rrce_predictions': rrce_predictions,
                'conventional_predictions': conventional_predictions,
                'accuracy_results': accuracy_results,
                'constraint_analysis': constraint_analysis,
                'calibration_results': calibration_results,
                'summary': comparison_summary,
                'data_split': {
                    'train_periods': len(train_data),
                    'test_periods': len(test_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {
                'error': str(e),
                'rrce_predictions': rrce_results.get('predictions', pd.DataFrame()),
                'conventional_predictions': {},
                'summary': {'comparison_failed': True}
            }
    
    def _split_data(self, data: pd.DataFrame, 
                   test_period: Optional[Tuple[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        if test_period:
            start_date, end_date = test_period
            test_mask = (data.index >= start_date) & (data.index <= end_date)
            test_data = data[test_mask]
            train_data = data[~test_mask]
        else:
            # Use last 20% for testing
            split_point = int(len(data) * 0.8)
            train_data = data.iloc[:split_point]
            test_data = data.iloc[split_point:]
        
        return train_data, test_data
    
    def _extract_initial_state(self, data_row: pd.Series) -> Dict[str, Any]:
        """Extract initial state from data row."""
        return {
            'gdp': data_row.get('GDP', data_row.get('gdp', 1000.0)),
            'gini': data_row.get('gini_coefficient', data_row.get('gini', 0.35)),
            'population': data_row.get('population', 1000000)
        }
    
    def _compute_prediction_accuracy(self, rrce_predictions: pd.DataFrame,
                                   conventional_predictions: Dict[str, pd.DataFrame],
                                   test_data: pd.DataFrame) -> Dict[str, Any]:
        """Compute prediction accuracy metrics."""
        accuracy_results = {}
        
        # Variables to compare
        comparison_vars = ['gdp', 'gini']
        
        for var in comparison_vars:
            if var not in rrce_predictions.columns:
                continue
            
            var_results = {}
            
            # Get actual values if available
            actual_values = None
            if len(test_data) > 0:
                actual_col = var.upper() if var == 'gdp' else f'{var}_coefficient'
                if actual_col in test_data.columns:
                    actual_values = test_data[actual_col]
                elif var in test_data.columns:
                    actual_values = test_data[var]
            
            # RRCE accuracy
            rrce_pred = rrce_predictions[var]
            if actual_values is not None and len(actual_values) >= len(rrce_pred):
                var_results['rrce'] = self._compute_metrics(
                    actual_values.iloc[:len(rrce_pred)], rrce_pred
                )
            else:
                var_results['rrce'] = {'note': 'No actual data for comparison'}
            
            # Conventional model accuracy
            for model_name, predictions in conventional_predictions.items():
                if var in predictions.columns:
                    conv_pred = predictions[var]
                    if actual_values is not None and len(actual_values) >= len(conv_pred):
                        var_results[model_name] = self._compute_metrics(
                            actual_values.iloc[:len(conv_pred)], conv_pred
                        )
                    else:
                        var_results[model_name] = {'note': 'No actual data for comparison'}
            
            accuracy_results[var] = var_results
        
        return accuracy_results
    
    def _compute_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Compute accuracy metrics."""
        try:
            # Align series by index to ensure matching labels
            actual, predicted = actual.align(predicted, join='inner')
            # Remove any NaN values
            mask = ~(actual.isna() | predicted.isna())
            actual = actual[mask]
            predicted = predicted[mask]
             
            if len(actual) == 0:
                return {'error': 'No valid data points for comparison'}
            
            # Compute metrics
            metrics = {}
            
            # Root Mean Square Error
            if 'rmse' in self.metrics:
                metrics['rmse'] = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # Mean Absolute Error
            if 'mae' in self.metrics:
                metrics['mae'] = np.mean(np.abs(actual - predicted))
            
            # Mean Absolute Percentage Error
            if 'mape' in self.metrics:
                # Avoid division by zero
                nonzero = actual != 0
                if nonzero.any():
                    metrics['mape'] = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
            
            # Directional Accuracy
            if 'directional_accuracy' in self.metrics and len(actual) > 1:
                actual_direction = np.sign(actual.diff().dropna())
                predicted_direction = np.sign(predicted.diff().dropna())
                # Align directional series
                actual_direction, predicted_direction = actual_direction.align(predicted_direction, join='inner')
                if len(actual_direction) > 0:
                    metrics['directional_accuracy'] = np.mean(actual_direction == predicted_direction)
            
            # R-squared
            if len(actual) > 1:
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to compute metrics: {e}")
            return {'error': str(e)}
    
    def _analyze_constraint_violations(self, rrce_predictions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RRCE constraint violations."""
        analysis = {}
        
        try:
            # Resource constraint violations
            if 'resource_factor' in rrce_predictions.columns:
                resource_violations = (rrce_predictions['resource_factor'] < 0.5).sum()
                analysis['resource_violations'] = {
                    'count': resource_violations,
                    'percentage': resource_violations / len(rrce_predictions) * 100
                }
            
            # Social constraint violations
            if 'gini' in rrce_predictions.columns:
                high_inequality_periods = (rrce_predictions['gini'] > 0.6).sum()
                analysis['social_violations'] = {
                    'count': high_inequality_periods,
                    'percentage': high_inequality_periods / len(rrce_predictions) * 100
                }
            
            # System health analysis
            if 'system_health' in rrce_predictions.columns:
                low_health_periods = (rrce_predictions['system_health'] < 0.5).sum()
                analysis['system_health_issues'] = {
                    'count': low_health_periods,
                    'percentage': low_health_periods / len(rrce_predictions) * 100,
                    'average_health': rrce_predictions['system_health'].mean()
                }
            
            # Equilibrium analysis
            if 'equilibrium_status' in rrce_predictions.columns:
                equilibrium_periods = rrce_predictions['equilibrium_status'].sum()
                analysis['equilibrium_analysis'] = {
                    'equilibrium_periods': equilibrium_periods,
                    'equilibrium_percentage': equilibrium_periods / len(rrce_predictions) * 100
                }
            
        except Exception as e:
            logger.warning(f"Constraint analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _generate_comparison_summary(self, accuracy_results: Dict[str, Any],
                                   constraint_analysis: Dict[str, Any],
                                   calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of comparison results."""
        summary = {
            'rrce_advantages': [],
            'conventional_advantages': [],
            'key_insights': [],
            'recommendations': []
        }
        
        try:
            # Analyze accuracy results
            for var, results in accuracy_results.items():
                if 'rrce' in results and isinstance(results['rrce'], dict):
                    rrce_rmse = results['rrce'].get('rmse', float('inf'))
                    
                    for model_name in ['dsge', 'var']:
                        if model_name in results and isinstance(results[model_name], dict):
                            conv_rmse = results[model_name].get('rmse', float('inf'))
                            
                            if rrce_rmse < conv_rmse:
                                summary['rrce_advantages'].append(
                                    f"Better {var} prediction accuracy vs {model_name.upper()}"
                                )
                            elif conv_rmse < rrce_rmse:
                                summary['conventional_advantages'].append(
                                    f"{model_name.upper()} has better {var} prediction accuracy"
                                )
            
            # Analyze constraint violations
            if 'system_health_issues' in constraint_analysis:
                avg_health = constraint_analysis['system_health_issues'].get('average_health', 0.5)
                if avg_health > 0.7:
                    summary['rrce_advantages'].append("Maintains high system health")
                elif avg_health < 0.3:
                    summary['key_insights'].append("System shows signs of stress")
            
            # Generate insights
            if constraint_analysis.get('equilibrium_analysis', {}).get('equilibrium_percentage', 0) > 80:
                summary['key_insights'].append("System maintains equilibrium in most periods")
            
            # Generate recommendations
            if not summary['rrce_advantages']:
                summary['recommendations'].append("Consider model refinement to improve accuracy")
            
            if constraint_analysis.get('resource_violations', {}).get('percentage', 0) > 20:
                summary['recommendations'].append("Focus on resource sustainability policies")
            
            if constraint_analysis.get('social_violations', {}).get('percentage', 0) > 30:
                summary['recommendations'].append("Address inequality to maintain social stability")
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            summary['error'] = str(e)
        
        return summary