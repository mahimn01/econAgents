"""
Analysis metrics for RRCE Framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class RRCEMetrics:
    """
    Comprehensive metrics for analyzing RRCE model performance.
    """
    
    @staticmethod
    def sustainability_index(predictions: pd.DataFrame) -> Dict[str, float]:
        """
        Compute sustainability index based on multiple factors.
        
        Args:
            predictions: RRCE model predictions
            
        Returns:
            Dictionary with sustainability metrics
        """
        try:
            sustainability = {}
            
            # Resource sustainability
            if 'resource_factor' in predictions.columns:
                resource_sustainability = predictions['resource_factor'].mean()
                sustainability['resource_sustainability'] = resource_sustainability
            
            # Social sustainability
            if 'social_factor' in predictions.columns:
                social_sustainability = predictions['social_factor'].mean()
                sustainability['social_sustainability'] = social_sustainability
            
            # Economic sustainability (stable growth)
            if 'gdp' in predictions.columns:
                gdp_growth = predictions['gdp'].pct_change().dropna()
                growth_stability = 1.0 / (1.0 + gdp_growth.std()) if len(gdp_growth) > 0 else 0.5
                sustainability['economic_sustainability'] = growth_stability
            
            # Overall sustainability index
            components = [
                sustainability.get('resource_sustainability', 0.5),
                sustainability.get('social_sustainability', 0.5),
                sustainability.get('economic_sustainability', 0.5)
            ]
            sustainability['overall_index'] = np.mean(components)
            
            return sustainability
            
        except Exception as e:
            logger.error(f"Sustainability index computation failed: {e}")
            return {'overall_index': 0.5, 'error': str(e)}
    
    @staticmethod
    def constraint_violation_analysis(predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze constraint violations in detail.
        
        Args:
            predictions: RRCE model predictions
            
        Returns:
            Detailed constraint violation analysis
        """
        try:
            analysis = {}
            
            # Resource constraint violations
            if 'resource_factor' in predictions.columns:
                resource_factor = predictions['resource_factor']
                analysis['resource_constraints'] = {
                    'severe_violations': (resource_factor < 0.3).sum(),
                    'moderate_violations': ((resource_factor >= 0.3) & (resource_factor < 0.5)).sum(),
                    'safe_periods': (resource_factor >= 0.5).sum(),
                    'minimum_factor': resource_factor.min(),
                    'average_factor': resource_factor.mean()
                }
            
            # Social constraint violations
            if 'gini' in predictions.columns:
                gini = predictions['gini']
                analysis['social_constraints'] = {
                    'high_inequality': (gini > 0.6).sum(),
                    'moderate_inequality': ((gini > 0.4) & (gini <= 0.6)).sum(),
                    'low_inequality': (gini <= 0.4).sum(),
                    'maximum_gini': gini.max(),
                    'average_gini': gini.mean()
                }
            
            # System health violations
            if 'system_health' in predictions.columns:
                health = predictions['system_health']
                analysis['system_health'] = {
                    'poor_health': (health < 0.3).sum(),
                    'moderate_health': ((health >= 0.3) & (health < 0.7)).sum(),
                    'good_health': (health >= 0.7).sum(),
                    'minimum_health': health.min(),
                    'average_health': health.mean()
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Constraint violation analysis failed: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def economic_stability_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
        """
        Compute economic stability metrics.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Economic stability metrics
        """
        try:
            metrics = {}
            
            # GDP stability
            if 'gdp' in predictions.columns:
                gdp_growth = predictions['gdp'].pct_change().dropna()
                if len(gdp_growth) > 0:
                    metrics['gdp_volatility'] = gdp_growth.std()
                    metrics['average_growth'] = gdp_growth.mean()
                    metrics['growth_stability'] = 1.0 / (1.0 + gdp_growth.std())
            
            # Currency stability
            if 'cwu_value' in predictions.columns:
                cwu_changes = predictions['cwu_value'].pct_change().dropna()
                if len(cwu_changes) > 0:
                    metrics['currency_volatility'] = cwu_changes.std()
                    metrics['currency_stability'] = 1.0 / (1.0 + cwu_changes.std())
            
            # Inequality stability
            if 'gini' in predictions.columns:
                gini_changes = predictions['gini'].diff().dropna()
                if len(gini_changes) > 0:
                    metrics['inequality_volatility'] = gini_changes.std()
                    metrics['inequality_stability'] = 1.0 / (1.0 + abs(gini_changes.std()))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Economic stability metrics computation failed: {e}")
            return {'error': str(e)}