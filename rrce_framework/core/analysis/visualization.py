"""
Enterprise-level visualization and reporting for RRCE Framework.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class RRCEVisualizer:
    """
    Enterprise-level visualization and reporting system for RRCE Framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RRCE visualizer."""
        self.config = config or {}
        self.theme = self.config.get('theme', 'plotly_white')
        self.color_palette = {
            'economic': '#1f77b4',
            'social': '#ff7f0e', 
            'environmental': '#2ca02c',
            'resource': '#d62728',
            'sustainability': '#9467bd',
            'warning': '#ff6b6b',
            'success': '#51cf66'
        }
        
        logger.info("Initialized RRCE Visualizer")
    
    def create_comprehensive_dashboard(self, predictions: pd.DataFrame, 
                                    initial_state: Dict[str, Any],
                                    country: str = "Country") -> Dict[str, Any]:
        """
        Create comprehensive dashboard with all RRCE indicators.
        """
        logger.info(f"Creating comprehensive dashboard for {country}")
        
        if not PLOTLY_AVAILABLE:
            return self._create_matplotlib_dashboard(predictions, country)
        
        dashboard = {
            'overview_plot': self._create_overview_plot(predictions, country),
            'economic_indicators': self._create_economic_dashboard(predictions, country),
            'sustainability_analysis': self._create_sustainability_dashboard(predictions, country),
            'executive_summary': self._generate_executive_summary(predictions, initial_state, country)
        }
        
        return dashboard
    
    def _create_overview_plot(self, predictions: pd.DataFrame, country: str):
        """Create high-level overview plot."""
        if not PLOTLY_AVAILABLE:
            return self._create_matplotlib_overview(predictions, country)
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Economic Growth (GDP)', 'System Health', 
                          'Sustainability Index', 'Currency Value (CWU)'),
        )
        
        # GDP trajectory
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['gdp'],
                      name='GDP', line=dict(color=self.color_palette['economic'], width=3)),
            row=1, col=1
        )
        
        # System Health
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['system_health'],
                      name='System Health', line=dict(color=self.color_palette['success'], width=3)),
            row=1, col=2
        )
        
        # Sustainability Index
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['sustainability_index'],
                      name='Sustainability', line=dict(color=self.color_palette['sustainability'], width=3)),
            row=2, col=1
        )
        
        # Currency Value
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['cwu_value'],
                      name='CWU Value', line=dict(color=self.color_palette['resource'], width=3)),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'RRCE Framework Overview - {country}',
            height=600,
            showlegend=False,
            template=self.theme
        )
        
        return fig
    
    def _create_economic_dashboard(self, predictions: pd.DataFrame, country: str):
        """Create detailed economic indicators dashboard."""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('GDP Growth Trajectory', 'Productivity Trends',
                          'Income Inequality (Gini)', 'Social Cohesion'),
        )
        
        # GDP
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['gdp'],
                      name='GDP', line=dict(color=self.color_palette['economic'])),
            row=1, col=1
        )
        
        # Productivity
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['productivity'],
                      name='Productivity', line=dict(color='green'), fill='tozeroy'),
            row=1, col=2
        )
        
        # Gini coefficient
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['gini_coefficient'],
                      name='Gini Coefficient', line=dict(color=self.color_palette['social'])),
            row=2, col=1
        )
        
        # Social cohesion
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['social_cohesion'],
                      name='Social Cohesion', line=dict(color='blue'), fill='tozeroy'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Economic Indicators Dashboard - {country}',
            height=700,
            template=self.theme
        )
        
        return fig
    
    def _create_sustainability_dashboard(self, predictions: pd.DataFrame, country: str):
        """Create comprehensive sustainability dashboard."""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sustainability Index', 'Environmental Quality',
                          'System Health', 'Currency Stability'),
        )
        
        # Sustainability index
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['sustainability_index'],
                      name='Sustainability Index',
                      line=dict(color=self.color_palette['sustainability'], width=4),
                      fill='tozeroy'),
            row=1, col=1
        )
        
        # Environmental quality
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['environmental_quality'],
                      name='Environmental Quality',
                      line=dict(color=self.color_palette['environmental'], width=3)),
            row=1, col=2
        )
        
        # System health
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['system_health'],
                      name='System Health',
                      line=dict(color='blue', width=3)),
            row=2, col=1
        )
        
        # Currency value
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions['cwu_value'],
                      name='CWU Value',
                      line=dict(color=self.color_palette['resource'], width=3)),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Sustainability Analysis Dashboard - {country}',
            height=700,
            template=self.theme
        )
        
        return fig
        
    def _create_matplotlib_dashboard(self, predictions: pd.DataFrame, country: str) -> Dict[str, Any]:
        """Create dashboard using matplotlib when plotly is not available."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'RRCE Framework Analysis - {country}', fontsize=16)
        
        # GDP
        axes[0,0].plot(predictions.index, predictions['gdp'], 'b-', linewidth=2)
        axes[0,0].set_title('GDP Growth')
        axes[0,0].set_ylabel('GDP')
        
        # Sustainability Index
        axes[0,1].plot(predictions.index, predictions['sustainability_index'], 'g-', linewidth=2)
        axes[0,1].set_title('Sustainability Index')
        axes[0,1].set_ylabel('Index')
        
        # Gini Coefficient
        axes[1,0].plot(predictions.index, predictions['gini_coefficient'], 'r-', linewidth=2)
        axes[1,0].set_title('Income Inequality (Gini)')
        axes[1,0].set_ylabel('Gini Coefficient')
        
        # System Health
        axes[1,1].plot(predictions.index, predictions['system_health'], 'purple', linewidth=2)
        axes[1,1].set_title('System Health')
        axes[1,1].set_ylabel('Health Index')
        
        plt.tight_layout()
        return {'matplotlib_figure': fig}
    
    def _create_matplotlib_overview(self, predictions: pd.DataFrame, country: str):
        """Create overview plot using matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(predictions.index, predictions['gdp'], label='GDP', linewidth=2)
        ax.plot(predictions.index, predictions['sustainability_index'] * predictions['gdp'].max(), 
                label='Sustainability Index (scaled)', linewidth=2)
        
        ax.set_title(f'RRCE Overview - {country}')
        ax.set_xlabel('Time Period')
        ax.legend()
        plt.grid(True, alpha=0.3)
        
        return fig
    
    def _generate_executive_summary(self, predictions: pd.DataFrame, 
                                  initial_state: Dict[str, Any], 
                                  country: str) -> Dict[str, Any]:
        """Generate executive summary with key insights."""
        last_period = predictions.iloc[-1]
        first_period = predictions.iloc[0]
        
        # Calculate key metrics
        gdp_growth = ((last_period['gdp'] - first_period['gdp']) / first_period['gdp']) * 100
        sustainability_change = last_period['sustainability_index'] - first_period['sustainability_index']
        
        summary = {
            'country': country,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'time_horizon': len(predictions),
            'key_metrics': {
                'total_gdp_growth': f"{gdp_growth:.1f}%",
                'final_sustainability_index': f"{last_period['sustainability_index']:.3f}",
                'final_gini_coefficient': f"{last_period['gini_coefficient']:.3f}",
                'final_environmental_quality': f"{last_period['environmental_quality']:.3f}"
            },
            'trends': {
                'sustainability': "improving" if sustainability_change > 0 else "declining",
                'gdp_growth_rate': f"{gdp_growth/len(predictions):.2f}% per period"
            },
            'recommendations': self._generate_recommendations(last_period)
        }
        
        return summary
    
    def _generate_recommendations(self, last_period: pd.Series) -> List[str]:
        """Generate policy recommendations."""
        recommendations = []
        
        if last_period['sustainability_index'] < 0.5:
            recommendations.append("Implement comprehensive sustainability policies")
        
        if last_period['gini_coefficient'] > 0.4:
            recommendations.append("Address income inequality through redistributive policies")
        
        if last_period['environmental_quality'] < 0.4:
            recommendations.append("Strengthen environmental protection measures")
        
        if not recommendations:
            recommendations.append("Continue monitoring key indicators")
        
        return recommendations
    
    def create_dashboard(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create dashboard from results."""
        if 'predictions' in results:
            return self.create_comprehensive_dashboard(
                results['predictions'], 
                results.get('initial_state', {}),
                results.get('country', 'Unknown')
            )
        else:
            return {'error': 'No predictions data available'}
    
    def generate_report(self, results: Dict[str, Any], output_path: str) -> bool:
        """Generate comprehensive report."""
        try:
            dashboard = self.create_dashboard(results)
            logger.info(f"Generated dashboard with {len(dashboard)} components")
            return True
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return False


class RRCEMetrics:
    """Compatibility class for metrics."""
    
    @staticmethod
    def sustainability_index(predictions: pd.DataFrame) -> Dict[str, float]:
        """Compute sustainability metrics."""
        try:
            if 'sustainability_index' in predictions.columns:
                return {
                    'overall_sustainability': predictions['sustainability_index'].mean(),
                    'sustainability_trend': predictions['sustainability_index'].iloc[-1] - predictions['sustainability_index'].iloc[0]
                }
            else:
                return {'overall_sustainability': 0.5, 'sustainability_trend': 0.0}
        except Exception as e:
            logger.error(f"Error computing sustainability index: {e}")
            return {'overall_sustainability': 0.0, 'sustainability_trend': 0.0}
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