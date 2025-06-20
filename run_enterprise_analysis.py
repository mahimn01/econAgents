"""
Enterprise-level analysis script for RRCE Framework.
Run comprehensive economic simulations with mathematical rigor.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from rrce_framework import RRCEFramework

def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/rrce_analysis.log'),
            logging.StreamHandler()
        ]
    )

def run_comprehensive_analysis(countries=['US'], output_dir='results'):
    """
    Run comprehensive RRCE analysis with enterprise-level output.
    
    Args:
        countries: List of country codes to analyze
        output_dir: Directory for results output
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive RRCE analysis")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize RRCE Framework
    rrce = RRCEFramework()
    
    # Setup data collection with API keys
    api_keys = {
        'fred_api_key': '2f4a85d0940055c0ccbd5d8fdf6fc481'  # Your FRED API key
    }
    rrce.setup_data_collection(api_keys)
    
    all_results = {}
    
    for country in countries:
        logger.info(f"\\n{'='*50}")
        logger.info(f"ANALYZING: {country}")
        logger.info(f"{'='*50}")
        
        try:
            # Run full analysis pipeline
            results = rrce.run_full_analysis(countries=[country], api_keys=api_keys)
            
            if country in results:
                country_results = results[country]
                
                # Generate comprehensive outputs
                analysis_summary = generate_analysis_summary(country_results, country)
                
                # Save detailed results
                save_detailed_results(country_results, country, output_path)
                
                # Print executive summary
                print_executive_summary(analysis_summary, country)
                
                all_results[country] = {
                    'analysis_summary': analysis_summary,
                    'detailed_results': country_results
                }
                
        except Exception as e:
            logger.error(f"Analysis failed for {country}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save consolidated results
    save_consolidated_results(all_results, output_path)
    
    logger.info(f"\\nAnalysis complete. Results saved to {output_path}")
    return all_results

def generate_analysis_summary(results, country):
    """Generate comprehensive analysis summary."""
    
    predictions = results.get('predictions', pd.DataFrame())
    initial_state = results.get('initial_state', {})
    
    if predictions.empty:
        return {'error': 'No predictions available'}
    
    # Time horizon
    time_horizon = len(predictions)
    
    # Key metrics from final period
    final_period = predictions.iloc[-1]
    initial_period = predictions.iloc[0]
    
    # Calculate changes
    gdp_change = ((final_period['gdp'] - initial_period['gdp']) / initial_period['gdp']) * 100
    sustainability_change = final_period['sustainability_index'] - initial_period['sustainability_index']
    inequality_change = final_period['gini_coefficient'] - initial_period['gini_coefficient']
    
    # Performance metrics
    avg_growth_rate = gdp_change / time_horizon
    sustainability_trend = "Improving" if sustainability_change > 0 else "Declining"
    inequality_trend = "Worsening" if inequality_change > 0 else "Improving"
    
    # Risk assessment
    risk_level = assess_risk_level(final_period)
    
    # Mathematical framework validation
    framework_validation = validate_mathematical_framework(predictions)
    
    summary = {
        'country': country,
        'analysis_timestamp': datetime.now().isoformat(),
        'time_horizon_periods': time_horizon,
        'time_horizon_years': time_horizon * 0.25,  # Assuming quarterly periods
        
        # Economic indicators
        'economic_performance': {
            'initial_gdp': float(initial_period['gdp']),
            'final_gdp': float(final_period['gdp']),
            'total_gdp_growth_percent': round(gdp_change, 2),
            'average_quarterly_growth_percent': round(avg_growth_rate, 3),
            'annualized_growth_percent': round(avg_growth_rate * 4, 2)
        },
        
        # Social indicators
        'social_performance': {
            'initial_gini': float(initial_period['gini_coefficient']),
            'final_gini': float(final_period['gini_coefficient']),
            'inequality_change': round(inequality_change, 3),
            'inequality_trend': inequality_trend,
            'final_social_cohesion': float(final_period['social_cohesion'])
        },
        
        # Environmental indicators
        'environmental_performance': {
            'initial_env_quality': float(initial_period['environmental_quality']),
            'final_env_quality': float(final_period['environmental_quality']),
            'environmental_change': round(final_period['environmental_quality'] - initial_period['environmental_quality'], 3)
        },
        
        # Sustainability metrics
        'sustainability_metrics': {
            'initial_index': float(initial_period['sustainability_index']),
            'final_index': float(final_period['sustainability_index']),
            'sustainability_change': round(sustainability_change, 3),
            'sustainability_trend': sustainability_trend,
            'average_sustainability': float(predictions['sustainability_index'].mean())
        },
        
        # System health
        'system_health': {
            'final_system_health': float(final_period['system_health']),
            'average_system_health': float(predictions['system_health'].mean()),
            'health_trend': "Improving" if final_period['system_health'] > initial_period['system_health'] else "Declining"
        },
        
        # Currency system
        'currency_performance': {
            'initial_cwu_value': float(initial_period['cwu_value']),
            'final_cwu_value': float(final_period['cwu_value']),
            'currency_appreciation_percent': round(((final_period['cwu_value'] - initial_period['cwu_value']) / initial_period['cwu_value']) * 100, 2)
        },
        
        # Risk assessment
        'risk_assessment': risk_level,
        
        # Mathematical framework validation
        'framework_validation': framework_validation,
        
        # Policy recommendations
        'recommendations': generate_policy_recommendations(final_period, sustainability_change, inequality_change)
    }
    
    return summary

def assess_risk_level(final_period):
    """Assess overall risk level based on key indicators."""
    risk_factors = []
    risk_score = 0
    
    # Sustainability risk
    if final_period['sustainability_index'] < 0.3:
        risk_factors.append("Critical sustainability level")
        risk_score += 3
    elif final_period['sustainability_index'] < 0.5:
        risk_factors.append("Low sustainability")
        risk_score += 2
    
    # Inequality risk
    if final_period['gini_coefficient'] > 0.5:
        risk_factors.append("Extreme inequality")
        risk_score += 3
    elif final_period['gini_coefficient'] > 0.4:
        risk_factors.append("High inequality")
        risk_score += 2
    
    # Environmental risk
    if final_period['environmental_quality'] < 0.2:
        risk_factors.append("Environmental crisis")
        risk_score += 3
    elif final_period['environmental_quality'] < 0.4:
        risk_factors.append("Environmental degradation")
        risk_score += 2
    
    # System health risk
    if final_period['system_health'] < 0.3:
        risk_factors.append("Poor system health")
        risk_score += 2
    
    # Determine overall risk level
    if risk_score >= 6:
        risk_level = "HIGH"
    elif risk_score >= 3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        'overall_risk_level': risk_level,
        'risk_score': risk_score,
        'risk_factors': risk_factors
    }

def validate_mathematical_framework(predictions):
    """Validate that mathematical framework is working correctly."""
    validation = {
        'conservation_laws': True,
        'equilibrium_conditions': True,
        'stability_checks': True,
        'issues': []
    }
    
    try:
        # Check for conservation violations (no negative resources)
        resource_cols = [col for col in predictions.columns if '_availability' in col]
        if resource_cols:
            for col in resource_cols:
                if (predictions[col] < 0).any():
                    validation['conservation_laws'] = False
                    validation['issues'].append(f"Negative resource levels detected in {col}")
        
        # Check for equilibrium stability (no extreme volatility)
        gdp_volatility = predictions['gdp'].pct_change().std()
        if gdp_volatility > 0.1:  # 10% volatility threshold
            validation['equilibrium_conditions'] = False
            validation['issues'].append(f"High GDP volatility detected: {gdp_volatility:.3f}")
        
        # Check sustainability bounds
        if (predictions['sustainability_index'] < 0).any() or (predictions['sustainability_index'] > 1).any():
            validation['stability_checks'] = False
            validation['issues'].append("Sustainability index out of bounds [0,1]")
        
        # Overall validation
        validation['overall_valid'] = all([
            validation['conservation_laws'],
            validation['equilibrium_conditions'],
            validation['stability_checks']
        ])
        
    except Exception as e:
        validation['overall_valid'] = False
        validation['issues'].append(f"Validation error: {str(e)}")
    
    return validation

def generate_policy_recommendations(final_period, sustainability_change, inequality_change):
    """Generate specific policy recommendations based on analysis."""
    recommendations = []
    
    # Sustainability recommendations
    if final_period['sustainability_index'] < 0.5:
        recommendations.append({
            'category': 'Sustainability',
            'priority': 'HIGH',
            'recommendation': 'Implement RRCE framework principles to improve resource management and sustainability',
            'specific_actions': [
                'Transition to resource-backed currency (CWU)',
                'Implement carrying capacity constraints',
                'Establish sustainability targets'
            ]
        })
    
    # Inequality recommendations
    if final_period['gini_coefficient'] > 0.4:
        recommendations.append({
            'category': 'Social Policy',
            'priority': 'HIGH',
            'recommendation': 'Address income inequality through redistributive policies',
            'specific_actions': [
                'Progressive taxation system',
                'Universal basic services',
                'Resource dividend distribution'
            ]
        })
    
    # Environmental recommendations
    if final_period['environmental_quality'] < 0.4:
        recommendations.append({
            'category': 'Environmental',
            'priority': 'CRITICAL',
            'recommendation': 'Immediate environmental protection and restoration measures',
            'specific_actions': [
                'Carbon pricing mechanisms',
                'Environmental restoration programs',
                'Green technology investment'
            ]
        })
    
    # Economic recommendations
    if sustainability_change < -0.1:
        recommendations.append({
            'category': 'Economic Policy',
            'priority': 'MEDIUM',
            'recommendation': 'Restructure economic model for long-term sustainability',
            'specific_actions': [
                'Circular economy principles',
                'Sustainable development goals integration',
                'Green investment incentives'
            ]
        })
    
    return recommendations

def save_detailed_results(results, country, output_path):
    """Save detailed results to files."""
    country_dir = output_path / country
    country_dir.mkdir(exist_ok=True)
    
    # Save predictions as CSV
    if 'predictions' in results and not results['predictions'].empty:
        results['predictions'].to_csv(country_dir / 'predictions.csv')
    
    # Save diagnostics as JSON
    if 'diagnostics' in results:
        with open(country_dir / 'diagnostics.json', 'w') as f:
            json.dump(results['diagnostics'], f, indent=2, default=str)
    
    # Save initial state
    if 'initial_state' in results:
        with open(country_dir / 'initial_state.json', 'w') as f:
            json.dump(results['initial_state'], f, indent=2, default=str)

def save_consolidated_results(all_results, output_path):
    """Save consolidated results across all countries."""
    
    # Save summary comparison
    summary_data = {}
    for country, results in all_results.items():
        if 'analysis_summary' in results:
            summary_data[country] = results['analysis_summary']
    
    with open(output_path / 'consolidated_analysis.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    # Create comparison CSV
    comparison_data = []
    for country, results in all_results.items():
        if 'analysis_summary' in results:
            summary = results['analysis_summary']
            comparison_data.append({
                'country': country,
                'final_gdp': summary.get('economic_performance', {}).get('final_gdp', 0),
                'gdp_growth_percent': summary.get('economic_performance', {}).get('total_gdp_growth_percent', 0),
                'final_sustainability': summary.get('sustainability_metrics', {}).get('final_index', 0),
                'final_gini': summary.get('social_performance', {}).get('final_gini', 0),
                'environmental_quality': summary.get('environmental_performance', {}).get('final_env_quality', 0),
                'risk_level': summary.get('risk_assessment', {}).get('overall_risk_level', 'UNKNOWN')
            })
    
    if comparison_data:
        pd.DataFrame(comparison_data).to_csv(output_path / 'country_comparison.csv', index=False)

def print_executive_summary(summary, country):
    """Print executive summary to console."""
    print(f"\\n{'='*60}")
    print(f"RRCE FRAMEWORK ANALYSIS - {country}")
    print(f"{'='*60}")
    
    print(f"\\n📊 ECONOMIC PERFORMANCE:")
    econ = summary.get('economic_performance', {})
    print(f"  • GDP Growth: {econ.get('total_gdp_growth_percent', 0):.1f}% total")
    print(f"  • Annualized Growth: {econ.get('annualized_growth_percent', 0):.1f}%")
    print(f"  • Final GDP: ${econ.get('final_gdp', 0):,.0f} billion")
    
    print(f"\\n🌱 SUSTAINABILITY METRICS:")
    sustain = summary.get('sustainability_metrics', {})
    print(f"  • Final Sustainability Index: {sustain.get('final_index', 0):.3f}")
    print(f"  • Trend: {sustain.get('sustainability_trend', 'Unknown')}")
    print(f"  • Change: {sustain.get('sustainability_change', 0):+.3f}")
    
    print(f"\\n🏛️ SOCIAL PERFORMANCE:")
    social = summary.get('social_performance', {})
    print(f"  • Final Gini Coefficient: {social.get('final_gini', 0):.3f}")
    print(f"  • Inequality Trend: {social.get('inequality_trend', 'Unknown')}")
    print(f"  • Social Cohesion: {social.get('final_social_cohesion', 0):.3f}")
    
    print(f"\\n🌍 ENVIRONMENTAL STATUS:")
    env = summary.get('environmental_performance', {})
    print(f"  • Environmental Quality: {env.get('final_env_quality', 0):.3f}")
    print(f"  • Change: {env.get('environmental_change', 0):+.3f}")
    
    print(f"\\n⚠️ RISK ASSESSMENT:")
    risk = summary.get('risk_assessment', {})
    print(f"  • Overall Risk Level: {risk.get('overall_risk_level', 'UNKNOWN')}")
    print(f"  • Risk Score: {risk.get('risk_score', 0)}/9")
    if risk.get('risk_factors'):
        print(f"  • Risk Factors: {', '.join(risk['risk_factors'])}")
    
    print(f"\\n💱 CURRENCY PERFORMANCE:")
    currency = summary.get('currency_performance', {})
    print(f"  • CWU Appreciation: {currency.get('currency_appreciation_percent', 0):+.1f}%")
    print(f"  • Final CWU Value: {currency.get('final_cwu_value', 1.0):.3f}")
    
    print(f"\\n🔬 FRAMEWORK VALIDATION:")
    validation = summary.get('framework_validation', {})
    print(f"  • Mathematical Framework Valid: {validation.get('overall_valid', False)}")
    if validation.get('issues'):
        print(f"  • Issues: {len(validation['issues'])} detected")
    
    print(f"\\n📋 TOP RECOMMENDATIONS:")
    recommendations = summary.get('recommendations', [])
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. [{rec.get('priority', 'MEDIUM')}] {rec.get('recommendation', 'No recommendation')}")

if __name__ == "__main__":
    setup_logging()
    
    # Run analysis for specified countries
    countries = ['US']  # Add more countries as needed: ['US', 'CA', 'DE', 'GB']
    
    results = run_comprehensive_analysis(countries, 'results')
    
    print(f"\\n🎉 Analysis complete! Check the 'results' directory for detailed outputs.")
