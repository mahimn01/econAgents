"""
Main entry point for RRCE Framework
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main entry point for the RRCE Framework"""
    parser = argparse.ArgumentParser(description='RRCE Framework - Resource-Rich Country Economics')
    parser.add_argument('--config', '-c', default='config/default_config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--countries', default=['US'], nargs='+',
                       help='Countries to analyze')
    parser.add_argument('--start-date', default='2020-01-01',
                       help='Start date for analysis')
    parser.add_argument('--end-date', default='2023-12-31',
                       help='End date for analysis')
    parser.add_argument('--mode', choices=['data', 'simulate', 'full'], default='full',
                       help='Mode: data collection only, simulation only, or full analysis')
    
    args = parser.parse_args()
    
    try:
        from .framework import RRCEFramework
        
        print("Initializing RRCE Framework...")
        
        # Initialize framework
        if Path(args.config).exists():
            rrce = RRCEFramework.from_config(args.config)
        else:
            print(f"Config file {args.config} not found, using default settings")
            rrce = RRCEFramework()
        
        if args.mode == 'data':
            print("Running data collection...")
            results = rrce.collect_data(args.countries, args.start_date, args.end_date)
            print("Data collection completed!")
            
        elif args.mode == 'simulate':
            print("Running simulation...")
            for country in args.countries:
                print(f"Simulating {country}...")
                results = rrce.simulate(country, args.start_date, args.end_date)
            print("Simulation completed!")
            
        else:  # full analysis
            print("Running full analysis...")
            results = rrce.run_full_analysis(countries=args.countries)
            print("Full analysis completed!")
            
        print("RRCE Framework execution finished successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed and the framework is properly configured.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running RRCE Framework: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()