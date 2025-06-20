"""
RRCE Framework: Complete Project Structure Setup
===============================================

This sets up the complete directory structure and files for the RRCE project.
Run this script to create all necessary files and directories.
"""

import os
from pathlib import Path
import json

def create_project_structure():
    """Create the complete RRCE project structure."""
    
    # Define the project structure
    structure = {
        "rrce_framework": {
            "__init__.py": "",
            "core": {
                "__init__.py": "",
                "mathematics": {
                    "__init__.py": "",
                    "resource_dynamics.py": "",
                    "pricing_mechanisms.py": "",
                    "currency_system.py": "",
                    "equilibrium_solver.py": "",
                },
                "simulation": {
                    "__init__.py": "",
                    "simulator.py": "",
                    "system_state.py": "",
                    "agents.py": "",
                },
                "data": {
                    "__init__.py": "",
                    "collectors.py": "",  # This will be our data infrastructure
                    "processors.py": "",
                    "validators.py": "",
                },
                "models": {
                    "__init__.py": "",
                    "conventional": {
                        "__init__.py": "",
                        "dsge.py": "",
                        "var_model.py": "",
                    },
                    "rrce_model.py": "",
                },
                "analysis": {
                    "__init__.py": "",
                    "comparisons.py": "",
                    "metrics.py": "",
                    "visualization.py": "",
                },
                "utils": {
                    "__init__.py": "",
                    "config.py": "",
                    "logger.py": "",
                    "helpers.py": "",
                }
            }
        },
        "tests": {
            "__init__.py": "",
            "test_data_collection.py": "",
            "test_mathematics.py": "",
            "test_simulation.py": "",
            "test_integration.py": "",
        },
        "scripts": {
            "run_data_collection.py": "",
            "run_simulation.py": "",
            "run_analysis.py": "",
            "setup_environment.py": "",
        },
        "config": {
            "default_config.yaml": "",
            "countries.yaml": "",
            "data_sources.yaml": "",
        },
        "data": {
            "raw": {},
            "processed": {},
            "cache": {},
            "results": {},
        },
        "docs": {
            "README.md": "",
            "API.md": "",
            "mathematical_foundation.md": "",
        },
        "notebooks": {
            "01_data_exploration.ipynb": "",
            "02_model_validation.ipynb": "",
            "03_comparative_analysis.ipynb": "",
        }
    }
    
    def create_directory_structure(base_path: Path, structure: dict):
        """Recursively create directory structure."""
        for name, content in structure.items():
            current_path = base_path / name
            
            if isinstance(content, dict):
                # It's a directory
                current_path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {current_path}")
                create_directory_structure(current_path, content)
            else:
                # It's a file
                current_path.parent.mkdir(parents=True, exist_ok=True)
                if not current_path.exists():
                    current_path.touch()
                    print(f"Created file: {current_path}")
    
    # Create the structure
    base_path = Path(".")
    create_directory_structure(base_path, structure)
    print("✅ Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()