"""
RRCE Framework: Resource-Reality Coupled Economics
==================================================

A comprehensive framework for modeling economic systems under physical, 
environmental, and social constraints.
"""

__version__ = "0.1.0"
__author__ = "RRCE Development Team"
__email__ = "contact@rrce-framework.org"

# Core imports
from .core.utils.config import Config
from .core.data.collectors import RRCEDataManager, DataConfig
from .core.simulation.simulator import RRCESimulator
from .core.models.rrce_model import RRCEModel
from .core.analysis.comparisons import ModelComparison

# Main framework class
from .framework import RRCEFramework

# Make key classes available at package level
__all__ = [
    'RRCEFramework',
    'Config', 
    'RRCEDataManager',
    'DataConfig',
    'RRCESimulator', 
    'RRCEModel',
    'ModelComparison',
]
