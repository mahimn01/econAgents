"""
System state representation for RRCE simulations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

@dataclass
class SystemState:
    """Represents the complete state of the RRCE system at a point in time."""
    
    # Time information
    time: float
    date: Optional[pd.Timestamp] = None
    
    # Resource state
    resources: Dict[str, float] = None  # Resource availability levels
    resource_flows: Dict[str, float] = None  # Resource flow rates
    carrying_capacity: Dict[str, float] = None  # Current carrying capacities
    
    # Economic state
    gdp: float = 0.0
    prices: Dict[str, float] = None  # Goods and resource prices
    currency_value: float = 1.0  # CWU value
    
    # Environmental state
    environmental_quality: Dict[str, float] = None
    emissions: Dict[str, float] = None
    
    # Social state
    gini_coefficient: float = 0.0
    inequality_measures: Dict[str, float] = None
    
    # System health indicators
    equilibrium_status: Dict[str, bool] = None  # Which constraints are satisfied
    stability_indicators: Dict[str, float] = None
    
    # Pricing factors
    rcm_factors: Dict[str, float] = None  # Resource Criticality Multipliers
    sif_factors: Dict[str, float] = None  # System Impact Factors
    ssc_factors: Dict[str, float] = None  # Social Stability Coefficients
    
    def __post_init__(self):
        """Initialize default values for None fields."""
        if self.resources is None:
            self.resources = {}
        if self.resource_flows is None:
            self.resource_flows = {}
        if self.carrying_capacity is None:
            self.carrying_capacity = {}
        if self.prices is None:
            self.prices = {}
        if self.environmental_quality is None:
            self.environmental_quality = {}
        if self.emissions is None:
            self.emissions = {}
        if self.inequality_measures is None:
            self.inequality_measures = {}
        if self.equilibrium_status is None:
            self.equilibrium_status = {
                'resource_balance': False,
                'capacity_constraint': False,
                'social_stability': False
            }
        if self.stability_indicators is None:
            self.stability_indicators = {}
        if self.rcm_factors is None:
            self.rcm_factors = {}
        if self.sif_factors is None:
            self.sif_factors = {}
        if self.ssc_factors is None:
            self.ssc_factors = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'time': self.time,
            'date': self.date.isoformat() if self.date else None,
            'resources': self.resources,
            'resource_flows': self.resource_flows,
            'carrying_capacity': self.carrying_capacity,
            'gdp': self.gdp,
            'prices': self.prices,
            'currency_value': self.currency_value,
            'environmental_quality': self.environmental_quality,
            'emissions': self.emissions,
            'gini_coefficient': self.gini_coefficient,
            'inequality_measures': self.inequality_measures,
            'equilibrium_status': self.equilibrium_status,
            'stability_indicators': self.stability_indicators,
            'rcm_factors': self.rcm_factors,
            'sif_factors': self.sif_factors,
            'ssc_factors': self.ssc_factors,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """Create from dictionary."""
        if data.get('date'):
            data['date'] = pd.Timestamp(data['date'])
        return cls(**data)
    
    def copy(self) -> 'SystemState':
        """Create a deep copy of the system state."""
        return SystemState.from_dict(self.to_dict())