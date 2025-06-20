"""
Configuration management for RRCE Framework.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataConfig:
    """Data collection configuration."""
    default_start_date: str = "2000-01-01"
    default_end_date: str = "2023-12-31"
    sources: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimulationConfig:
    """Simulation configuration."""
    time_step: float = 0.25
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    solver: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self):
        """Return a copy of the configuration"""
        import copy
        return copy.deepcopy(self)
    
    # Add dictionary-like access methods
    def get(self, key, default=None):
        """Get attribute like a dictionary"""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Allow dictionary-style assignment"""
        setattr(self, key, value)

@dataclass
class ModelConfig:
    """Model parameters configuration."""
    resources: Dict[str, Any] = field(default_factory=dict)
    social: Dict[str, Any] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    pricing: Dict[str, Any] = field(default_factory=dict)
    currency: Dict[str, Any] = field(default_factory=dict)
    equilibrium: Dict[str, Any] = field(default_factory=dict)
    time_step: float = 0.25  # Quarterly time step
    
    def copy(self):
        """Return a copy of the configuration"""
        import copy
        return copy.deepcopy(self)

@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    conventional_models: list = field(default_factory=lambda: ["dsge", "var"])
    metrics: list = field(default_factory=lambda: ["rmse", "mae"])
    validation: Dict[str, Any] = field(default_factory=dict)
    
    # Add dictionary-like access methods
    def get(self, key, default=None):
        """Get attribute like a dictionary"""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Allow dictionary-style assignment"""
        setattr(self, key, value)

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/rrce_framework.log"
    max_file_size_mb: int = 10
    backup_count: int = 5

    # Add dictionary-like access methods
    def get(self, key, default=None):
        """Get attribute like a dictionary"""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Allow dictionary-style assignment"""
        setattr(self, key, value)
    
    def keys(self):
        """Return attribute names"""
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]
    
    def items(self):
        """Return key-value pairs"""
        return [(key, getattr(self, key)) for key in self.keys()]


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    default_theme: str = "plotly_white"
    figure_size: list = field(default_factory=lambda: [12, 8])
    dpi: int = 300
    save_format: str = "png"
    interactive: bool = True
    
    # Add dictionary-like access methods
    def get(self, key, default=None):
        """Get attribute like a dictionary"""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Allow dictionary-style assignment"""
        setattr(self, key, value)

class Config:
    """Main configuration class for RRCE Framework."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary."""
        self._config = config_dict
        
        # Parse configuration sections
        self.data = DataConfig(**config_dict.get('data', {}))
        self.simulation = SimulationConfig(**config_dict.get('simulation', {}))
        self.model = ModelConfig(**config_dict.get('model', {}))
        self.analysis = AnalysisConfig(**config_dict.get('analysis', {}))
        self.logging = LoggingConfig(**config_dict.get('logging', {}))
        self.visualization = VisualizationConfig(**config_dict.get('visualization', {}))
        
        # Add other top-level config items
        for key, value in config_dict.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config.update(updates)
        # Re-initialize parsed sections
        self.__init__(self._config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self._config, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
