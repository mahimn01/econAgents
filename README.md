# RRCE Framework: Resource-Reality Coupled Economics

A comprehensive framework for modeling economic systems under physical, environmental, and social constraints.

## Overview

The Resource-Reality Coupled Economics (RRCE) Framework provides a mathematically rigorous approach to economic modeling that incorporates:

- **Physical constraints** (conservation laws, thermodynamics, carrying capacity)
- **Environmental factors** (resource depletion, pollution, climate impacts)  
- **Social dynamics** (inequality, stability, coordination effects)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/rrce-framework.git
cd rrce-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Configuration

1. Copy `.env.example` to `.env` and fill in your API keys
2. Modify `config/default_config.yaml` as needed
3. Select countries in `config/countries.yaml`

### Basic Usage

```python
from rrce_framework import RRCEFramework

# Initialize the framework
rrce = RRCEFramework.from_config('config/default_config.yaml')

# Collect data
rrce.collect_data(['US', 'DE', 'CN'])

# Run simulation
results = rrce.simulate(start_date='2020-01-01', end_date='2023-12-31')

# Analyze results
rrce.analyze_results(results)
```

## Features

- **Comprehensive Data Collection**: Automated collection from World Bank, FRED, OECD
- **Mathematical Rigor**: All components derived from first principles
- **Flexible Simulation**: Configurable models and parameters
- **Comparative Analysis**: Built-in comparison with conventional economic models
- **Rich Visualization**: Interactive plots and dashboards

## Documentation

- [Mathematical Foundation](docs/mathematical_foundation.md)
- [API Documentation](docs/API.md)
- [Examples and Tutorials](notebooks/)

## License

MIT License - see LICENSE file for details.
