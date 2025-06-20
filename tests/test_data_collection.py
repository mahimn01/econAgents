"""
Tests for data collection functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime

from rrce_framework.core.data.collectors import (
    DataConfig, RRCEDataPoint, RRCEDataManager,
    EconomicDataCollector, ResourceDataCollector,
    EnvironmentalDataCollector, SocialDataCollector
)

@pytest.fixture
def temp_data_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_config(temp_data_dir):
    """Create test configuration."""
    return DataConfig(
        countries=['US'],
        start_date='2020-01-01',
        end_date='2021-12-31',
        data_dir=temp_data_dir,
        cache_enabled=True,
        api_keys={}
    )

def test_data_config_creation(test_config):
    """Test DataConfig creation and initialization."""
    assert test_config.countries == ['US']
    assert test_config.start_date == '2020-01-01'
    assert test_config.end_date == '2021-12-31'
    assert test_config.cache_enabled is True
    assert test_config.data_dir.exists()

def test_rrce_data_point():
    """Test RRCEDataPoint creation."""
    data_point = RRCEDataPoint(
        country='US',
        date=datetime(2020, 1, 1),
        category='economic',
        subcategory='macroeconomic',
        variable='GDP',
        value=20000.0,
        unit='billions_usd',
        source='World Bank'
    )
    
    assert data_point.country == 'US'
    assert data_point.variable == 'GDP'
    assert data_point.value == 20000.0

def test_data_manager_initialization(test_config):
    """Test RRCEDataManager initialization."""
    manager = RRCEDataManager(test_config)
    
    assert manager.config == test_config
    assert 'economic' in manager.collectors
    assert 'resource' in manager.collectors
    assert 'environmental' in manager.collectors
    assert 'social' in manager.collectors

def test_data_processing():
    """Test data processing functionality."""
    # Create sample data points
    sample_data = {
        'US': {
            'economic': [
                RRCEDataPoint(
                    country='US',
                    date=datetime(2020, 1, 1),
                    category='economic',
                    subcategory='macroeconomic',
                    variable='GDP',
                    value=20000.0,
                    unit='billions_usd',
                    source='Test'
                ),
                RRCEDataPoint(
                    country='US',
                    date=datetime(2021, 1, 1),
                    category='economic',
                    subcategory='macroeconomic',
                    variable='GDP',
                    value=21000.0,
                    unit='billions_usd',
                    source='Test'
                )
            ]
        }
    }
    
    # Create manager and process data
    with tempfile.TemporaryDirectory() as tmpdir:
        config = DataConfig(
            countries=['US'],
            start_date='2020-01-01',
            end_date='2021-12-31',
            data_dir=Path(tmpdir)
        )
        manager = RRCEDataManager(config)
        
        processed_data = manager.process_for_rrce(sample_data)
        
        assert 'US' in processed_data
        assert 'GDP' in processed_data['US'].columns
        assert len(processed_data['US']) == 2
        assert processed_data['US']['GDP'].iloc[0] == 20000.0

if __name__ == '__main__':
    pytest.main([__file__])