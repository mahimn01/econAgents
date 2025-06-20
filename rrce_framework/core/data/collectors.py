"""
Data collection infrastructure for RRCE Framework.
This integrates the comprehensive data collection system into the main framework.
"""

import pandas as pd
import numpy as np
import requests
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings

# Data source libraries
try:
    import pandas_datareader.data as web
    import wbdata
    import fredapi
    import yfinance as yf
except ImportError as e:
    logging.warning(f"Some data source libraries not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data collection parameters."""
    countries: List[str]
    start_date: str
    end_date: str
    data_dir: Path
    cache_enabled: bool = True
    api_keys: Dict[str, str] = None
    update_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.api_keys is None:
            self.api_keys = {}

@dataclass
class RRCEDataPoint:
    """Standardized data point for RRCE framework."""
    country: str
    date: datetime
    category: str  # economic, resource, environmental, social
    subcategory: str
    variable: str
    value: float
    unit: str
    source: str
    metadata: Dict = None

class DataCollector(ABC):
    """Abstract base class for all data collectors."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.cache_dir = config.data_dir / "cache" / self.__class__.__name__
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def collect_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect data for specified country and time period."""
        pass
    
    def _get_cache_path(self, country: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path."""
        return self.cache_dir / f"{country}_{start_date}_{end_date}.pkl"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[List[RRCEDataPoint]]:
        """Load data from cache if available and recent."""
        if not self.config.cache_enabled or not cache_path.exists():
            return None
            
        try:
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age.days > 7:  # Cache expires after 7 days
                return None
                
            data = pd.read_pickle(cache_path)
            logger.info(f"Loaded {len(data)} records from cache: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, data: List[RRCEDataPoint], cache_path: Path):
        """Save data to cache."""
        if self.config.cache_enabled:
            try:
                pd.to_pickle(data, cache_path)
                logger.info(f"Saved {len(data)} records to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")

class EconomicDataCollector(DataCollector):
    """Collects economic indicators from multiple sources."""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.fred = None
        if 'fred_api_key' in config.api_keys:
            try:
                self.fred = fredapi.Fred(api_key=config.api_keys['fred_api_key'])
            except Exception as e:
                logger.warning(f"Failed to initialize FRED API: {e}")
    
    def collect_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect economic data for a country."""
        cache_path = self._get_cache_path(country, start_date, end_date)
        cached_data = self._load_from_cache(cache_path)
        if cached_data:
            return cached_data
        
        data_points = []
        
        # Collect World Bank data
        data_points.extend(self._collect_world_bank_data(country, start_date, end_date))
        
        # Collect FRED data (primarily for US)
        if country.upper() in ['US', 'USA', 'UNITED STATES'] and self.fred:
            data_points.extend(self._collect_fred_data(country, start_date, end_date))
        
        # Collect OECD data
        data_points.extend(self._collect_oecd_data(country, start_date, end_date))
        
        self._save_to_cache(data_points, cache_path)
        return data_points
    
    def _collect_world_bank_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect data from World Bank."""
        data_points = []
        
        # World Bank indicator mappings
        wb_indicators = {
            'NY.GDP.MKTP.CD': ('GDP', 'current_usd', 'GDP at current market prices'),
            'NY.GDP.PCAP.CD': ('GDP_per_capita', 'current_usd', 'GDP per capita'),
            'FP.CPI.TOTL.ZG': ('inflation', 'percent', 'Inflation rate'),
            'SL.UEM.TOTL.ZS': ('unemployment', 'percent', 'Unemployment rate'),
            'NE.TRD.GNFS.ZS': ('trade_openness', 'percent_gdp', 'Trade as % of GDP'),
            'GC.DOD.TOTL.GD.ZS': ('debt_to_gdp', 'percent', 'Government debt to GDP'),
            'BX.KLT.DINV.CD.WD': ('fdi_inflows', 'current_usd', 'Foreign direct investment inflows'),
        }
        
        try:
            # Convert country code
            country_code = self._get_wb_country_code(country)
            
            for indicator, (var_name, unit, description) in wb_indicators.items():
                try:
                    wb_data = wbdata.get_data(indicator, country=country_code, 
                                            date=(start_date, end_date))
                    
                    for record in wb_data:
                        if record['value'] is not None:
                            data_points.append(RRCEDataPoint(
                                country=country,
                                date=datetime.strptime(record['date'], '%Y'),
                                category='economic',
                                subcategory='macroeconomic',
                                variable=var_name,
                                value=float(record['value']),
                                unit=unit,
                                source='World Bank',
                                metadata={'indicator': indicator, 'description': description}
                            ))
                            
                except Exception as e:
                    logger.warning(f"Failed to collect WB data for {indicator}: {e}")
                    
        except Exception as e:
            logger.error(f"World Bank data collection failed for {country}: {e}")
        
        return data_points
    
    def _collect_fred_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect data from FRED (US data)."""
        data_points = []
        
        fred_indicators = {
            'GDP': ('GDP', 'billions_usd', 'Gross Domestic Product'),
            'CPIAUCSL': ('cpi', 'index', 'Consumer Price Index'),
            'UNRATE': ('unemployment_rate', 'percent', 'Unemployment Rate'),
            'FEDFUNDS': ('fed_funds_rate', 'percent', 'Federal Funds Rate'),
            'DGS10': ('treasury_10y', 'percent', '10-Year Treasury Rate'),
            'DEXUSEU': ('usd_eur_rate', 'rate', 'USD/EUR Exchange Rate'),
        }
        
        try:
            for series_id, (var_name, unit, description) in fred_indicators.items():
                try:
                    fred_data = self.fred.get_series(series_id, start_date, end_date)
                    
                    for date, value in fred_data.items():
                        if pd.notna(value):
                            data_points.append(RRCEDataPoint(
                                country=country,
                                date=date,
                                category='economic',
                                subcategory='financial',
                                variable=var_name,
                                value=float(value),
                                unit=unit,
                                source='FRED',
                                metadata={'series_id': series_id, 'description': description}
                            ))
                            
                except Exception as e:
                    logger.warning(f"Failed to collect FRED data for {series_id}: {e}")
                    
        except Exception as e:
            logger.error(f"FRED data collection failed: {e}")
        
        return data_points
    
    def _collect_oecd_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect data from OECD."""
        # OECD data collection would go here
        # For now, return empty list as OECD API requires special handling
        return []
    
    def _get_wb_country_code(self, country: str) -> str:
        """Convert country name to World Bank country code."""
        country_mapping = {
            'united states': 'US', 'usa': 'US', 'us': 'US',
            'china': 'CN', 'germany': 'DE', 'japan': 'JP',
            'united kingdom': 'GB', 'uk': 'GB', 'france': 'FR',
            'india': 'IN', 'italy': 'IT', 'brazil': 'BR',
            'canada': 'CA', 'russia': 'RU', 'south korea': 'KR',
            'spain': 'ES', 'australia': 'AU', 'mexico': 'MX',
        }
        return country_mapping.get(country.lower(), country.upper())

class ResourceDataCollector(DataCollector):
    """Collects resource availability and extraction data."""
    
    def collect_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect resource data for a country."""
        cache_path = self._get_cache_path(country, start_date, end_date)
        cached_data = self._load_from_cache(cache_path)
        if cached_data:
            return cached_data
        
        data_points = []
        
        # Collect energy data
        data_points.extend(self._collect_energy_data(country, start_date, end_date))
        
        # Collect agricultural resource data
        data_points.extend(self._collect_agricultural_data(country, start_date, end_date))
        
        self._save_to_cache(data_points, cache_path)
        return data_points
    
    def _collect_energy_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect energy resource data."""
        data_points = []
        
        # World Bank energy indicators
        energy_indicators = {
            'EG.USE.PCAP.KG.OE': ('energy_use_per_capita', 'kg_oil_equivalent', 'Energy use per capita'),
            'EG.USE.COMM.FO.ZS': ('fossil_fuel_consumption', 'percent', 'Fossil fuel energy consumption'),
            'EG.ELC.RNEW.ZS': ('renewable_electricity', 'percent', 'Renewable electricity output'),
            'EG.IMP.CONS.ZS': ('energy_imports', 'percent', 'Energy imports, net'),
        }
        
        try:
            country_code = self._get_wb_country_code(country)
            
            for indicator, (var_name, unit, description) in energy_indicators.items():
                try:
                    wb_data = wbdata.get_data(indicator, country=country_code, 
                                            date=(start_date, end_date))
                    
                    for record in wb_data:
                        if record['value'] is not None:
                            data_points.append(RRCEDataPoint(
                                country=country,
                                date=datetime.strptime(record['date'], '%Y'),
                                category='resource',
                                subcategory='energy',
                                variable=var_name,
                                value=float(record['value']),
                                unit=unit,
                                source='World Bank',
                                metadata={'indicator': indicator, 'description': description}
                            ))
                            
                except Exception as e:
                    logger.warning(f"Failed to collect energy data for {indicator}: {e}")
                    
        except Exception as e:
            logger.error(f"Energy data collection failed for {country}: {e}")
        
        return data_points
    
    def _collect_agricultural_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect agricultural resource data."""
        data_points = []
        
        # FAO/World Bank agricultural indicators
        agri_indicators = {
            'AG.LND.ARBL.HA.PC': ('arable_land_per_capita', 'hectares', 'Arable land per person'),
            'AG.LND.FRST.ZS': ('forest_area', 'percent_land', 'Forest area as % of land area'),
            'AG.PRD.FOOD.XD': ('food_production_index', 'index', 'Food production index'),
            'ER.H2O.FWTL.K3': ('freshwater_withdrawal', 'billion_cubic_meters', 'Annual freshwater withdrawals'),
        }
        
        try:
            country_code = self._get_wb_country_code(country)
            
            for indicator, (var_name, unit, description) in agri_indicators.items():
                try:
                    wb_data = wbdata.get_data(indicator, country=country_code, 
                                            date=(start_date, end_date))
                    
                    for record in wb_data:
                        if record['value'] is not None:
                            data_points.append(RRCEDataPoint(
                                country=country,
                                date=datetime.strptime(record['date'], '%Y'),
                                category='resource',
                                subcategory='agricultural',
                                variable=var_name,
                                value=float(record['value']),
                                unit=unit,
                                source='World Bank',
                                metadata={'indicator': indicator, 'description': description}
                            ))
                            
                except Exception as e:
                    logger.warning(f"Failed to collect agricultural data for {indicator}: {e}")
                    
        except Exception as e:
            logger.error(f"Agricultural data collection failed for {country}: {e}")
        
        return data_points
    
    def _get_wb_country_code(self, country: str) -> str:
        """Convert country name to World Bank country code."""
        country_mapping = {
            'united states': 'US', 'usa': 'US', 'us': 'US',
            'china': 'CN', 'germany': 'DE', 'japan': 'JP',
            'united kingdom': 'GB', 'uk': 'GB', 'france': 'FR',
            'india': 'IN', 'italy': 'IT', 'brazil': 'BR',
            'canada': 'CA', 'russia': 'RU', 'south korea': 'KR',
        }
        return country_mapping.get(country.lower(), country.upper())

class EnvironmentalDataCollector(DataCollector):
    """Collects environmental indicators and degradation metrics."""
    
    def collect_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect environmental data for a country."""
        cache_path = self._get_cache_path(country, start_date, end_date)
        cached_data = self._load_from_cache(cache_path)
        if cached_data:
            return cached_data
        
        data_points = []
        
        # Collect emissions data
        data_points.extend(self._collect_emissions_data(country, start_date, end_date))
        
        self._save_to_cache(data_points, cache_path)
        return data_points
    
    def _collect_emissions_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect greenhouse gas emissions data."""
        data_points = []
        
        # World Bank emissions indicators
        emissions_indicators = {
            'EN.ATM.CO2E.PC': ('co2_emissions_per_capita', 'metric_tons', 'CO2 emissions per capita'),
            'EN.ATM.CO2E.KT': ('co2_emissions_total', 'kilotons', 'CO2 emissions total'),
            'EN.ATM.METH.KT.CE': ('methane_emissions', 'kt_co2_equivalent', 'Methane emissions'),
            'EN.ATM.NOXE.KT.CE': ('nitrous_oxide_emissions', 'kt_co2_equivalent', 'Nitrous oxide emissions'),
        }
        
        try:
            country_code = self._get_wb_country_code(country)
            
            for indicator, (var_name, unit, description) in emissions_indicators.items():
                try:
                    wb_data = wbdata.get_data(indicator, country=country_code, 
                                            date=(start_date, end_date))
                    
                    for record in wb_data:
                        if record['value'] is not None:
                            data_points.append(RRCEDataPoint(
                                country=country,
                                date=datetime.strptime(record['date'], '%Y'),
                                category='environmental',
                                subcategory='emissions',
                                variable=var_name,
                                value=float(record['value']),
                                unit=unit,
                                source='World Bank',
                                metadata={'indicator': indicator, 'description': description}
                            ))
                            
                except Exception as e:
                    logger.warning(f"Failed to collect emissions data for {indicator}: {e}")
                    
        except Exception as e:
            logger.error(f"Emissions data collection failed for {country}: {e}")
        
        return data_points
    
    def _get_wb_country_code(self, country: str) -> str:
        """Convert country name to World Bank country code."""
        country_mapping = {
            'united states': 'US', 'usa': 'US', 'us': 'US',
            'china': 'CN', 'germany': 'DE', 'japan': 'JP',
            'united kingdom': 'GB', 'uk': 'GB', 'france': 'FR',
            'india': 'IN', 'italy': 'IT', 'brazil': 'BR',
            'canada': 'CA', 'russia': 'RU', 'south korea': 'KR',
        }
        return country_mapping.get(country.lower(), country.upper())

class SocialDataCollector(DataCollector):
    """Collects social indicators including inequality and welfare metrics."""
    
    def collect_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect social data for a country."""
        cache_path = self._get_cache_path(country, start_date, end_date)
        cached_data = self._load_from_cache(cache_path)
        if cached_data:
            return cached_data
        
        data_points = []
        
        # Collect inequality data
        data_points.extend(self._collect_inequality_data(country, start_date, end_date))
        
        self._save_to_cache(data_points, cache_path)
        return data_points
    
    def _collect_inequality_data(self, country: str, start_date: str, end_date: str) -> List[RRCEDataPoint]:
        """Collect income inequality and distribution data."""
        data_points = []
        
        # World Bank inequality indicators
        inequality_indicators = {
            'SI.POV.GINI': ('gini_coefficient', 'index', 'Gini coefficient'),
            'SI.POV.NAHC': ('poverty_headcount_national', 'percent', 'Poverty headcount at national poverty lines'),
            'SI.POV.DDAY': ('poverty_headcount_190', 'percent', 'Poverty headcount at $1.90 a day'),
            'NY.GNP.PCAP.PP.CD': ('gni_per_capita_ppp', 'current_ppp_usd', 'GNI per capita, PPP'),
        }
        
        try:
            country_code = self._get_wb_country_code(country)
            
            for indicator, (var_name, unit, description) in inequality_indicators.items():
                try:
                    wb_data = wbdata.get_data(indicator, country=country_code, 
                                            date=(start_date, end_date))
                    
                    for record in wb_data:
                        if record['value'] is not None:
                            data_points.append(RRCEDataPoint(
                                country=country,
                                date=datetime.strptime(record['date'], '%Y'),
                                category='social',
                                subcategory='inequality',
                                variable=var_name,
                                value=float(record['value']),
                                unit=unit,
                                source='World Bank',
                                metadata={'indicator': indicator, 'description': description}
                            ))
                            
                except Exception as e:
                    logger.warning(f"Failed to collect inequality data for {indicator}: {e}")
                    
        except Exception as e:
            logger.error(f"Inequality data collection failed for {country}: {e}")
        
        return data_points
    
    def _get_wb_country_code(self, country: str) -> str:
        """Convert country name to World Bank country code."""
        country_mapping = {
            'united states': 'US', 'usa': 'US', 'us': 'US',
            'china': 'CN', 'germany': 'DE', 'japan': 'JP',
            'united kingdom': 'GB', 'uk': 'GB', 'france': 'FR',
            'india': 'IN', 'italy': 'IT', 'brazil': 'BR',
            'canada': 'CA', 'russia': 'RU', 'south korea': 'KR',
        }
        return country_mapping.get(country.lower(), country.upper())

class RRCEDataManager:
    """Main coordinator for all data collection and processing."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.collectors = {
            'economic': EconomicDataCollector(config),
            'resource': ResourceDataCollector(config),
            'environmental': EnvironmentalDataCollector(config),
            'social': SocialDataCollector(config),
        }
        
        # Initialize data storage
        self.raw_data: List[RRCEDataPoint] = []
        self.processed_data: Dict[str, pd.DataFrame] = {}
        
    def collect_all_data(self) -> Dict[str, List[RRCEDataPoint]]:
        """Collect all data for all countries and time periods."""
        all_data = {}
        
        for country in self.config.countries:
            logger.info(f"Collecting data for {country}")
            country_data = {}
            
            for category, collector in self.collectors.items():
                logger.info(f"  Collecting {category} data...")
                try:
                    data = collector.collect_data(
                        country, self.config.start_date, self.config.end_date
                    )
                    country_data[category] = data
                    logger.info(f"    Collected {len(data)} {category} data points")
                except Exception as e:
                    logger.error(f"    Failed to collect {category} data: {e}")
                    country_data[category] = []
            
            all_data[country] = country_data
        
        return all_data
    
    def process_for_rrce(self, raw_data: Dict[str, List[RRCEDataPoint]]) -> Dict[str, pd.DataFrame]:
        """Process raw data into RRCE model inputs."""
        processed_data = {}
        
        for country, country_data in raw_data.items():
            logger.info(f"Processing data for {country}")
            
            # Combine all data points for this country
            all_points = []
            for category_data in country_data.values():
                all_points.extend(category_data)
            
            if not all_points:
                logger.warning(f"No data points found for {country}")
                continue
            
            # Convert to DataFrame
            df_data = []
            for point in all_points:
                df_data.append({
                    'country': point.country,
                    'date': point.date,
                    'category': point.category,
                    'subcategory': point.subcategory,
                    'variable': point.variable,
                    'value': point.value,
                    'unit': point.unit,
                    'source': point.source,
                })
            
            df = pd.DataFrame(df_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create pivot table for easier access
            pivot_df = df.pivot_table(
                index='date',
                columns='variable',
                values='value',
                aggfunc='mean'  # Average if multiple values per date
            )
            
            processed_data[country] = pivot_df
            logger.info(f"  Processed {len(df)} data points into {pivot_df.shape} matrix")
        
        return processed_data
    
    def validate_data_quality(self) -> Dict[str, Dict[str, float]]:
        """Validate data quality and completeness."""
        quality_report = {}
        
        for country, df in self.processed_data.items():
            country_quality = {}
            
            # Calculate completeness
            total_cells = df.size
            non_null_cells = df.count().sum()
            country_quality['completeness'] = non_null_cells / total_cells if total_cells > 0 else 0
            
            # Calculate temporal coverage
            date_range = df.index.max() - df.index.min()
            expected_range = pd.to_datetime(self.config.end_date) - pd.to_datetime(self.config.start_date)
            country_quality['temporal_coverage'] = date_range / expected_range if expected_range.days > 0 else 0
            
            # Count available variables
            country_quality['variable_count'] = df.shape[1]
            
            quality_report[country] = country_quality
        
        return quality_report
    
    def save_processed_data(self, filepath: Optional[Path] = None):
        """Save processed data to disk."""
        if filepath is None:
            filepath = self.config.data_dir / "processed_data.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pd.to_pickle(self.processed_data, f)
        
        logger.info(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filepath: Optional[Path] = None):
        """Load processed data from disk."""
        if filepath is None:
            filepath = self.config.data_dir / "processed_data.pkl"
        
        if filepath.exists():
            with open(filepath, 'rb') as f:
                self.processed_data = pd.read_pickle(f)
            logger.info(f"Loaded processed data from {filepath}")
        else:
            logger.warning(f"No processed data file found at {filepath}")
