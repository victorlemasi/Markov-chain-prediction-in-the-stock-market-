import numpy as np
import pandas as pd
import requests
import time
import json
import websocket
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Container for market data"""
    symbol: str
    price: float
    timestamp: datetime
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

@dataclass
class TradingSignal:
    """Trading signal from Markov model"""
    symbol: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    predicted_state: int
    timestamp: datetime

class BrokerAPI(ABC):
    """Abstract base class for broker APIs"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    def get_live_price(self, symbol: str) -> MarketData:
        """Get current live price for symbol"""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get historical price data"""
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'market') -> Dict:
        """Place a trading order"""
        pass

class AlphaVantageAPI(BrokerAPI):
    """Alpha Vantage API implementation for market data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.connected = False
    
    def connect(self) -> bool:
        """Test API connection"""
        try:
            response = requests.get(
                self.base_url,
                params={
                    'function': 'GLOBAL_QUOTE',
                    'symbol': 'AAPL',
                    'apikey': self.api_key
                },
                timeout=10
            )
            self.connected = response.status_code == 200
            return self.connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def get_live_price(self, symbol: str) -> MarketData:
        """Get current live price"""
        try:
            response = requests.get(
                self.base_url,
                params={
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
            )
            data = response.json()
            quote = data.get('Global Quote', {})
            
            return MarketData(
                symbol=symbol,
                price=float(quote.get('05. price', 0)),
                timestamp=datetime.now(),
                volume=float(quote.get('06. volume', 0))
            )
        except Exception as e:
            logger.error(f"Error getting live price for {symbol}: {e}")
            return MarketData(symbol, 0, datetime.now())
    
    def get_historical_data(self, symbol: str, period: str = 'compact') -> pd.DataFrame:
        """Get historical daily prices"""
        try:
            response = requests.get(
                self.base_url,
                params={
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': symbol,
                    'outputsize': period,
                    'apikey': self.api_key
                }
            )
            data = response.json()
            time_series = data.get('Time Series (Daily)', {})
            
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'date': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            return df.sort_values('date').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'market') -> Dict:
        """Alpha Vantage doesn't support trading - this is for demo purposes"""
        logger.warning("Alpha Vantage is data-only. Simulating order placement.")
        return {
            'status': 'simulated',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'timestamp': datetime.now().isoformat()
        }

class YahooFinanceAPI(BrokerAPI):
    """Yahoo Finance API implementation (free alternative)"""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com"
        self.connected = False
    
    def connect(self) -> bool:
        """Test connection"""
        try:
            response = requests.get(f"{self.base_url}/v8/finance/chart/AAPL", timeout=10)
            self.connected = response.status_code == 200
            return self.connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def get_live_price(self, symbol: str) -> MarketData:
        """Get current live price"""
        try:
            response = requests.get(f"{self.base_url}/v8/finance/chart/{symbol}")
            data = response.json()
            result = data['chart']['result'][0]
            
            current_price = result['meta']['regularMarketPrice']
            volume = result['meta']['regularMarketVolume']
            
            return MarketData(
                symbol=symbol,
                price=current_price,
                timestamp=datetime.now(),
                volume=volume
            )
        except Exception as e:
            logger.error(f"Error getting live price for {symbol}: {e}")
            return MarketData(symbol, 0, datetime.now())
    
    def get_historical_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get historical data"""
        try:
            # Calculate timestamp range
            end_time = int(time.time())
            if period == '1d':
                start_time = end_time - 86400
                interval = '1m'
            elif period == '1w':
                start_time = end_time - 604800
                interval = '5m'
            elif period == '1mo':
                start_time = end_time - 2592000
                interval = '1h'
            else:  # Default to 1 year
                start_time = end_time - 31536000
                interval = '1d'
            
            url = f"{self.base_url}/v8/finance/chart/{symbol}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': interval
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            result = data['chart']['result'][0]
            
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            df_data = []
            for i, ts in enumerate(timestamps):
                if all(x is not None for x in [quotes['open'][i], quotes['high'][i], 
                                             quotes['low'][i], quotes['close'][i]]):
                    df_data.append({
                        'date': pd.to_datetime(ts, unit='s'),
                        'open': quotes['open'][i],
                        'high': quotes['high'][i],
                        'low': quotes['low'][i],
                        'close': quotes['close'][i],
                        'volume': quotes['volume'][i] if quotes['volume'][i] else 0
                    })
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'market') -> Dict:
        """Yahoo Finance doesn't support trading - simulation only"""
        logger.warning("Yahoo Finance is data-only. Simulating order placement.")
        return {
            'status': 'simulated',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'timestamp': datetime.now().isoformat()
        }

class AdvancedMarkovTrader:
    """Advanced trading system using complex multi-dimensional Markov chain predictions"""
    
    def __init__(self, broker_api: BrokerAPI, symbols: List[str]):
        self.broker = broker_api
        self.symbols = symbols
        self.models = {}  # Complex Markov models for each symbol
        self.price_history = {symbol: [] for symbol in symbols}
        self.volume_history = {symbol: [] for symbol in symbols}
        self.volatility_history = {symbol: [] for symbol in symbols}
        self.signals = []
        self.running = False
        self.update_interval = 60  # seconds
        
        # Advanced model parameters
        self.markov_order = 3  # Higher order Markov chain
        self.n_return_states = 7  # More granular return states
        self.n_volatility_states = 5  # Volatility states
        self.n_volume_states = 3  # Volume states
        self.regime_lookback = 50  # Days for regime detection
        
    def initialize_models(self, lookback_days: int = 250):
        """Initialize complex multi-dimensional Markov models"""
        logger.info("Initializing advanced Markov models...")
        
        for symbol in self.symbols:
            try:
                # Get historical data
                hist_data = self.broker.get_historical_data(symbol, 'compact')
                
                if len(hist_data) > 100:  # More data required for complex model
                    prices = hist_data['close'].values[-lookback_days:]
                    volumes = hist_data['volume'].values[-lookback_days:]
                    
                    # Create comprehensive model
                    model = self._create_advanced_markov_model(prices, volumes, hist_data)
                    self.models[symbol] = model
                    
                    # Initialize histories
                    self.price_history[symbol] = list(prices[-50:])
                    self.volume_history[symbol] = list(volumes[-50:])
                    
                    # Calculate initial volatility history
                    returns = np.diff(np.log(prices))
                    volatility = self._calculate_rolling_volatility(returns, window=20)
                    self.volatility_history[symbol] = list(volatility[-30:])
                    
                    logger.info(f"Advanced model initialized for {symbol}")
                else:
                    logger.warning(f"Insufficient data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error initializing model for {symbol}: {e}")
    
    def _create_advanced_markov_model(self, prices: np.ndarray, volumes: np.ndarray, hist_data: pd.DataFrame) -> Dict:
        """Create a sophisticated multi-dimensional Markov model"""
        returns = np.diff(np.log(prices))
        
        # 1. Market Regime Detection using Hidden Markov Model approach
        regimes = self._detect_market_regimes(returns, prices)
        
        # 2. Multi-dimensional state space
        return_states = self._discretize_returns_adaptive(returns)
        volatility_states = self._discretize_volatility(returns)
        volume_states = self._discretize_volume(volumes)
        
        # 3. Combined state space (return, volatility, volume, regime)
        combined_states = []
        min_len = min(len(return_states), len(volatility_states), len(volume_states), len(regimes))
        
        for i in range(min_len):
            combined_state = (
                return_states[i],
                volatility_states[i], 
                volume_states[i],
                regimes[i]
            )
            combined_states.append(combined_state)
        
        # 4. Higher-order transition matrices
        transition_matrices = self._build_higher_order_transitions(combined_states)
        
        # 5. Volatility clustering model
        volatility_model = self._build_volatility_clustering_model(returns)
        
        # 6. Momentum and mean reversion features
        technical_features = self._extract_technical_features(hist_data)
        
        # 7. Seasonal patterns
        seasonal_patterns = self._detect_seasonal_patterns(hist_data)
        
        return {
            'transition_matrices': transition_matrices,
            'return_quantiles': np.quantile(returns, np.linspace(0, 1, self.n_return_states + 1)),
            'volatility_quantiles': self._get_volatility_quantiles(returns),
            'volume_quantiles': np.quantile(volumes, np.linspace(0, 1, self.n_volume_states + 1)),
            'volatility_model': volatility_model,
            'technical_features': technical_features,
            'seasonal_patterns': seasonal_patterns,
            'regime_boundaries': self._get_regime_boundaries(returns),
            'markov_order': self.markov_order,
            'state_dimensions': {
                'returns': self.n_return_states,
                'volatility': self.n_volatility_states,
                'volume': self.n_volume_states,
                'regimes': 3
            }
        }
    
    def _detect_market_regimes(self, returns: np.ndarray, prices: np.ndarray) -> List[int]:
        """Detect market regimes (bull, bear, sideways) using statistical methods"""
        # Calculate rolling statistics
        window = self.regime_lookback
        rolling_return = pd.Series(returns).rolling(window).mean()
        rolling_vol = pd.Series(returns).rolling(window).std()
        
        # Price trend analysis
        price_series = pd.Series(prices)
        rolling_slope = []
        
        for i in range(window, len(prices)):
            x = np.arange(window)
            y = prices[i-window:i]
            slope = np.polyfit(x, y, 1)[0] / prices[i-1]  # Normalized slope
            rolling_slope.append(slope)
        
        rolling_slope = np.array(rolling_slope)
        
        # Regime classification
        regimes = []
        for i in range(len(rolling_slope)):
            slope = rolling_slope[i]
            vol = rolling_vol.iloc[i + window]
            ret = rolling_return.iloc[i + window]
            
            # Bull market: positive trend, moderate volatility
            if slope > 0.001 and ret > 0.0005:
                regime = 0  # Bull
            # Bear market: negative trend, high volatility
            elif slope < -0.001 and ret < -0.0005:
                regime = 1  # Bear  
            # Sideways: low trend, variable volatility
            else:
                regime = 2  # Sideways
                
            regimes.append(regime)
        
        # Pad beginning to match return array length
        regimes = [2] * (len(returns) - len(regimes)) + regimes
        return regimes[:len(returns)]
    
    def _discretize_returns_adaptive(self, returns: np.ndarray) -> List[int]:
        """Adaptive return discretization based on distribution characteristics"""
        # Use mixture of quantiles and standard deviations
        std_dev = np.std(returns)
        mean_ret = np.mean(returns)
        
        # Create boundaries based on standard deviations and quantiles
        std_boundaries = [mean_ret + i * std_dev for i in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]]
        quantile_boundaries = np.quantile(returns, np.linspace(0.1, 0.9, 6))
        
        # Combine and sort boundaries
        boundaries = sorted(list(set(std_boundaries + list(quantile_boundaries))))
        boundaries = boundaries[:self.n_return_states-1]  # Limit to desired states
        
        return [np.digitize([ret], boundaries)[0] for ret in returns]
    
    def _discretize_volatility(self, returns: np.ndarray) -> List[int]:
        """Discretize volatility using GARCH-like approach"""
        # Calculate rolling volatility with different windows
        vol_short = pd.Series(returns).rolling(5).std().fillna(0)
        vol_long = pd.Series(returns).rolling(20).std().fillna(0)
        
        # Volatility ratio as a measure of volatility clustering
        vol_ratio = vol_short / vol_long
        vol_ratio = vol_ratio.fillna(1.0)
        
        # Discretize based on quantiles
        boundaries = np.quantile(vol_ratio, np.linspace(0, 1, self.n_volatility_states + 1))
        return [np.digitize([vol], boundaries[1:-1])[0] for vol in vol_ratio]
    
    def _discretize_volume(self, volumes: np.ndarray) -> List[int]:
        """Discretize volume into high, medium, low"""
        # Use relative volume (compared to recent average)
        vol_series = pd.Series(volumes)
        avg_volume = vol_series.rolling(20).mean().fillna(vol_series.mean())
        relative_volume = vol_series / avg_volume
        
        # Simple discretization: low, normal, high
        volume_states = []
        for rv in relative_volume:
            if rv < 0.8:
                volume_states.append(0)  # Low
            elif rv > 1.2:
                volume_states.append(2)  # High
            else:
                volume_states.append(1)  # Normal
                
        return volume_states
    
    def _build_higher_order_transitions(self, combined_states: List[tuple]) -> Dict:
        """Build higher-order transition matrices"""
        from collections import defaultdict, Counter
        
        transitions = {}
        
        # Build transitions for different orders
        for order in range(1, self.markov_order + 1):
            order_transitions = defaultdict(Counter)
            
            for i in range(len(combined_states) - order):
                current_sequence = tuple(combined_states[i:i + order])
                next_state = combined_states[i + order]
                order_transitions[current_sequence][next_state] += 1
            
            # Convert to probabilities with smoothing
            order_probs = {}
            for seq, next_counts in order_transitions.items():
                total = sum(next_counts.values())
                # Laplace smoothing
                smoothed_probs = {}
                all_possible_states = set([s for seq_list in order_transitions.values() 
                                         for s in seq_list.keys()])
                
                for state in all_possible_states:
                    count = next_counts.get(state, 0)
                    smoothed_probs[state] = (count + 1) / (total + len(all_possible_states))
                
                order_probs[seq] = smoothed_probs
            
            transitions[f'order_{order}'] = order_probs
        
        return transitions
    
    def _build_volatility_clustering_model(self, returns: np.ndarray) -> Dict:
        """Build GARCH-like volatility clustering model"""
        # Simple GARCH(1,1) approximation
        returns_sq = returns ** 2
        
        # Estimate GARCH parameters using simple regression
        lagged_vol = pd.Series(returns_sq).shift(1).fillna(returns_sq.mean())
        
        try:
            # Simple linear regression for GARCH parameters
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(lagged_vol, returns_sq)
            
            return {
                'alpha': max(0.01, min(0.99, slope)),  # Constrain parameters
                'ocomplex code version 1.2
