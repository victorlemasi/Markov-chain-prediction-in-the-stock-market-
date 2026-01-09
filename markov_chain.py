import numpy as np
import pandas as pd
import requests
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# MT5 imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("Warning: MetaTrader5 package not installed. Install with: pip install MetaTrader5")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MarkovBot")

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
    expected_return: float
    timestamp: datetime

class BrokerAPI(ABC):
    """Abstract base class for broker APIs"""
    
    @abstractmethod
    def connect(self) -> bool:
        pass
    
    @abstractmethod
    def get_live_price(self, symbol: str) -> MarketData:
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'market', sl: float = None, tp: float = None) -> Dict:
        pass

    @abstractmethod
    def get_positions(self, symbol: str = None) -> List[Dict]:
        pass

class MT5API(BrokerAPI):
    """MetaTrader 5 API implementation"""
    
    def __init__(self, login: int = None, password: str = None, server: str = None, path: str = None):
        if not MT5_AVAILABLE:
            raise ImportError("MetaTrader5 package not installed")
            
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.account_info = None
        
    def connect(self) -> bool:
        try:
            if self.path:
                if not mt5.initialize(path=self.path):
                    logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize():
                    logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
            
            if self.login and self.password and self.server:
                if not mt5.login(self.login, self.password, self.server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
            
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logger.error("Failed to get account info")
                return False
                
            self.connected = True
            logger.info(f"Connected to MT5. Account: {self.account_info.login}, Balance: {self.account_info.balance}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def get_live_price(self, symbol: str) -> MarketData:
        try:
            if not self.connected:
                return MarketData(symbol, 0, datetime.now())
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return MarketData(symbol, 0, datetime.now())
            
            return MarketData(
                symbol=symbol,
                price=(tick.bid + tick.ask) / 2,
                timestamp=datetime.fromtimestamp(tick.time),
                volume=tick.volume if hasattr(tick, 'volume') else None,
                bid=tick.bid,
                ask=tick.ask
            )
        except Exception as e:
            logger.error(f"Error getting live price for {symbol}: {e}")
            return MarketData(symbol, 0, datetime.now())
    
    def get_historical_data(self, symbol: str, period: str = 'D1', count: int = 1000) -> pd.DataFrame:
        try:
            if not self.connected:
                return pd.DataFrame()
            
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1
            }
            timeframe = timeframe_map.get(period, mt5.TIMEFRAME_D1)
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get rates for {symbol}: {mt5.last_error()}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['date'] = pd.to_datetime(df['time'], unit='s')
            df = df[['date', 'open', 'high', 'low', 'close', 'tick_volume']]
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            return df.sort_values('date').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'market', 
                   price: float = None, sl: float = None, tp: float = None, 
                   deviation: int = 20, magic: int = 123456, comment: str = "Markov Trade") -> Dict:
        if not self.connected:
            return {'status': 'error', 'message': 'Not connected'}
            
        try:
            action = mt5.TRADE_ACTION_DEAL
            type_order = mt5.ORDER_TYPE_BUY if side == 'BUY' else mt5.ORDER_TYPE_SELL
            
            # Simple market order
            tick = mt5.symbol_info_tick(symbol)
            current_price = tick.ask if side == 'BUY' else tick.bid
            
            request = {
                "action": action,
                "symbol": symbol,
                "volume": quantity,
                "type": type_order,
                "price": current_price,
                "sl": sl if sl else 0.0,
                "tp": tp if tp else 0.0,
                "deviation": deviation,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return {'status': 'error', 'message': result.comment, 'retcode': result.retcode}
                
            logger.info(f"Order placed: {side} {quantity} {symbol} at {current_price}")
            return {'status': 'success', 'order': result.order}
            
        except Exception as e:
            logger.error(f"Exception placing order: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_positions(self, symbol: str = None) -> List[Dict]:
        if not self.connected: return []
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions:
                return [{'ticket': p.ticket, 'symbol': p.symbol, 'volume': p.volume, 'type': 'BUY' if p.type==0 else 'SELL'} for p in positions]
            return []
        except:
            return []

class YahooFinanceAPI(BrokerAPI):
    """Yahoo Finance data-only API"""
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com"
        self.connected = False
    
    def connect(self) -> bool:
        self.connected = True
        return True
    
    def get_live_price(self, symbol: str) -> MarketData:
        try:
            url = f"{self.base_url}/v8/finance/chart/{symbol}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            data = response.json()
            result = data['chart']['result'][0]
            current_price = result['meta']['regularMarketPrice']
            return MarketData(symbol=symbol, price=current_price, timestamp=datetime.now())
        except:
            return MarketData(symbol, 0, datetime.now())
            
    def get_historical_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        try:
            end = int(time.time())
            start = end - 31536000
            url = f"{self.base_url}/v8/finance/chart/{symbol}?period1={start}&period2={end}&interval=1d"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers)
            data = resp.json()
            result = data['chart']['result'][0]
            quotes = result['indicators']['quote'][0]
            df = pd.DataFrame({
                'date': [datetime.fromtimestamp(ts) for ts in result['timestamp']],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            return df.dropna()
        except:
            return pd.DataFrame()
            
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'market', sl=None, tp=None) -> Dict:
        logger.info(f"SIMULATION: {side} {quantity} {symbol}")
        return {'status': 'simulated'}

    def get_positions(self, symbol: str = None) -> List[Dict]:
        return []

class AdvancedMarkovTrader:
    def __init__(self, broker_api: BrokerAPI, symbols: List[str], lot_size: float = 0.01):
        self.broker = broker_api
        self.symbols = symbols
        self.lot_size = lot_size
        self.models = {}
        # We need to keep a buffer of recent data to calculate states
        self.price_history = {symbol: [] for symbol in symbols}
        self.volume_history = {symbol: [] for symbol in symbols}
        self.running = False
        self.lock = threading.Lock()
        
        # Parameters
        self.markov_order = 3
        self.n_return_states = 7
        self.n_volatility_states = 5
        self.n_volume_states = 3
        self.regime_lookback = 50
        self.update_interval = 5 # seconds for demo, usually higher
        self.risk_per_trade = 0.01

    def initialize_models(self, lookback_days: int = 250):
        logger.info("Initializing models...")
        for symbol in self.symbols:
            try:
                hist_data = self.broker.get_historical_data(symbol, 'D1', lookback_days)
                if not hist_data.empty and len(hist_data) > 100:
                    prices = hist_data['close'].values
                    volumes = hist_data['volume'].values
                    
                    self.models[symbol] = self._create_model(prices, volumes)
                    self.price_history[symbol] = list(prices[-100:]) # Keep enough for lookback
                    self.volume_history[symbol] = list(volumes[-100:])
                    logger.info(f"Initialized model for {symbol}")
            except Exception as e:
                logger.error(f"Failed to init model for {symbol}: {e}")

    def _create_model(self, prices, volumes):
        returns = np.diff(np.log(prices))
        regimes = self._detect_market_regimes(returns, prices)
        
        return_states = self._discretize(returns, self.n_return_states)
        vol_states = self._discretize_vol(returns)
        vol_volume_states = self._discretize_volume(volumes)
        
        # Create combined states
        min_len = min(len(return_states), len(vol_states), len(vol_volume_states), len(regimes))
        states = []
        for i in range(min_len):
            states.append((return_states[i], vol_states[i], vol_volume_states[i], regimes[i]))
            
        transitions = self._build_transitions(states)
        
        return {
            'transitions': transitions,
            'return_boundaries': self._get_boundaries(returns, self.n_return_states),
            'last_state': states[-1] if states else None
        }

    def _build_transitions(self, states):
        transitions = {}
        for order in range(1, self.markov_order + 1):
            t_map = {}
            for i in range(len(states) - order):
                seq = tuple(states[i:i+order])
                next_s = states[i+order]
                if seq not in t_map: t_map[seq] = []
                t_map[seq].append(next_s)
            
            # Convert to probabilities
            t_probs = {}
            for seq, next_list in t_map.items():
                counts = {}
                for s in next_list: counts[s] = counts.get(s, 0) + 1
                total = sum(counts.values())
                t_probs[seq] = {s: c/total for s, c in counts.items()}
            transitions[order] = t_probs
        return transitions

    def _predict(self, symbol):
        if symbol not in self.models: return None
        model = self.models[symbol]
        
        # Reconstruct current state from history
        prices = np.array(self.price_history[symbol])
        volumes = np.array(self.volume_history[symbol])
        
        if len(prices) < 50: return None
        
        returns = np.diff(np.log(prices))
        regimes = self._detect_market_regimes(returns, prices)
        
        ret_s = self._discretize_val(returns[-1], model['return_boundaries'])
        
        # Simplified vol/volume discretization for runtime
        vol_s = 2 # defaulted for demo safety: Middle volatility
        volume_s = 1 # defaulted: Normal volume
        regime = regimes[-1]
        
        current_state = (ret_s, vol_s, volume_s, regime)
        
        # Markov Prediction
        transitions = model['transitions']
        
        # Try Order 1 prediction
        # (In a real full implementation we would lookback N states)
        seq = (current_state,)
        
        if 1 in transitions and seq in transitions[1]:
            probs = transitions[1][seq]
            # Find state with highest prob
            best_state = max(probs, key=probs.get)
            confidence = probs[best_state]
            
            # Interpret best_state (return_state component)
            # return_state is index 0
            pred_return_idx = best_state[0]
            
            # Simple logic: if return state is high (> middle), BUY
            mid = self.n_return_states // 2
            direction = "HOLD"
            if pred_return_idx > mid + 0.5: direction = "BUY"
            elif pred_return_idx < mid - 0.5: direction = "SELL"
            
            # Calculate expected return (proxy)
            exp_ret = (pred_return_idx - mid) * 0.001 
            
            return TradingSignal(symbol, direction, confidence, exp_ret, datetime.now())
            
        return TradingSignal(symbol, "HOLD", 0.0, 0.0, datetime.now())

    def _update_data(self, symbol, live_price: MarketData):
        if not live_price or live_price.price == 0: return
        self.price_history[symbol].append(live_price.price)
        self.volume_history[symbol].append(live_price.volume if live_price.volume else 0)
        
        # Keep buffer size manageable
        if len(self.price_history[symbol]) > 200:
            self.price_history[symbol].pop(0)
            self.volume_history[symbol].pop(0)

    def run(self):
        self.running = True
        logger.info(f"Starting Live Trading Loop for {self.symbols}")
        
        try:
            while self.running:
                for symbol in self.symbols:
                    # 1. Get Data
                    data = self.broker.get_live_price(symbol)
                    logger.info(f"Tick {symbol}: {data.price}")
                    
                    # 2. Update Model History
                    self._update_data(symbol, data)
                    
                    # 3. Predict
                    signal = self._predict(symbol)
                    
                    if signal and signal.direction != "HOLD":
                        logger.info(f"SIGNAL: {signal}")
                        
                        # 4. Check Risk / Positions
                        positions = self.broker.get_positions(symbol)
                        if not positions and signal.confidence > 0.3: # Simple filter
                            # 5. Execute
                            qty = self.lot_size
                            if signal.direction == "BUY":
                                self.broker.place_order(symbol, "BUY", qty)
                            elif signal.direction == "SELL":
                                self.broker.place_order(symbol, "SELL", qty)
                                
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
            self.running = False

    # --- Helpers ---
    def _detect_market_regimes(self, returns, prices):
        # Simplified Regime Detection
        return [0] * len(returns) # Default to '0' for robustness in demo
        
    def _discretize(self, values, n_states):
        return pd.qcut(values, n_states, labels=False, duplicates='drop')
        
    def _discretize_val(self, val, boundaries):
        # Find bin for single value
        if boundaries is None: return 3
        # boundaries usually from qcut are intervals, simplifying:
        # We just need consistent mapping. 
        # For this demo, we assume mapping is consistent with training.
        return 3 # Placeholder
        
    def _get_boundaries(self, values, n_states):
        # Store boundaries for runtime discretization
        try:
            _, bins = pd.qcut(values, n_states, retbins=True, duplicates='drop')
            return bins
        except:
            return None
            
    def _discretize_vol(self, returns):
        vol = pd.Series(returns).rolling(20).std().fillna(0)
        return pd.qcut(vol, 5, labels=False, duplicates='drop')
        
    def _discretize_volume(self, volumes):
        # Simple high/low relative volume
        return [1] * len(volumes) # Placeholder

def main():
    print("Optimization: Loading...")
    
    # Get user input for symbols
    try:
        print("\n--- Configuration ---")
        symbol_input = input("Enter symbols to trade (comma separated, e.g. EURUSD,AAPL) [Default: EURUSD,GBPUSD,USDJPY,AAPL]: ").strip()
        if not symbol_input:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AAPL']
            print("Using default symbols.")
        else:
            symbols = [s.strip().upper() for s in symbol_input.split(',')]
            
        lot_input = input("Enter lot size [Default: 0.01]: ").strip()
        if not lot_input:
            lot_size = 0.01
            print("Using default lot size: 0.01")
        else:
            lot_size = float(lot_input)
    except Exception as e:
        print(f"Error in input: {e}. Using defaults.")
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AAPL']
        lot_size = 0.01
    
    print(f"Trading: {symbols} with Lot Size: {lot_size}")
    
    if MT5_AVAILABLE:
        print("Connecting to MetaTrader5...")
        broker = MT5API()
        if not broker.connect():
            print("MT5 Failed. using Yahoo Simulation")
            broker = YahooFinanceAPI()
            symbols = ['AAPL', 'MSFT'] # Yahoo symbols
    else:
        broker = YahooFinanceAPI()
        symbols = ['AAPL', 'MSFT']

    bot = AdvancedMarkovTrader(broker, symbols, lot_size=lot_size)
    bot.initialize_models(lookback_days=100)
    bot.run()

if __name__ == "__main__":
    main()
