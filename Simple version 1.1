def demo_live_mt5_trading():
    """Demonstration of live MT5 Markov trading system with real-time data"""
    
    # Configuration for live trading
    SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']  # Major forex pairs
    
    # Create live trader
    trader = MT5LiveDataTrader(SYMBOLS, timeframe=mt5.TIMEFRAME_M15)
    
    print("=" * 80)
    print("🔴 LIVE MT5 MARKOV CHAIN TRADING SYSTEM")
    print("=" * 80)
    print(f"📊 Symbols: {SYMBOLS}")
    print(f"⏰ Timeframe: 15 minutes")
    print(f"🔄 Tick Updates: Every 1 second")
    print(f"📡 Signal Updates: Every 5 seconds")
    print("\n🚀 LIVE FEATURES:")
    print("✅ Real-time tick data streaming")
    print("✅ Dynamic spread monitoring") 
    print("✅ Live volatility adjustment")
    print("✅ Market stability detection")
    print("✅ Volume trend analysis")
    print("✅ Automatic risk management")
    print("✅ Live performance monitoring")
    
    try:
        # Test connection and show account info
        if trader.connect_mt5():
            account = trader.get_account_info()
            print(f"\n💰 ACCOUNT INFO:")
            print(f"   Balance: {account.get('balance', 0):.2f} {account.get('currency', 'USD')}")
            print(f"   Equity: {account.get('equity', 0):.2f}")
            print(f"   Free Margin: {account.get('free_margin', 0):.2f}")
            
            # Show initial market status
            print(f"\n📈 INITIAL MARKET STATUS:")
            for symbol in SYMBOLS:
                tick = trader.get_live_tick_data(symbol)
                if tick:
                    print(f"   {symbol}: {tick['mid_price']:.5f} (Spread: {tick['spread']:.5f})")
            
            print(f"\n🔴 STARTING LIVE TRADING...")
            print("📊 Live data will stream every second")
            print("📡 Signals generated based on real-time analysis")
            print("⚠️  Press Ctrl+C to stop safely")
            print("=" * 80)
            
            # Start live trading
            trader.start_live_trading()
            
        else:
            print("\n❌ CONNECTION FAILED")
            print("Please ensure:")
            print("1. 📱 MT5 terminal is running")
            print("2. 🔐 You're logged into a trading account") 
            print("3. ⚙️  Python connections enabled (Tools → Options → Expert Advisors)")
            print("4. 🌐 Internet connection is stable")
            
    except KeyboardInterrupt:
        print(f"\n🛑 SHUTDOWN INITIATED...")
        trader.stop_trading()
        
        # Show final performance
        performance = trader.get_live_performance_metrics()
        if 'error' not in performance:
            print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
            signals = performance.get('signals', {})
            positions = performance.get('positions', {})
            
            print(f"   Total Signals: {signals.get('total_recent', 0)}")
            print(f"   Buy Signals: {signals.get('buy_signals', 0)}")
            print(f"   Sell Signals: {signals.get('sell_signals', 0)}")
            print(f"   Avg Confidence: {signals.get('avg_confidence', 0):.3f}")
            print(f"   Open Positions: {positions.get('count', 0)}")
            print(f"   Total P&L: {positions.get('total_profit', 0):.2f}")
        
        # Show recent signals
        recent_signals = trader.get_recent_signals(10)
        if recent_signals:
            print(f"\n📡 LAST 5 SIGNALS:")
            for signal in recent_signals[-5:]:
                age = (datetime.now() - signal.timestamp).seconds
                print(f"   {signal.timestamp.strftime('%H:%M:%S')} - {signal.symbol}: "
                      f"{signal.direction} (Conf: {signal.confidence:.3f}) [{age}s ago]")
    
    print(f"\n✅ Live trading session completed")

def live_market_monitor():
    """Real-time market monitor without trading"""
    
    SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    trader = MT5LiveDataTrader(SYMBOLS)
    
    print("📊 LIVE MARKET MONITOR (No Trading)")
    print("=" * 60)
    
    if trader.connect_mt5():
        trader.running = True
        
        # Start data streaming
        import threading
        data_thread = threading.Thread(target=trader.stream_live_data, daemon=True)
        data_thread.start()
        
        try:
            while True:
                # Display live market data every 5 seconds
                print(f"\n🔴 LIVE DATA ({datetime.now().strftime('%H:%M:%S')}):")
                
                for symbol in SYMBOLS:
                    live_data = trader.live_prices.get(symbol, {})
                    if live_data:
                        price = live_data.get('mid_price', 0)
                        spread = live_data.get('spread', 0) * 10000  # Convert to pips
                        volume = live_data.get('volume',import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
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
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0

@dataclass
class TradingSignal:
    """Simple trading signal"""
    symbol: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    predicted_state: int
    timestamp: datetime

class SimpleMarkovModel:
    """Simple Markov chain model for price prediction"""
    
    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.transition_matrix = {}
        self.state_boundaries = None
        self.fitted = False
    
    def fit(self, returns: np.ndarray):
        """Train the model on historical returns"""
        # Discretize returns into states using quantiles
        self.state_boundaries = np.quantile(returns, np.linspace(0, 1, self.n_states + 1))
        states = np.digitize(returns, self.state_boundaries) - 1
        states = np.clip(states, 0, self.n_states - 1)
        
        # Build transition matrix
        transitions = defaultdict(Counter)
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transitions[current_state][next_state] += 1
        
        # Convert to probabilities
        self.transition_matrix = {}
        for current_state, next_counts in transitions.items():
            total = sum(next_counts.values())
            self.transition_matrix[current_state] = {
                next_state: count / total 
                for next_state, count in next_counts.items()
            }
        
        self.fitted = True
        logger.info(f"Markov model trained with {len(states)} data points")
        return self
    
    def predict(self, current_return: float) -> Dict:
        """Predict next state probabilities"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get current state
        current_state = np.digitize([current_return], self.state_boundaries) - 1
        current_state = np.clip(current_state[0], 0, self.n_states - 1)
        
        # Get transition probabilities
        if current_state in self.transition_matrix:
            return self.transition_matrix[current_state]
        else:
            # Uniform distribution if state not seen
            return {i: 1/self.n_states for i in range(self.n_states)}
    
    def get_signal(self, current_return: float) -> str:
        """Get trading signal based on prediction"""
        if not self.fitted:
            return 'HOLD'
        
        next_state_probs = self.predict(current_return)
        most_likely_state = max(next_state_probs.keys(), key=lambda k: next_state_probs[k])
        confidence = next_state_probs[most_likely_state]
        
        neutral_state = self.n_states // 2
        
        if most_likely_state > neutral_state and confidence > 0.4:
            return 'BUY'
        elif most_likely_state < neutral_state and confidence > 0.4:
            return 'SELL'
        else:
            return 'HOLD'

class MT5LiveDataTrader:
    """Live data trading system using MT5 and Markov chain predictions"""
    
    def __init__(self, symbols: List[str], timeframe=mt5.TIMEFRAME_M15):
        self.symbols = symbols
        self.timeframe = timeframe
        self.models = {}
        self.price_history = {}
        self.live_prices = {}  # Store current live prices
        self.tick_data = {symbol: [] for symbol in symbols}  # Store tick data
        self.signals = []
        self.running = False
        self.update_interval = 5  # Reduced to 5 seconds for more responsive trading
        self.last_signal_time = {}  # Track last signal time per symbol
        self.min_signal_interval = 60  # Minimum seconds between signals per symbol
        
    def connect_mt5(self, login: int = None, password: str = None, server: str = None) -> bool:
        """Connect to MT5 terminal"""
        try:
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False
            
            # Login if credentials provided
            if login and password and server:
                if not mt5.login(login, password=password, server=server):
                    logger.error("Failed to login to MT5")
                    return False
                logger.info("Successfully logged into MT5")
            else:
                logger.info("Connected to MT5 (using existing login)")
            
            # Check connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            logger.info(f"Connected to account: {account_info.login}")
            logger.info(f"Balance: {account_info.balance}")
            logger.info(f"Server: {account_info.server}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect_mt5(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        logger.info("Disconnected from MT5")
    
    def get_historical_data(self, symbol: str, count: int = 1000) -> pd.DataFrame:
        """Get historical price data from MT5"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, count)
            
            if rates is None:
                logger.error(f"Failed to get historical data for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            logger.info(f"Retrieved {len(df)} historical records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_live_tick_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time tick data from MT5"""
        try:
            # Get latest tick
            tick = mt5.symbol_info_tick(symbol)
            
            if tick is None:
                return None
            
            # Convert tick time to datetime
            tick_time = datetime.fromtimestamp(tick.time)
            
            live_data = {
                'symbol': symbol,
                'time': tick_time,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'spread': tick.ask - tick.bid,
                'mid_price': (tick.bid + tick.ask) / 2
            }
            
            return live_data
            
        except Exception as e:
            logger.error(f"Error getting live tick data for {symbol}: {e}")
            return None
    
    def stream_live_data(self):
        """Continuously stream live data and update models"""
        logger.info("Starting live data streaming...")
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Get live tick data
                    live_tick = self.get_live_tick_data(symbol)
                    
                    if live_tick:
                        # Store current live price
                        self.live_prices[symbol] = live_tick
                        
                        # Add to tick data history (keep last 1000 ticks)
                        self.tick_data[symbol].append(live_tick)
                        if len(self.tick_data[symbol]) > 1000:
                            self.tick_data[symbol] = self.tick_data[symbol][-1000:]
                        
                        # Update price history with mid prices
                        if len(self.tick_data[symbol]) > 1:
                            self.price_history[symbol].append(live_tick['mid_price'])
                            
                            # Keep only recent prices for model
                            if len(self.price_history[symbol]) > 100:
                                self.price_history[symbol] = self.price_history[symbol][-100:]
                        
                        # Update live model predictions
                        self.update_live_model(symbol)
                
                time.sleep(1)  # 1 second tick updates
                
            except Exception as e:
                logger.error(f"Error in live data streaming: {e}")
                time.sleep(5)
    
    def update_live_model(self, symbol: str):
        """Update model with latest live data and generate signals"""
        try:
            # Need at least 20 price points for meaningful signals
            if (symbol not in self.models or 
                len(self.price_history.get(symbol, [])) < 20):
                return
            
            # Check if enough time has passed since last signal
            current_time = datetime.now()
            if (symbol in self.last_signal_time and 
                (current_time - self.last_signal_time[symbol]).seconds < self.min_signal_interval):
                return
            
            # Generate live signal
            signal = self.generate_live_signal(symbol)
            
            if signal and signal.direction != 'HOLD':
                self.signals.append(signal)
                self.last_signal_time[symbol] = current_time
                
                # Log live signal with current market data
                live_data = self.live_prices.get(symbol, {})
                logger.info(f"LIVE Signal: {signal.symbol} - {signal.direction} "
                          f"(Confidence: {signal.confidence:.3f}, "
                          f"Price: {live_data.get('mid_price', 0):.5f}, "
                          f"Spread: {live_data.get('spread', 0):.5f})")
                
                # Execute trade with live data
                if signal.confidence > 0.65:  # Higher threshold for live trading
                    self.execute_live_trade(signal)
                    
        except Exception as e:
            logger.error(f"Error updating live model for {symbol}: {e}")
    
    def generate_live_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal using live data"""
        if (symbol not in self.models or 
            len(self.price_history.get(symbol, [])) < 2):
            return None
        
        try:
            model = self.models[symbol]
            prices = np.array(self.price_history[symbol])
            
            # Use more recent data for live signals
            recent_prices = prices[-10:]  # Last 10 prices
            recent_returns = np.diff(np.log(recent_prices))
            
            if len(recent_returns) == 0:
                return None
            
            # Calculate current return and volatility
            current_return = recent_returns[-1]
            recent_volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0
            
            # Get prediction from model
            next_state_probs = model.predict(current_return)
            most_likely_state = max(next_state_probs.keys(), 
                                  key=lambda k: next_state_probs[k])
            confidence = next_state_probs[most_likely_state]
            
            # Adjust confidence based on market conditions
            adjusted_confidence = self.adjust_confidence_live(
                symbol, confidence, recent_volatility
            )
            
            # Generate signal
            direction = model.get_signal(current_return)
            
            # Override to HOLD if market conditions are unfavorable
            if self.is_market_unstable(symbol):
                direction = 'HOLD'
                adjusted_confidence = 0.5
            
            return TradingSignal(
                symbol=symbol,
                direction=direction,
                confidence=adjusted_confidence,
                predicted_state=most_likely_state,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating live signal for {symbol}: {e}")
            return None
    
    def adjust_confidence_live(self, symbol: str, base_confidence: float, volatility: float) -> float:
        """Adjust signal confidence based on live market conditions"""
        try:
            live_data = self.live_prices.get(symbol, {})
            
            # Spread adjustment - wider spreads reduce confidence
            spread = live_data.get('spread', 0)
            typical_spread = self.get_typical_spread(symbol)
            spread_factor = max(0.5, 1 - (spread / typical_spread - 1)) if typical_spread > 0 else 1.0
            
            # Volatility adjustment - high volatility reduces confidence
            avg_volatility = self.get_average_volatility(symbol)
            vol_factor = max(0.3, 1 - (volatility / avg_volatility - 1)) if avg_volatility > 0 else 1.0
            
            # Volume adjustment - low volume reduces confidence
            recent_volume = self.get_recent_volume_trend(symbol)
            volume_factor = min(1.2, max(0.7, recent_volume))
            
            # Combined adjustment
            adjusted_confidence = base_confidence * spread_factor * vol_factor * volume_factor
            
            return min(1.0, max(0.0, adjusted_confidence))
            
        except Exception as e:
            logger.error(f"Error adjusting confidence for {symbol}: {e}")
            return base_confidence
    
    def get_typical_spread(self, symbol: str) -> float:
        """Calculate typical spread for symbol"""
        try:
            if symbol in self.tick_data and len(self.tick_data[symbol]) > 50:
                recent_spreads = [tick['spread'] for tick in self.tick_data[symbol][-50:]]
                return np.median(recent_spreads)
            return 0.0001  # Default spread
        except:
            return 0.0001
    
    def get_average_volatility(self, symbol: str) -> float:
        """Calculate average volatility for symbol"""
        try:
            if symbol in self.price_history and len(self.price_history[symbol]) > 20:
                prices = np.array(self.price_history[symbol][-50:])
                returns = np.diff(np.log(prices))
                return np.std(returns)
            return 0.01  # Default volatility
        except:
            return 0.01
    
    def get_recent_volume_trend(self, symbol: str) -> float:
        """Analyze recent volume trend"""
    
