volume = live_data.get('volume', 0)
                        stable = "✅" if not trader.is_market_unstable(symbol) else "⚠️"
                        
                        print(f"   {stable} {symbol}: {price:.5f} | Spread: {spread:.1f} pips | Vol: {volume}")
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped")
            trader.running = False
        finally:
            trader.disconnect_mt5()
    else:
        print("❌ Failed to connect to MT5")

def advanced_live_analysis():
    """Advanced live market analysis with detailed metrics"""
    
    SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY']
    trader = MT5LiveDataTrader(SYMBOLS)
    
    print("🔬 ADVANCED LIVE MARKET ANALYSIS")
    print("=" * 70)
    
    if not trader.connect_mt5():
        print("❌ Failed to connect to MT5")
        return
    
    # Initialize models for analysis
    trader.initialize_models(lookback_periods=1000)
    trader.running = True
    
    # Start live data streaming
    import threading
    data_thread = threading.Thread(target=trader.stream_live_data, daemon=True)
    data_thread.start()
    
    # Wait for initial data
    print("⏳ Collecting initial live data...")
    time.sleep(10)
    
    try:
        analysis_count = 0
        while True:
            analysis_count += 1
            
            print(f"\n📊 ANALYSIS #{analysis_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 70)
            
            for symbol in SYMBOLS:
                print(f"\n🔍 {symbol} DETAILED ANALYSIS:")
                
                # Current market data
                live_data = trader.live_prices.get(symbol, {})
                if not live_data:
                    print("   ❌ No live data available")
                    continue
                
                # Basic metrics
                price = live_data.get('mid_price', 0)
                spread = live_data.get('spread', 0) * 10000  # Pips
                bid = live_data.get('bid', 0)
                ask = live_data.get('ask', 0)
                
                print(f"   💰 Price: {price:.5f} | Bid: {bid:.5f} | Ask: {ask:.5f}")
                print(f"   📏 Spread: {spread:.1f} pips")
                
                # Volatility analysis
                volatility = trader.get_average_volatility(symbol)
                vol_pct = volatility * 100
                
                # Volume analysis
                volume_trend = trader.get_recent_volume_trend(symbol)
                volume_status = "🔥 High" if volume_trend > 1.5 else "📉 Low" if volume_trend < 0.7 else "➡️ Normal"
                
                print(f"   📈 Volatility: {vol_pct:.3f}% | Volume: {volume_status} ({volume_trend:.2f}x)")
                
                # Market stability
                is_stable = not trader.is_market_unstable(symbol)
                stability_status = "✅ Stable" if is_stable else "⚠️ Unstable"
                
                print(f"   🎯 Market Status: {stability_status}")
                
                # Generate signal analysis
                if symbol in trader.models:
                    signal = trader.generate_live_signal(symbol)
                    if signal:
                        direction_emoji = "🟢" if signal.direction == 'BUY' else "🔴" if signal.direction == 'SELL' else "⚪"
                        conf_level = "🔥 High" if signal.confidence > 0.7 else "📊 Medium" if signal.confidence > 0.5 else "📉 Low"
                        
                        print(f"   🎯 Signal: {direction_emoji} {signal.direction} | Confidence: {conf_level} ({signal.confidence:.3f})")
                        
                        # Signal quality assessment
                        if signal.confidence > 0.65 and is_stable:
                            print(f"   ✅ TRADE READY - High quality signal")
                        elif signal.confidence > 0.5:
                            print(f"   ⚠️ MODERATE - Consider market conditions")
                        else:
                            print(f"   ❌ LOW QUALITY - Avoid trading")
                    else:
                        print(f"   ⏳ No signal generated")
                
                # Price movement prediction
                if len(trader.price_history.get(symbol, [])) > 5:
                    recent_prices = trader.price_history[symbol][-5:]
                    price_trend = "📈 Rising" if recent_prices[-1] > recent_prices[0] else "📉 Falling"
                    price_change = ((recent_prices[-1] / recent_prices[0]) - 1) * 100
                    
                    print(f"   📊 5-Period Trend: {price_trend} ({price_change:+.3f}%)")
                
                # Risk assessment
                typical_spread = trader.get_typical_spread(symbol) * 10000
                spread_ratio = spread / typical_spread if typical_spread > 0 else 1
                
                risk_level = "🟢 Low" if spread_ratio < 1.5 and is_stable else "🟡 Medium" if spread_ratio < 2.5 else "🔴 High"
                print(f"   ⚖️ Risk Level: {risk_level} | Spread Ratio: {spread_ratio:.2f}x")
            
            # Overall market summary
            print(f"\n📈 OVERALL MARKET SUMMARY:")
            stable_count = sum(1 for symbol in SYMBOLS if not trader.is_market_unstable(symbol))
            print(f"   Stable Markets: {stable_count}/{len(SYMBOLS)}")
            
            # Recent signals summary
            recent_signals = [s for s in trader.signals if 
                            (datetime.now() - s.timestamp).seconds < 300]  # Last 5 minutes
            
            if recent_signals:
                buy_count = len([s for s in recent_signals if s.direction == 'BUY'])
                sell_count = len([s for s in recent_signals if s.direction == 'SELL'])
                avg_conf = np.mean([s.confidence for s in recent_signals])
                
                print(f"   Recent Signals (5min): {len(recent_signals)} total | {buy_count} BUY | {sell_count} SELL")
                print(f"   Average Confidence: {avg_conf:.3f}")
            else:
                print(f"   No recent signals")
            
            print(f"\n⏰ Next analysis in 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Analysis stopped")
    finally:
        trader.running = False
        trader.disconnect_mt5()
        print(f"✅ Advanced analysis session completed")

if __name__ == "__main__":
    print("=" * 80)
    print("🔴 LIVE MT5 MARKOV TRADING SYSTEM WITH REAL-TIME DATA")
    print("=" * 80)
    print("⚠️  IMPORTANT DISCLAIMERS:")
    print("1. 🚨 This system uses LIVE market data and can execute real trades!")
    print("2. 💰 Only use with DEMO accounts for testing")
    print("3. 📊 Real-time data requires stable internet connection")
    print("4. ⚡ System reacts to live market movements instantly")
    print("5. 🔒 Ensure proper risk management settings")
    print("6. 📱 Keep MT5 terminal running during operation")
    print("=" * 80)
    print()
    
    # Enhanced menu options
    print("🎯 SELECT OPERATION MODE:")
    print("1. 🔴 Live Trading System (Full Auto-Trading)")
    print("2. 📊 Live Market Monitor (Data Only)")
    print("3. 🔬 Advanced Live Analysis (Detailed Metrics)")
    print("4. 🧪 Test MT5 Connection")
    print("5. ❌ Exit")
    
    try:
        choice = input("\n👉 Enter choice (1-5): ")
        
        if choice == "1":
            print(f"\n🚨 LIVE TRADING MODE SELECTED!")
            print("⚠️  This will execute REAL trades based on live data!")
            confirm = input("⚠️  Type 'LIVE-DEMO' to confirm you're using DEMO account: ")
            
            if confirm.upper() == 'LIVE-DEMO':
                print(f"\n🔴 Starting Live Trading System...")
                demo_live_mt5_trading()
            else:
                print("❌ Live demo confirmation not provided. Exiting for safety.")
                
        elif choice == "2":
            print(f"\n📊 Starting Live Market Monitor...")
            live_market_monitor()
            
        elif choice == "3":
            print(f"\n🔬 Starting Advanced Live Analysis...")
            advanced_live_analysis()
            
        elif choice == "4":
            print(f"\n🧪 Testing MT5 Connection...")
            test_mt5_connection()
            
        else:
            print("👋 Exiting...")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

# Enhanced installation and setup guide
print(f"""
📋 INSTALLATION & SETUP GUIDE:
================================

1. 📦 INSTALL REQUIRED PACKAGES:
   pip install MetaTrader5 numpy pandas

2. 🏗️ MT5 TERMINAL SETUP:
   • Download MT5 from MetaQuotes
   • Open demo account with any broker
   • Login to demo account

3. ⚙️ PYTHON INTEGRATION SETUP:
   • Tools → Options → Expert Advisors
   • ✅ Check "Allow automated trading"
   • ✅ Check "Allow DLL imports"
   • ✅ Add Python.exe to allowed programs

4. 🌐 NETWORK REQUIREMENTS:
   • Stable internet connection (< 100ms latency)
   • Firewall exceptions for MT5
   • No VPN interference

5. 🔴 LIVE TRADING FEATURES:
   • ⚡ 1-second tick updates
   • 📊 Real-time spread monitoring
   • 🎯 Dynamic volatility adjustment
   • 📈 Live volume analysis
   • 🛡️ Market stability detection
   • 💰 Automatic position sizing
   • def demo_live_mt5_trading():
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
    
    def get_historical_data(self
