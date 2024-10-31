import websockets
import json
import asyncio
import numpy as np
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DerivTradingBot:
    def __init__(self):
        # Strategy parameters (converted from MQL4)
        self.rsi_length = 2
        self.half_length = 2
        self.dev_period = 100
        self.deviations = 0.7
        
        # Trading parameters
        self.api_url = "wss://ws.derivws.com/websockets/v3?app_id="
        self.app_id = os.getenv('DERIV_APP_ID')
        self.api_token = os.getenv('DERIV_API_TOKEN')
        self.history_size = 3000
        
        # Trading configuration
        self.symbol = "R_100"
        self.stake_amount = 10.00
        self.contract_duration = 5
        self.contract_duration_unit = "m"  # minutes
        self.min_price_update_count = 10  # Minimum number of price updates before trading
        
        # State tracking
        self.prices = []
        self.rs_values = []
        self.ch_mid = []
        self.ch_up = []
        self.ch_down = []
        self.authorized = False
        self.current_price = None
        self.last_trade_time = None
        self.min_time_between_trades = 60  # Minimum seconds between trades
        
        # Validate credentials
        if not self.app_id or not self.api_token:
            raise ValueError("Missing DERIV_APP_ID or DERIV_API_TOKEN environment variables")

    async def connect(self):
        """Establish WebSocket connection with Deriv"""
        try:
            full_url = f"{self.api_url}{self.app_id}"
            logger.info(f"Connecting to Deriv API...")
            self.connection = await websockets.connect(full_url)
            
            # Authenticate with API token
            auth_req = {
                "authorize": self.api_token,
            }
            await self.connection.send(json.dumps(auth_req))
            auth_response = await self.connection.recv()
            auth_data = json.loads(auth_response)
            
            if "error" in auth_data:
                raise ConnectionError(f"Authentication failed: {auth_data['error']['message']}")
            
            if "authorize" in auth_data:
                self.authorized = True
                logger.info("Successfully authenticated with Deriv API")
                balance = auth_data["authorize"]["balance"]
                currency = auth_data["authorize"]["currency"]
                logger.info(f"Account Balance: {balance} {currency}")
            else:
                raise ConnectionError("Unexpected authentication response")
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise

    def can_trade(self):
        """Check if we can place a new trade based on various conditions"""
        # Check if we have enough price history
        if len(self.prices) < self.min_price_update_count:
            return False
            
        # Check time between trades
        if self.last_trade_time is not None:
            time_since_last_trade = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last_trade < self.min_time_between_trades:
                return False
                
        return True

    async def get_price_proposal(self, contract_type):
        """Get price proposal for a contract"""
        try:
            proposal_req = {
                "proposal": 1,
                "amount": self.stake_amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": self.contract_duration,
                "duration_unit": self.contract_duration_unit,
                "symbol": self.symbol
            }
            
            await self.connection.send(json.dumps(proposal_req))
            response = await self.connection.recv()
            proposal_data = json.loads(response)
            
            if "error" in proposal_data:
                logger.error(f"Proposal failed: {proposal_data['error']['message']}")
                return None
            
            return proposal_data.get("proposal", None)
            
        except Exception as e:
            logger.error(f"Error getting price proposal: {str(e)}")
            return None

    async def place_trade(self, direction, amount=None):
        """Place a trade on Deriv with proper price proposal"""
        try:
            if not self.authorized:
                logger.error("Cannot place trade: Not authorized")
                return
                
            if not self.can_trade():
                logger.info("Skipping trade: Trading conditions not met")
                return
            
            if amount is None:
                amount = self.stake_amount
                
            contract_type = "CALL" if direction == "BUY" else "PUT"
            
            # Get price proposal first
            proposal = await self.get_price_proposal(contract_type)
            if not proposal:
                logger.error("Could not get price proposal")
                return
                
            trade_req = {
                "buy": 1,
                "price": proposal["id"],
                "parameters": proposal
            }
            
            await self.connection.send(json.dumps(trade_req))
            response = await self.connection.recv()
            trade_data = json.loads(response)
            
            if "error" in trade_data:
                logger.error(f"Trade failed: {trade_data['error']['message']}")
            else:
                logger.info(f"Trade placed successfully: {direction} {amount} USD")
                if "buy" in trade_data:
                    contract_id = trade_data["buy"]["contract_id"]
                    logger.info(f"Contract ID: {contract_id}")
                    self.last_trade_time = datetime.now()
                    
        except Exception as e:
            logger.error(f"Error placing trade: {str(e)}")

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI values with proper handling of edge cases"""
        if len(prices) <= period:
            return np.zeros_like(prices)
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average gain and loss
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        # Subsequent values using Wilder's smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period
        
        # Handle division by zero
        rs = np.where(avg_loss == 0, 100, avg_gain / (avg_loss + 1e-10))
        rsi = 100 - (100 / (1 + rs))
        
        # Replace invalid values with 50 (neutral)
        rsi = np.nan_to_num(rsi, nan=50.0)
        return rsi

    def calculate_channels(self, rs_values):
        """Calculate channel values based on RSI"""
        half_length = self.half_length
        dev_period = self.dev_period
        deviations = self.deviations
        
        ch_mid = []
        ch_up = []
        ch_down = []
        
        for i in range(len(rs_values)):
            if i >= half_length:
                weights = list(range(1, half_length + 2))
                segment = rs_values[i-half_length:i+1]
                mid = np.average(segment, weights=weights[:len(segment)])
                
                if i >= dev_period:
                    dev = np.std(rs_values[i-dev_period:i])
                    up = mid + dev * deviations
                    down = mid - dev * deviations
                else:
                    up = mid
                    down = mid
            else:
                mid = rs_values[i]
                up = mid
                down = mid
                
            ch_mid.append(mid)
            ch_up.append(up)
            ch_down.append(down)
            
        return ch_mid, ch_up, ch_down

    async def subscribe_to_ticks(self, symbol=None):
        """Subscribe to price ticks for a symbol"""
        try:
            if symbol is None:
                symbol = self.symbol
                
            if not self.authorized:
                raise ConnectionError("Not authorized. Please connect first.")
                
            subscribe_req = {
                "ticks": symbol,
                "subscribe": 1
            }
            await self.connection.send(json.dumps(subscribe_req))
            response = await self.connection.recv()
            
            data = json.loads(response)
            if "error" in data:
                raise ConnectionError(f"Tick subscription failed: {data['error']['message']}")
            logger.info(f"Successfully subscribed to {symbol} ticks")
            
        except Exception as e:
            logger.error(f"Error subscribing to ticks: {str(e)}")
            raise

    def check_signals(self, rs_values, ch_up, ch_down):
        """Check for trading signals with additional validation"""
        signals = []
        
        # Only check for signals if we have enough data
        if len(rs_values) < 2:
            return signals
            
        for i in range(1, len(rs_values)):
            # Validate RSI values
            if not (0 <= rs_values[i] <= 100) or not (0 <= rs_values[i-1] <= 100):
                continue
                
            # Buy signal
            if (rs_values[i] < ch_down[i] and 
                rs_values[i-1] > ch_down[i-1]):
                signals.append(("BUY", i))
                logger.info(f"BUY Signal detected at RSI: {rs_values[i]:.2f}")
                
            # Sell signal
            elif (rs_values[i] > ch_up[i] and 
                  rs_values[i-1] < ch_up[i-1]):
                signals.append(("SELL", i))
                logger.info(f"SELL Signal detected at RSI: {rs_values[i]:.2f}")
                
        return signals

    async def run(self):
        """Main bot loop"""
        try:
            await self.connect()
            await self.subscribe_to_ticks()
            
            while True:
                try:
                    response = await self.connection.recv()
                    data = json.loads(response)
                    
                    if "tick" in data:
                        price = data["tick"]["quote"]
                        self.current_price = price
                        self.prices.append(price)
                        
                        if len(self.prices) > self.history_size:
                            self.prices.pop(0)
                        
                        if len(self.prices) > self.rsi_length:
                            rs_values = self.calculate_rsi(np.array(self.prices), 
                                                         self.rsi_length)
                            ch_mid, ch_up, ch_down = self.calculate_channels(rs_values)
                            
                            signals = self.check_signals(rs_values, ch_up, ch_down)
                            
                            for direction, _ in signals:
                                # Fixed: Using self.place_trade instead of place_trade
                                await self.place_trade(direction)
                                
                    await asyncio.sleep(0.1)  # Add small delay to prevent CPU overuse
                    
                except Exception as e:
                    logger.error(f"Error processing tick: {str(e)}")
                    continue
                            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            if hasattr(self, 'connection'):
                await self.connection.close()

async def main():
    while True:
        try:
            bot = DerivTradingBot()
            await bot.run()
        except Exception as e:
            logger.error(f"Bot crashed with error: {str(e)}")
            logger.info("Restarting bot in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    if not os.path.exists('.env'):
        logger.error("Please create a .env file with DERIV_APP_ID and DERIV_API_TOKEN")
        exit(1)
    
    asyncio.run(main())