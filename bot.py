import os
import json
import logging
import asyncio
import numpy as np
import traceback
import websockets
from datetime import datetime
from dotenv import load_dotenv

class DerivTradingBot:
    def __init__(self):
        # Strategy parameters
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
        self.contract_duration_unit = "m"
        
        # State tracking
        self.prices = []
        self.rs_values = []
        self.ch_mid = []
        self.ch_up = []
        self.ch_down = []
        self.authorized = False
        self.current_price = None
        self.last_trade_time = None
        self.min_time_between_trades = 60
        self.last_log_time = datetime.now()
        
        # Trade tracking
        self.active_trade = False
        self.current_contract_id = None
        self.connection = None

    async def connect(self):
        """Connect to Deriv websocket API"""
        try:
            self.connection = await websockets.connect(f"{self.api_url}{self.app_id}")
            await self.authorize()
            logger.info("Connected to Deriv API")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise

    async def authorize(self):
        """Authorize with the API token"""
        try:
            auth_req = {
                "authorize": self.api_token
            }
            await self.connection.send(json.dumps(auth_req))
            response = await self.connection.recv()
            auth_data = json.loads(response)
            
            if "error" in auth_data:
                raise Exception(f"Authorization failed: {auth_data['error']['message']}")
            
            self.authorized = True
            logger.info("Authorization successful")
        except Exception as e:
            logger.error(f"Authorization error: {str(e)}")
            raise

    def calculate_rsi(self, prices, period):
        """Calculate RSI values"""
        try:
            deltas = np.diff(prices)
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum()/period
            down = -seed[seed < 0].sum()/period
            rs = up/down if down != 0 else float('inf')
            rsi = np.zeros_like(prices)
            rsi[:period] = 100. - 100./(1. + rs)

            for i in range(period, len(prices)):
                delta = deltas[i-1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta

                up = (up*(period-1) + upval)/period
                down = (down*(period-1) + downval)/period
                rs = up/down if down != 0 else float('inf')
                rsi[i] = 100. - 100./(1. + rs)

            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return np.zeros_like(prices)

    def calculate_channels(self, rs_values):
        """Calculate channels based on RSI values"""
        try:
            period = self.half_length * 2
            ch_mid = np.zeros_like(rs_values)
            ch_up = np.zeros_like(rs_values)
            ch_down = np.zeros_like(rs_values)

            for i in range(period, len(rs_values)):
                slice_data = rs_values[i-period:i]
                ch_mid[i] = np.mean(slice_data)
                std_dev = np.std(slice_data)
                ch_up[i] = ch_mid[i] + (std_dev * self.deviations)
                ch_down[i] = ch_mid[i] - (std_dev * self.deviations)

            return ch_mid, ch_up, ch_down
        except Exception as e:
            logger.error(f"Error calculating channels: {str(e)}")
            return np.zeros_like(rs_values), np.zeros_like(rs_values), np.zeros_like(rs_values)

    def calculate_macd(self, prices):
        """Calculate MACD values"""
        try:
            ema12 = self.calculate_ema(prices, 12)
            ema26 = self.calculate_ema(prices, 26)
            macd_line = ema12 - ema26
            signal_line = self.calculate_ema(macd_line, 9)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)

    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        try:
            multiplier = 2 / (period + 1)
            ema = [data[0]]
            for price in data[1:]:
                ema.append((price - ema[-1]) * multiplier + ema[-1])
            return np.array(ema)
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return np.zeros_like(data)

    def calculate_bollinger_bands(self, data, period=20, num_std=2):
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = np.array([np.mean(data[max(0, i-period):i]) for i in range(1, len(data)+1)])
            rolling_std = np.array([np.std(data[max(0, i-period):i]) for i in range(1, len(data)+1)])
            
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            
            return rolling_mean, upper_band, lower_band
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return np.zeros_like(data), np.zeros_like(data), np.zeros_like(data)

    async def place_trade(self, direction):
        """Place a trade"""
        try:
            if self.active_trade:
                return

            contract_type = "CALL" if direction == "BUY" else "PUT"
            
            trade_req = {
                "buy": 1,
                "subscribe": 1,
                "price": self.stake_amount,
                "parameters": {
                    "amount": self.stake_amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": "USD",
                    "duration": self.contract_duration,
                    "duration_unit": self.contract_duration_unit,
                    "symbol": self.symbol
                }
            }
            
            await self.connection.send(json.dumps(trade_req))
            self.active_trade = True
            self.last_trade_time = datetime.now()
            logger.info(f"Placed {direction} trade for {self.stake_amount} USD")
            
        except Exception as e:
            logger.error(f"Error placing trade: {str(e)}")
            self.active_trade = False

    async def subscribe_to_ticks(self, symbol):
        """Subscribe to price ticks for a symbol"""
        try:
            ticks_req = {
                "ticks": symbol,
                "subscribe": 1
            }
            await self.connection.send(json.dumps(ticks_req))
            logger.info(f"Subscribed to ticks for {symbol}")
        except Exception as e:
            logger.error(f"Error subscribing to ticks: {str(e)}")
            raise

    async def run(self):
        """Main bot loop"""
        try:
            await self.connect()
            await self.subscribe_to_ticks(self.symbol)
            
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
                            # Calculate all indicators
                            rs_values = self.calculate_rsi(np.array(self.prices), self.rsi_length)
                            ch_mid, ch_up, ch_down = self.calculate_channels(rs_values)
                            macd_line, signal_line, histogram = self.calculate_macd(np.array(self.prices))
                            bb_middle, bb_upper, bb_lower = self.calculate_bollinger_bands(np.array(self.prices))
                            ema50 = self.calculate_ema(np.array(self.prices), 50)
                            ema200 = self.calculate_ema(np.array(self.prices), 200)
                            
                            # Get current indicator values
                            current_rsi = rs_values[-1]
                            current_macd = macd_line[-1]
                            current_signal = signal_line[-1]
                            current_hist = histogram[-1]
                            current_bb_middle = bb_middle[-1] if len(bb_middle) > 0 else None
                            current_bb_upper = bb_upper[-1] if len(bb_upper) > 0 else None
                            current_bb_lower = bb_lower[-1] if len(bb_lower) > 0 else None
                            current_ema50 = ema50[-1]
                            current_ema200 = ema200[-1]
                            
                            # Log indicators every 5 seconds
                            current_time = datetime.now()
                            if (current_time - self.last_log_time).seconds >= 5:
                                self.last_log_time = current_time
                                logger.info(
                                    f"Price: {price:.5f} | "
                                    f"RSI: {current_rsi:.2f} | "
                                    f"MACD: {current_macd:.2f}/{current_signal:.2f} | "
                                    f"BB: {current_bb_lower:.2f}/{current_bb_middle:.2f}/{current_bb_upper:.2f} | "
                                    f"EMA: 50={current_ema50:.2f}/200={current_ema200:.2f} | "
                                    f"Active: {'Yes' if self.active_trade else 'No'}"
                                )
                            
                            # Only check for signals if no active trade
                            if not self.active_trade:
                                # Buy Signals (Multiple conditions must be met)
                                buy_conditions = [
                                    current_rsi < 30,  # RSI oversold
                                    price < ch_down[-1],  # Price below lower channel
                                    current_macd > current_signal,  # MACD crossover
                                    current_hist > 0,  # MACD histogram positive
                                    price < current_bb_lower if current_bb_lower else False,  # Price below BB lower
                                    current_ema50 > current_ema200  # Golden cross (uptrend)
                                ]
                                
                                # Sell Signals (Multiple conditions must be met)
                                sell_conditions = [
                                    current_rsi > 70,  # RSI overbought
                                    price > ch_up[-1],  # Price above upper channel
                                    current_macd < current_signal,  # MACD crossover
                                    current_hist < 0,  # MACD histogram negative
                                    price > current_bb_upper if current_bb_upper else False,  # Price above BB upper
                                    current_ema50 < current_ema200  # Death cross (downtrend)
                                ]

                                                                # Generate signals only if majority of conditions are met
                                if sum(buy_conditions) >= 4:  # At least 4 out of 6 conditions
                                    logger.info(
                                        f"BUY Signal | "
                                        f"RSI: {current_rsi:.2f} | "
                                        f"MACD: {current_macd:.2f} | "
                                        f"BB: {current_bb_middle:.2f} | "
                                        f"EMA50/200: {current_ema50:.2f}/{current_ema200:.2f}"
                                    )
                                    await self.place_trade("BUY")
                                elif sum(sell_conditions) >= 4:  # At least 4 out of 6 conditions
                                    logger.info(
                                        f"SELL Signal | "
                                        f"RSI: {current_rsi:.2f} | "
                                        f"MACD: {current_macd:.2f} | "
                                        f"BB: {current_bb_middle:.2f} | "
                                        f"EMA50/200: {current_ema50:.2f}/{current_ema200:.2f}"
                                    )
                                    await self.place_trade("SELL")
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing tick: {str(e)}")
                    traceback.print_exc()
                    continue
                            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            traceback.print_exc()
        finally:
            if hasattr(self, 'connection'):
                await self.connection.close()

    async def connect(self):
        """Connect to Deriv websocket API"""
        try:
            self.connection = await websockets.connect(f"{self.api_url}{self.app_id}")
            
            # Authorize
            auth_req = {
                "authorize": self.api_token
            }
            await self.connection.send(json.dumps(auth_req))
            response = await self.connection.recv()
            
            if "error" in json.loads(response):
                raise Exception(f"Authorization failed: {json.loads(response)['error']['message']}")
            
            self.authorized = True
            logger.info("Successfully connected and authorized")
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise

    async def place_trade(self, direction):
        """Place a trade with the specified direction"""
        try:
            # Check if enough time has passed since last trade
            if self.last_trade_time:
                time_since_last_trade = (datetime.now() - self.last_trade_time).seconds
                if time_since_last_trade < self.min_time_between_trades:
                    logger.info(f"Skipping trade, only {time_since_last_trade}s since last trade")
                    return

            # Prepare the contract request
            contract_type = "CALL" if direction == "BUY" else "PUT"
            
            contract_req = {
                "proposal": 1,
                "amount": self.stake_amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": self.contract_duration,
                "duration_unit": self.contract_duration_unit,
                "symbol": self.symbol
            }

            # Send the proposal request
            await self.connection.send(json.dumps(contract_req))
            response = await self.connection.recv()
            proposal = json.loads(response)

            if "error" in proposal:
                raise Exception(f"Proposal error: {proposal['error']['message']}")

            # Buy the contract
            buy_req = {
                "buy": proposal["proposal"]["id"],
                "price": self.stake_amount
            }

            await self.connection.send(json.dumps(buy_req))
            response = await self.connection.recv()
            result = json.loads(response)

            if "error" in result:
                raise Exception(f"Buy error: {result['error']['message']}")

            # Update trade tracking
            self.active_trade = True
            self.current_contract_id = result["buy"]["contract_id"]
            self.last_trade_time = datetime.now()

            logger.info(
                f"Trade placed | "
                f"Direction: {direction} | "
                f"Contract ID: {self.current_contract_id} | "
                f"Price: {self.current_price:.5f}"
            )

            # Start monitoring the trade
            await self.monitor_trade(self.current_contract_id)

        except Exception as e:
            logger.error(f"Error placing trade: {str(e)}")
            traceback.print_exc()

    async def monitor_trade(self, contract_id):
        """Monitor an active trade until it's completed"""
        try:
            # Subscribe to contract updates
            proposal_open_contract_req = {
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            }

            await self.connection.send(json.dumps(proposal_open_contract_req))

            while self.active_trade:
                response = await self.connection.recv()
                contract_update = json.loads(response)

                if "error" in contract_update:
                    raise Exception(f"Contract monitoring error: {contract_update['error']['message']}")

                if "proposal_open_contract" in contract_update:
                    contract = contract_update["proposal_open_contract"]

                    # Check if contract is finished
                    if contract["status"] in ["sold", "expired"]:
                        profit_amount = float(contract["profit"])
                        logger.info(
                            f"Trade completed | "
                            f"Contract ID: {contract_id} | "
                            f"Profit: ${profit_amount:.2f} | "
                            f"Status: {contract['status']}"
                        )
                        
                        # Reset trade tracking
                        self.active_trade = False
                        self.current_contract_id = None
                        break

        except Exception as e:
            logger.error(f"Error monitoring trade: {str(e)}")
            traceback.print_exc()
            self.active_trade = False
            self.current_contract_id = None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Load environment variables
    load_dotenv()

    # Check for required environment variables
    required_env_vars = ['DERIV_APP_ID', 'DERIV_API_TOKEN']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file")
        exit(1)

    # Start the bot
    while True:
        try:
            bot = DerivTradingBot()
            asyncio.run(bot.run())
        except Exception as e:
            logger.error(f"Bot crashed with error: {str(e)}")
            traceback.print_exc()
            logger.info("Restarting bot in 5 seconds...")
            time.sleep(5)
