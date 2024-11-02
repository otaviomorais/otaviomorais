import tkinter as tk
from tkinter import ttk
import threading
import os
import json
import logging
import asyncio
import numpy as np
import traceback
import websockets
from datetime import datetime
from dotenv import load_dotenv
import time

class DerivTradingBot:
    def __init__(self):
        self.rsi_period = 7
        self.bollinger_period = 20
        self.bollinger_std = 2
        self.cci_period = 5

        self.api_url = "wss://ws.derivws.com/websockets/v3?app_id="
        self.app_id = os.getenv('DERIV_APP_ID')
        self.api_token = os.getenv('DERIV_API_TOKEN')
        self.history_size = 3000

        self.symbol = "R_100"
        self.stake_amount = 1.00
        self.contract_duration = 1
        self.contract_duration_unit = "m"

        self.prices = []
        self.authorized = False
        self.current_price = None
        self.last_trade_time = None
        self.min_time_between_trades = 60
        self.last_log_time = datetime.now()

        self.active_trade = False
        self.current_contract_id = None
        self.connection = None

    async def connect(self):
        try:
            self.connection = await websockets.connect(f"{self.api_url}{self.app_id}")
            await self.authorize()
            logger.info("Connected to Deriv API")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise

    async def authorize(self):
        try:
            auth_req = {"authorize": self.api_token}
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

    def calculate_rsi(self, prices):
        period = self.rsi_period
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else float('inf')
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            upval = delta if delta > 0 else 0.
            downval = -delta if delta < 0 else 0.
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else float('inf')
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    def calculate_bollinger_bands(self, prices):
        period = self.bollinger_period
        std_dev = self.bollinger_std
        rolling_mean = np.array([np.mean(prices[max(0, i-period):i]) for i in range(1, len(prices)+1)])
        rolling_std = np.array([np.std(prices[max(0, i-period):i]) for i in range(1, len(prices)+1)])
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return rolling_mean, upper_band, lower_band

    def calculate_cci(self, prices):
        period = self.cci_period
        prices_array = np.array(prices)
        typical_price = (prices_array + prices_array + prices_array) / 3
        rolling_mean = np.array([np.mean(typical_price[max(0, i-period):i]) for i in range(1, len(typical_price)+1)])
        rolling_std = np.array([np.std(typical_price[max(0, i-period):i]) for i in range(1, len(typical_price)+1)])
        rolling_std = np.where(rolling_std == 0, 1e-10, rolling_std)  # Avoid division by zero
        cci = (typical_price - rolling_mean) / (0.015 * rolling_std)
        return cci

    async def place_trade(self, direction):
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

        await asyncio.sleep(self.contract_duration * 60)  # Assuming duration is in minutes
        self.active_trade = False

    async def subscribe_to_ticks(self, symbol):
        ticks_req = {"ticks": symbol, "subscribe": 1}
        await self.connection.send(json.dumps(ticks_req))
        logger.info(f"Subscribed to ticks for {symbol}")

    async def run(self):
        while True:
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

                            if len(self.prices) >= max(self.rsi_period, self.bollinger_period, self.cci_period):
                                bb_middle, bb_upper, bb_lower = self.calculate_bollinger_bands(np.array(self.prices))
                                cci_values = self.calculate_cci(np.array(self.prices))
                                rsi_values = self.calculate_rsi(np.array(self.prices))

                                current_rsi = rsi_values[-1]
                                current_bb_middle = bb_middle[-1]
                                current_bb_upper = bb_upper[-1]
                                current_bb_lower = bb_lower[-1]
                                current_cci = cci_values[-1]
                                current_cci = cci_values[-1]

                                current_time = datetime.now()
                                if (current_time - self.last_log_time).seconds >= 5:
                                    self.last_log_time = current_time
                                    logger.info(
                                        f"Price: {price:.5f} | "
                                        f"RSI: {current_rsi:.2f} | "
                                        f"BB: {current_bb_lower:.2f}/{current_bb_middle:.2f}/{current_bb_upper:.2f} | "
                                        f"CCI: {current_cci:.2f} | "
                                        f"Active: {'Yes' if self.active_trade else 'No'}"
                                    )

                                if not self.active_trade:
                                    if price < current_bb_lower and current_rsi < 30 and current_cci < -100:
                                        logger.info("BUY Signal")
                                        await self.place_trade("BUY")
                                    elif price > current_bb_upper and current_rsi > 70 and current_cci > 100:
                                        logger.info("SELL Signal")
                                        await self.place_trade("SELL")

                        await asyncio.sleep(0.1)

                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.error(f"WebSocket connection closed: {str(e)}")
                        break
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
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

class DerivBotGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Deriv Trading Bot")
        self.root.geometry("400x600")
        
        self.bot = None
        self.bot_thread = None
        self.running = False
        
        # Create frames
        self.control_frame = ttk.LabelFrame(self.root, text="Bot Controls")
        self.control_frame.pack(padx=10, pady=5, fill="x")
        
        self.status_frame = ttk.LabelFrame(self.root, text="Status")
        self.status_frame.pack(padx=10, pady=5, fill="x")
        
        self.settings_frame = ttk.LabelFrame(self.root, text="Settings")
        self.settings_frame.pack(padx=10, pady=5, fill="x")
        
        # Add controls
        self.start_button = ttk.Button(self.control_frame, text="Start Bot", command=self.start_bot)
        self.start_button.pack(pady=5)
        
        self.stop_button = ttk.Button(self.control_frame, text="Stop Bot", command=self.stop_bot)
        self.stop_button.pack(pady=5)
        self.stop_button["state"] = "disabled"
        
        # Status labels
        self.status_label = ttk.Label(self.status_frame, text="Bot Status: Stopped")
        self.status_label.pack(pady=5)
        
        self.price_label = ttk.Label(self.status_frame, text="Current Price: --")
        self.price_label.pack(pady=5)
        
        self.rsi_label = ttk.Label(self.status_frame, text="RSI: --")
        self.rsi_label.pack(pady=5)
        
        self.bb_label = ttk.Label(self.status_frame, text="Bollinger Bands: -- / -- / --")
        self.bb_label.pack(pady=5)
        
        self.cci_label = ttk.Label(self.status_frame, text="CCI: --")
        self.cci_label.pack(pady=5)
        
        # Settings inputs
        ttk.Label(self.settings_frame, text="Stake Amount:").pack()
        self.stake_entry = ttk.Entry(self.settings_frame)
        self.stake_entry.insert(0, "1.00")
        self.stake_entry.pack()
        
        ttk.Label(self.settings_frame, text="Symbol:").pack()
        self.symbol_entry = ttk.Entry(self.settings_frame)
        self.symbol_entry.insert(0, "R_100")
        self.symbol_entry.pack()
        
        ttk.Label(self.settings_frame, text="RSI Period:").pack()
        self.rsi_period_entry = ttk.Entry(self.settings_frame)
        self.rsi_period_entry.insert(0, "7")
        self.rsi_period_entry.pack()
        
        ttk.Label(self.settings_frame, text="Bollinger Period:").pack()
        self.bollinger_period_entry = ttk.Entry(self.settings_frame)
        self.bollinger_period_entry.insert(0, "20")
        self.bollinger_period_entry.pack()
        
        ttk.Label(self.settings_frame, text="Bollinger Std Dev:").pack()
        self.bollinger_std_entry = ttk.Entry(self.settings_frame)
        self.bollinger_std_entry.insert(0, "2")
        self.bollinger_std_entry.pack()
        
        ttk.Label(self.settings_frame, text="CCI Period:").pack()
        self.cci_period_entry = ttk.Entry(self.settings_frame)
        self.cci_period_entry.insert(0, "5")
        self.cci_period_entry.pack()

    def start_bot(self):
        self.running = True
        self.bot = DerivTradingBot()
        self.bot.stake_amount = float(self.stake_entry.get())
        self.bot.symbol = self.symbol_entry.get()
        self.bot.rsi_period = int(self.rsi_period_entry.get())
        self.bot.bollinger_period = int(self.bollinger_period_entry.get())
        self.bot.bollinger_std = float(self.bollinger_std_entry.get())
        self.bot.cci_period = int(self.cci_period_entry.get())
        
        self.bot_thread = threading.Thread(target=lambda: asyncio.run(self.bot.run()))
        self.bot_thread.start()
        
        self.start_button["state"] = "disabled"
        self.stop_button["state"] = "normal"
        self.status_label["text"] = "Bot Status: Running"
        
        self.update_status()

    def stop_bot(self):
        self.running = False
        if self.bot and self.bot.connection:
            asyncio.run(self.bot.connection.close())
        
        self.start_button["state"] = "normal"
        self.stop_button["state"] = "disabled"
        self.status_label["text"] = "Bot Status: Stopped"

    def update_status(self):
        if self.running:
            self.price_label["text"] = f"Current Price: {self.bot.current_price:.5f}" if self.bot.current_price else "Current Price: --"
            self.rsi_label["text"] = f"RSI: {self.bot.calculate_rsi(self.bot.prices)[-1]:.2f}" if len(self.bot.prices) >= self.bot.rsi_period else "RSI: --"
            bb_middle, bb_upper, bb_lower = self.bot.calculate_bollinger_bands(np.array(self.bot.prices)) if len(self.bot.prices) >= self.bot.bollinger_period else (None, None, None)
            self.bb_label["text"] = f"Bollinger Bands: {bb_lower[-1]:.2f} / {bb_middle[-1]:.2f} / {bb_upper[-1]:.2f}" if bb_middle is not None else "Bollinger Bands: -- / -- / --"
            self.cci_label["text"] = f"CCI: {self.bot.calculate_cci(self.bot.prices)[-1]:.2f}" if len(self.bot.prices) >= self.bot.cci_period else "CCI: --"
            self.root.after(1000, self.update_status)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    load_dotenv()

    required_env_vars = ['DERIV_APP_ID', 'DERIV_API_TOKEN']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file")
        exit(1)

    gui = DerivBotGUI()
    gui.run()
