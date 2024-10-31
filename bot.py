import websockets
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from config import *

# Configuração de logging avançada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_logs.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BOB5KingsBot:
    def __init__(self):
        self.app_id = DERIV_APP_ID
        self.token = DERIV_API_TOKEN
        self.ws_url = f"{DERIV_WS_URL}?app_id={self.app_id}"
        self.websocket = None
        
        # Parâmetros BOB 5 KINGS
        self.rsi_length = 2
        self.half_length = 2
        self.dev_period = 100
        self.deviations = 0.7
        
        # Gestão de risco
        self.max_daily_trades = 10
        self.max_consecutive_losses = 3
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.total_profit = 0
        self.win_rate = 0
        self.trades_history = []
        
        self.balance = 0
        self.active_trade = False
        self.last_trade_result = None

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.ws_url)
            
            auth_req = {
                "authorize": self.token
            }
            
            await self.websocket.send(json.dumps(auth_req))
            auth_response = await self.websocket.recv()
            auth_data = json.loads(auth_response)
            
            if 'error' in auth_data:
                logger.error(f"Erro de autenticação: {auth_data['error']}")
                return False
            
            self.balance = auth_data.get('authorize', {}).get('balance', 0)
            logger.info(f"Conectado com sucesso! Saldo inicial: {self.balance}")
            return True
            
        except Exception as e:
            logger.error(f"Erro na conexão: {str(e)}")
            return False

    async def get_candles(self, symbol, count=100):
        try:
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": 60
            }
            
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if 'error' in data:
                logger.error(f"Erro ao obter candles: {data['error']}")
                return None
            
            df = pd.DataFrame(data['candles'])
            df['time'] = pd.to_datetime(df['epoch'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter candles: {str(e)}")
            return None

    def calculate_signals(self, df):
        try:
            # Calcular RSI
            close_prices = df['close']
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_length).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_length).mean()
            rs = gain / loss
            df['RS'] = 100 - (100 / (1 + rs))
            
            # Calcular Half Length Moving Average (ChMid)
            weights = np.array([(self.half_length + 1 - i) for i in range(self.half_length + 1)])
            df['ChMid'] = df['RS'].rolling(window=self.half_length + 1).apply(
                lambda x: np.sum(x * weights) / np.sum(weights)
            )
            
            # Calcular desvio padrão e bandas
            df['RS_std'] = df['RS'].rolling(window=self.dev_period).std()
            df['ChUp'] = df['ChMid'] + df['RS_std'] * self.deviations
            df['ChDn'] = df['ChMid'] - df['RS_std'] * self.deviations
            
            return df
            
        except Exception as e:
            logger.error(f"Erro no cálculo de sinais: {str(e)}")
            return None

    def check_signals(self, df):
        try:
            if self.daily_trades >= self.max_daily_trades:
                logger.info("Limite diário de trades atingido")
                return None
                
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.warning("Máximo de perdas consecutivas atingido")
                return None
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Sinal de compra
            if (current['RS'] < current['ChDn'] and 
                previous['RS'] > previous['ChDn']):
                return "CALL"
            
            # Sinal de venda
            if (current['RS'] > current['ChUp'] and 
                previous['RS'] < previous['ChUp']):
                return "PUT"
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na verificação de sinais: {str(e)}")
            return None

    async def place_trade(self, direction, amount, symbol):
        if self.active_trade:
            return None
            
        try:
            request = {
                "buy": 1,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": direction,
                    "currency": "USD",
                    "duration": 5,
                    "duration_unit": "m",
                    "symbol": symbol
                }
            }
            
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            trade_data = json.loads(response)
            
            if 'error' in trade_data:
                logger.error(f"Erro ao executar trade: {trade_data['error']}")
                return None
            
            contract_id = trade_data.get('buy', {}).get('contract_id')
            if contract_id:
                self.active_trade = True
                self.daily_trades += 1
                
                logger.info(f"Trade {direction} executado: Contract ID {contract_id}")
                
                # Aguardar resultado do trade
                await self.monitor_trade(contract_id)
            
            return trade_data
            
        except Exception as e:
            logger.error(f"Erro ao executar trade: {str(e)}")
            return None

    async def monitor_trade(self, contract_id):
        try:
            request = {
                "proposal_open_contract": 1,
                "contract_id": contract_id
            }
            
            while self.active_trade:
                await self.websocket.send(json.dumps(request))
                response = await self.websocket.recv()
                data = json.loads(response)
                
                if 'error' in data:
                    logger.error(f"Erro ao monitorar trade: {data['error']}")
                    break
                
                contract = data.get('proposal_open_contract', {})
                
                if contract.get('is_sold', 0) == 1:
                    profit = contract.get('profit', 0)
                    self.total_profit += profit
                    
                    if profit > 0:
                        self.consecutive_losses = 0
                        logger.info(f"Trade GANHO! Lucro: ${profit}")
                    else:
                        self.consecutive_losses += 1
                        logger.info(f"Trade PERDIDO! Perda: ${profit}")
                    
                    self.trades_history.append({
                        'time': datetime.now(),
                        'profit': profit,
                        'contract_id': contract_id
                    })
                    
                    self.update_statistics()
                    self.active_trade = False
                    break
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Erro no monitoramento do trade: {str(e)}")
            self.active_trade = False

    def update_statistics(self):
        if not self.trades_history:
            return
            
        total_trades = len(self.trades_history)
        winning_trades = len([t for t in self.trades_history if t['profit'] > 0])
        self.win_rate = (winning_trades / total_trades) * 100
        
        logger.info(f"""
        === Estatísticas Atualizadas ===
        Total de Trades: {total_trades}
        Taxa de Acerto: {self.win_rate:.2f}%
        Lucro Total: ${self.total_profit:.2f}
        Perdas Consecutivas: {self.consecutive_losses}
        =============================
        """)

    async def run(self, symbol, amount):
        if not await self.connect():
            return
            
        logger.info(f"BOB 5 KINGS iniciado - Symbol: {symbol}, Amount: {amount}")
        
        while True:
            try:
                # Obter dados
                df = await self.get_candles(symbol)
                if df is None:
                    continue
                    
                # Calcular indicadores
                df = self.calculate_signals(df)
                if df is None:
                    continue
                
                # Verificar sinais
                signal = self.check_signals(df)
                
                if signal and not self.active_trade:
                    logger.info(f"Sinal encontrado: {signal}")
                    await self.place_trade(signal, amount, symbol)
                
                # Log dos valores atuais
                last_row = df.iloc[-1]
                logger.info(f"""
                === Valores Atuais ===
                Preço: {last_row['close']:.5f}
                RSI: {last_row['RS']:.2f}
                Banda Superior: {last_row['ChUp']:.2f}
                Média: {last_row['ChMid']:.2f}
                Banda Inferior: {last_row['ChDn']:.2f}
                ====================
                """)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Erro no loop principal: {str(e)}")
                await asyncio.sleep(5)

async def main():
    bot = BOB5KingsBot()
    await bot.run(SYMBOL, TRADE_AMOUNT)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot encerrado pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")