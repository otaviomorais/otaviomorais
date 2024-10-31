import websockets
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, UTC, time
import logging
import sys
import sqlite3
from pathlib import Path
import aiosqlite
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import signal
from decimal import Decimal

# Configurações
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

@dataclass
class TradeRecord:
    """Classe para armazenar informações de trades"""
    contract_id: str
    time: datetime
    symbol: str
    direction: str
    amount: float
    profit: float
    entry_price: float
    exit_price: float
    status: str

class DatabaseManager:
    """Gerenciador de banco de dados"""
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Cria tabelas necessárias se não existirem"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    contract_id TEXT PRIMARY KEY,
                    time TIMESTAMP,
                    symbol TEXT,
                    direction TEXT,
                    amount REAL,
                    profit REAL,
                    entry_price REAL,
                    exit_price REAL,
                    status TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    async def save_trade(self, trade: TradeRecord):
        """Salva um trade no banco de dados"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO trades
                VALUES (:contract_id, :time, :symbol, :direction, :amount, 
                        :profit, :entry_price, :exit_price, :status)
            """, asdict(trade))
            await db.commit()

    async def load_trades(self, limit: int = 100) -> List[TradeRecord]:
        """Carrega trades do banco de dados"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM trades ORDER BY time DESC LIMIT ?", 
                (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [TradeRecord(*row) for row in rows]

    async def save_bot_state(self, state: Dict):
        """Salva o estado do bot"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM bot_state")
            for key, value in state.items():
                await db.execute(
                    "INSERT INTO bot_state (key, value) VALUES (?, ?)",
                    (key, json.dumps(value))
                )
            await db.commit()

    async def load_bot_state(self) -> Dict:
        """Carrega o estado do bot"""
        state = {}
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT key, value FROM bot_state") as cursor:
                async for key, value in cursor:
                    state[key] = json.loads(value)
        return state

class CircuitBreaker:
    """Implementação de Circuit Breaker para segurança"""
    def __init__(self):
        self.max_daily_loss = MAX_DAILY_LOSS
        self.max_consecutive_losses = MAX_CONSECUTIVE_LOSSES
        self.max_daily_trades = MAX_DAILY_TRADES
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.daily_loss = 0
        self.last_reset = datetime.now(UTC)
        self.is_broken = False

    def reset_daily(self):
        """Reseta contadores diários"""
        now = datetime.now(UTC)
        if now.date() > self.last_reset.date():
            self.daily_trades = 0
            self.daily_loss = 0
            self.last_reset = now

    def check(self, last_trade_profit: Optional[float] = None) -> bool:
        """Verifica se o circuit breaker deve ser ativado"""
        self.reset_daily()

        if last_trade_profit is not None:
            if last_trade_profit < 0:
                self.consecutive_losses += 1
                self.daily_loss += abs(last_trade_profit)
            else:
                self.consecutive_losses = 0

        conditions = [
            (self.daily_loss >= self.max_daily_loss, "Limite de perda diária atingido"),
            (self.consecutive_losses >= self.max_consecutive_losses, "Máximo de perdas consecutivas atingido"),
            (self.daily_trades >= self.max_daily_trades, "Limite de trades diários atingido")
        ]

        for condition, message in conditions:
            if condition:
                logger.warning(f"Circuit Breaker ativado: {message}")
                self.is_broken = True
                return True

        return False

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
        
        # Gestão de risco e estado
        self.circuit_breaker = CircuitBreaker()
        self.db_manager = DatabaseManager()
        self.active_trade = False
        self.last_trade_result = None
        self.balance = 0
        self.trades_history = []
        self.total_profit = 0
        self.win_rate = 0
        
        # Configurações de timeout e reconexão
        self.websocket_timeout = 30
        self.max_reconnect_attempts = 3
        self.reconnect_delay = 5
        
        # Signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Manipula o desligamento gracioso do bot"""
        logger.info("Iniciando desligamento gracioso...")
        asyncio.create_task(self.cleanup())

    async def cleanup(self):
        """Limpa recursos e salva estado antes de desligar"""
        if self.active_trade:
            logger.warning("Aguardando conclusão do trade ativo...")
            for _ in range(10):  # Espera até 10 segundos
                if not self.active_trade:
                    break
                await asyncio.sleep(1)

        await self.save_state()
        if self.websocket:
            await self.websocket.close()
        logger.info("Bot encerrado com sucesso")
        sys.exit(0)

    async def save_state(self):
        """Salva o estado atual do bot"""
        state = {
            'total_profit': self.total_profit,
            'win_rate': self.win_rate,
            'circuit_breaker_state': {
                'consecutive_losses': self.circuit_breaker.consecutive_losses,
                'daily_trades': self.circuit_breaker.daily_trades,
                'daily_loss': self.circuit_breaker.daily_loss,
                'last_reset': self.circuit_breaker.last_reset.isoformat()
            }
        }
        await self.db_manager.save_bot_state(state)

    async def load_state(self):
        """Carrega o estado salvo do bot"""
        state = await self.db_manager.load_bot_state()
        if state:
            self.total_profit = state.get('total_profit', 0)
            self.win_rate = state.get('win_rate', 0)
            cb_state = state.get('circuit_breaker_state', {})
            self.circuit_breaker.consecutive_losses = cb_state.get('consecutive_losses', 0)
            self.circuit_breaker.daily_trades = cb_state.get('daily_trades', 0)
            self.circuit_breaker.daily_loss = cb_state.get('daily_loss', 0)
            if 'last_reset' in cb_state:
                self.circuit_breaker.last_reset = datetime.fromisoformat(cb_state['last_reset'])

    async def connect(self) -> bool:
        """Estabelece conexão com WebSocket com retry"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                self.websocket = await asyncio.wait_for(
                    websockets.connect(self.ws_url),
                    timeout=self.websocket_timeout
                )
                
                auth_req = {
                    "authorize": self.token
                }
                
                await self.websocket.send(json.dumps(auth_req))
                auth_response = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.websocket_timeout
                )
                auth_data = json.loads(auth_response)
                
                if 'error' in auth_data:
                    logger.error(f"Erro de autenticação: {auth_data['error']}")
                    continue
                
                self.balance = auth_data.get('authorize', {}).get('balance', 0)
                logger.info(f"Conectado com sucesso! Saldo inicial: {self.balance}")
                return True
                
            except Exception as e:
                logger.error(f"Tentativa {attempt + 1} falhou: {str(e)}")
                if self.websocket:
                    await self.websocket.close()
                if attempt < self.max_reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay * (attempt + 1))
                
        return False

    async def get_candles(self, symbol: str, count: int = 150) -> Optional[pd.DataFrame]:
        """Obtém dados de candles com timeout"""
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
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.websocket_timeout
            )
            data = json.loads(response)
            
            if 'error' in data:
                logger.error(f"Erro ao obter candles: {data['error']}")
                return None
            
            df = pd.DataFrame(data['candles'])
            df['time'] = pd.to_datetime(df['epoch'], unit='s')
            return df
            
        except asyncio.TimeoutError:
            logger.error("Timeout ao obter candles")
            return None
        except Exception as e:
            logger.error(f"Erro ao obter candles: {str(e)}")
            return None

    def calculate_signals(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calcula indicadores técnicos com otimizações"""
        try:
            if len(df) < max(self.rsi_length, self.half_length, self.dev_period):
                logger.warning("Dados insuficientes para cálculos")
                return None

            # Usar numpy para otimizar cálculos
            close_prices = df['close'].astype(float).values
            delta = np.diff(close_prices, prepend=close_prices[0])
            
            # Calcular ganhos e perdas
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            # Calcular médias móveis de ganhos e perdas
            avg_gains = pd.Series(gains).rolling(window=self.rsi_length, min_periods=1).mean().values
            avg_losses = pd.Series(losses).rolling(window=self.rsi_length, min_periods=1).mean().values
            
            # Evitar divisão por zero
            avg_losses = np.where(avg_losses == 0, np.nan, avg_losses)
            rs = avg_gains / avg_losses
            rs = np.nan_to_num(rs, 0)
            
            df['RS'] = 100 - (100 / (1 + rs))
            
            # Calcular Half Length Moving Average (ChMid)
            weights = np.array([(self.half_length + 1 - i) for i in range(self.half_length + 1)])
            df['ChMid'] = df['RS'].rolling(window=self.half_length + 1, min_periods=1).apply(
                lambda x: np.sum(weights[:len(x)] * x) / np.sum(weights[:len(x)])
            )
            
            # Calcular desvio padrão e bandas
            df['RS_std'] = df['RS'].rolling(window=self.dev_period, min_periods=1).std()
            df['ChUp'] = df['ChMid'] + df['RS_std'] * self.deviations
            df['ChDn'] = df['ChMid'] - df['RS_std'] * self.deviations
            
            # Usar bfill() em vez de fillna(method='bfill')
            df = df.bfill()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro no cálculo de sinais: {str(e)}")
            return None

    def check_signals(self, df: pd.DataFrame) -> Optional[str]:
        """Verifica sinais de trading com validações adicionais"""
        try:
            # Verificar circuit breaker
            if self.circuit_breaker.is_broken:
                logger.warning("Circuit breaker ativo - ignorando sinais")
                return None
                
            if len(df) < 2:
                logger.warning("Dados insuficientes para verificar sinais")
                return None

            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Verificar se os valores são válidos
            required_columns = ['RS', 'ChUp', 'ChDn']
            if any(pd.isna(current[col]) or pd.isna(previous[col]) for col in required_columns):
                logger.warning("Dados inválidos para verificação de sinais")
                return None
            
            # Validar volatilidade
            volatility = df['close'].pct_change().std()
            if volatility > MAX_VOLATILITY:
                logger.warning(f"Volatilidade muito alta: {volatility:.2f}")
                return None
            
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
            logger.error(f"Erro ao verificar sinais: {str(e)}")
            return None

    async def place_trade(self, symbol: str, direction: str, amount: float) -> Optional[str]:
        """Executa uma operação de trading"""
        try:
            if self.active_trade:
                logger.warning("Já existe um trade ativo")
                return None
            
            # Verificar horário de trading
            current_time = datetime.now(UTC).time()
            if not (TRADING_START_TIME <= current_time <= TRADING_END_TIME):
                logger.info("Fora do horário de trading")
                return None
            
            # Preparar requisição
            contract_type = "CALL" if direction == "CALL" else "PUT"
            request = {
                "buy": 1,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": "USD",
                    "duration": 1,
                    "duration_unit": "m",
                    "symbol": symbol
                },
                "price": amount
            }
            
            # Enviar ordem
            await self.websocket.send(json.dumps(request))
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.websocket_timeout
            )
            data = json.loads(response)
            
            if 'error' in data:
                logger.error(f"Erro ao executar trade: {data['error']}")
                return None
                
            contract_id = data.get('buy', {}).get('contract_id')
            if contract_id:
                self.active_trade = True
                logger.info(f"Trade iniciado: {contract_id}")
                return contract_id
                
            return None
            
        except Exception as e:
            logger.error(f"Erro ao executar trade: {str(e)}")
            return None

    async def monitor_trade(self, contract_id: str) -> Optional[TradeRecord]:
        """Monitora o resultado de um trade"""
        try:
            request = {
                "proposal_open_contract": 1,
                "contract_id": contract_id
            }
            
            while self.active_trade:
                await self.websocket.send(json.dumps(request))
                response = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.websocket_timeout
                )
                data = json.loads(response)
                
                if 'error' in data:
                    logger.error(f"Erro ao monitorar trade: {data['error']}")
                    return None
                
                contract = data.get('proposal_open_contract', {})
                
                if contract.get('status') == 'sold':
                    self.active_trade = False
                    profit = float(contract.get('profit', 0))
                    
                    trade_record = TradeRecord(
                        contract_id=contract_id,
                        time=datetime.fromtimestamp(contract.get('purchase_time', 0), UTC),
                        symbol=contract.get('symbol', ''),
                        direction=contract.get('contract_type', ''),
                        amount=float(contract.get('buy_price', 0)),
                        profit=profit,
                        entry_price=float(contract.get('entry_spot', 0)),
                        exit_price=float(contract.get('exit_spot', 0)),
                        status='completed'
                    )
                    
                    # Atualizar estatísticas
                    self.trades_history.append(trade_record)
                    self.total_profit += profit
                    total_trades = len(self.trades_history)
                    winning_trades = sum(1 for t in self.trades_history if t.profit > 0)
                    self.win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    
                    logger.info(f"Trade finalizado - Lucro: {profit:.2f}")
                    return trade_record
                
                await asyncio.sleep(1)
                
            return None
            
        except Exception as e:
            logger.error(f"Erro ao monitorar trade: {str(e)}")
            self.active_trade = False
            return None

    async def run(self):
        """Loop principal do bot"""
        try:
            # Carregar estado anterior
            await self.load_state()
            
            # Estabelecer conexão
            if not await self.connect():
                logger.error("Falha ao conectar. Encerrando...")
                return
            
            logger.info("Bot iniciado com sucesso!")
            
            while True:
                try:
                    # Verificar horário de operação
                    current_time = datetime.now(UTC).time()
                    if not (TRADING_START_TIME <= current_time <= TRADING_END_TIME):
                        await asyncio.sleep(60)
                        continue
                    
                    # Obter dados do mercado
                    df = await self.get_candles(SYMBOL)
                    if df is None:
                        continue
                    
                    # Calcular sinais
                    df = self.calculate_signals(df)
                    if df is None:
                        continue
                    
                    # Verificar sinais
                    signal = self.check_signals(df)
                    if signal:
                        # Executar trade
                        contract_id = await self.place_trade(SYMBOL, signal, TRADE_AMOUNT)
                        if contract_id:
                            # Monitorar resultado
                            trade_record = await self.monitor_trade(contract_id)
                            if trade_record:
                                await self.db_manager.save_trade(trade_record)
                                # Verificar circuit breaker
                                if self.circuit_breaker.check(trade_record.profit):
                                    logger.warning("Circuit breaker ativado - Pausando operações")
                    
                    # Salvar estado periodicamente
                    await self.save_state()
                    
                    # Aguardar próximo ciclo
                    await asyncio.sleep(CHECK_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Erro no loop principal: {str(e)}")
                    await asyncio.sleep(ERROR_RETRY_DELAY)
                
        except Exception as e:
            logger.error(f"Erro fatal: {str(e)}")
        finally:
            await self.cleanup()

if __name__ == "__main__":
    bot = BOB5KingsBot()
    asyncio.run(bot.run())
