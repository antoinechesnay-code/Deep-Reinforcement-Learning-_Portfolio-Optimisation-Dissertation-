# DDPG Trading Algorithm with Differential Sharpe Ratio and S&P 500 Benchmark
import numpy as np
import pandas as pd
import os
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        return self.net(state) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDPG:
    def __init__(self, state_dim, action_dim, max_action=1.0, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.99, tau=0.001, noise_std=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-4)
        
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.95)
        
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.max_action = max_action
        
        self.replay_buffer = ReplayBuffer()
        self.best_validation_score = -float('inf')
        self.episodes_without_improvement = 0
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            
        return action
    
    def update(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (self.gamma * target_q * (1 - done))
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
    
    def step_schedulers(self):
        self.actor_scheduler.step()
        self.critic_scheduler.step()
    
    def update_validation_score(self, score):
        if score > self.best_validation_score:
            self.best_validation_score = score
            self.episodes_without_improvement = 0
            return True
        else:
            self.episodes_without_improvement += 1
            return False
    
    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

def load_sp500_data(start_date='2015-01-01', end_date='2020-01-01'):
    try:
        print("Downloading S&P 500 data from Yahoo Finance...")
        sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        
        if sp500.empty:
            print("Warning: No S&P 500 data downloaded")
            return None
            
        sp500 = sp500.dropna()
        
        if hasattr(sp500.index, 'tz') and sp500.index.tz is not None:
            sp500.index = sp500.index.tz_localize(None)
        
        sp500.index = sp500.index.normalize()
        sp500['Daily_Return'] = sp500['Close'].pct_change()
        sp500 = sp500.dropna()
        
        print(f"S&P 500 data loaded: {len(sp500)} rows")
        return sp500
    except Exception as e:
        print(f"Error downloading S&P 500 data: {e}")
        return None

def create_synthetic_sp500_benchmark(stock_data, period_start, period_end):
    print("Creating synthetic S&P 500 benchmark...")
    
    common_dates = None
    for ticker, data in stock_data.items():
        try:
            period_data = data.loc[period_start:period_end]
            if common_dates is None:
                common_dates = set(period_data.index)
            else:
                common_dates = common_dates.intersection(set(period_data.index))
        except KeyError:
            continue
    
    if not common_dates:
        return None
    
    common_dates = sorted(list(common_dates))
    synthetic_sp500 = pd.DataFrame(index=common_dates)
    daily_returns = [0]
    
    for i in range(1, len(common_dates)):
        date = common_dates[i]
        prev_date = common_dates[i-1]
        date_returns = []
        
        for ticker, data in stock_data.items():
            try:
                if date in data.index and prev_date in data.index:
                    curr_close = data.loc[date, 'Close']
                    prev_close = data.loc[prev_date, 'Close']
                    if prev_close > 0:
                        daily_ret = (curr_close - prev_close) / prev_close
                        date_returns.append(daily_ret)
            except (KeyError, IndexError):
                continue
        
        if date_returns:
            avg_return = np.mean(date_returns)
            daily_returns.append(avg_return)
        else:
            daily_returns.append(0)
    
    synthetic_sp500['Close'] = 100.0
    synthetic_sp500['Daily_Return'] = daily_returns
    
    for i in range(1, len(synthetic_sp500)):
        synthetic_sp500.iloc[i, synthetic_sp500.columns.get_loc('Close')] = \
            synthetic_sp500.iloc[i-1, synthetic_sp500.columns.get_loc('Close')] * (1 + daily_returns[i])
    
    print(f"Created synthetic S&P 500 with {len(synthetic_sp500)} data points")
    return synthetic_sp500

def preprocess_data(data):
    return data.dropna()

def split_data_by_periods(data, sp500_data=None):
    train_start = pd.Timestamp('2015-01-01')
    train_end = pd.Timestamp('2017-12-31')
    val_start = pd.Timestamp('2018-01-01')
    val_end = pd.Timestamp('2018-12-31')
    test_start = pd.Timestamp('2019-01-01')
    test_end = pd.Timestamp('2019-12-31')
    
    splits = {}
    
    for ticker, ticker_data in data.items():
        if not isinstance(ticker_data.index, pd.DatetimeIndex):
            ticker_data.index = pd.to_datetime(ticker_data.index)
        
        try:
            splits[ticker] = {
                'train': ticker_data.loc[train_start:train_end].copy(),
                'validation': ticker_data.loc[val_start:val_end].copy(),
                'test': ticker_data.loc[test_start:test_end].copy()
            }
            
            for period, period_data in splits[ticker].items():
                if len(period_data) == 0:
                    print(f"Warning: No data found for {ticker} in {period} period")
                    
        except Exception as e:
            print(f"Error splitting data for {ticker}: {e}")
            total_rows = len(ticker_data)
            train_end_idx = int(total_rows * 0.6)
            val_end_idx = int(total_rows * 0.8)
            
            splits[ticker] = {
                'train': ticker_data.iloc[:train_end_idx].copy(),
                'validation': ticker_data.iloc[train_end_idx:val_end_idx].copy(),
                'test': ticker_data.iloc[val_end_idx:].copy()
            }
    
    if sp500_data is not None:
        try:
            if not isinstance(sp500_data.index, pd.DatetimeIndex):
                sp500_data.index = pd.to_datetime(sp500_data.index)
            
            splits['SP500'] = {
                'train': sp500_data.loc[train_start:train_end].copy(),
                'validation': sp500_data.loc[val_start:val_end].copy(),
                'test': sp500_data.loc[test_start:test_end].copy()
            }
            
            for period in ['train', 'validation', 'test']:
                if len(splits['SP500'][period]) < 10:
                    print(f"Creating synthetic benchmark for {period} period.")
                    if period == 'train':
                        synthetic_data = create_synthetic_sp500_benchmark(data, train_start, train_end)
                    elif period == 'validation':
                        synthetic_data = create_synthetic_sp500_benchmark(data, val_start, val_end)
                    else:
                        synthetic_data = create_synthetic_sp500_benchmark(data, test_start, test_end)
                    
                    splits['SP500'][period] = synthetic_data
        
        except Exception as e:
            print(f"Error with S&P 500 data: {e}")
            print("Creating synthetic S&P 500 benchmark for all periods...")
            
            splits['SP500'] = {
                'train': create_synthetic_sp500_benchmark(data, train_start, train_end),
                'validation': create_synthetic_sp500_benchmark(data, val_start, val_end),
                'test': create_synthetic_sp500_benchmark(data, test_start, test_end)
            }
    else:
        print("No S&P 500 data available. Creating synthetic benchmark...")
        splits['SP500'] = {
            'train': create_synthetic_sp500_benchmark(data, train_start, train_end),
            'validation': create_synthetic_sp500_benchmark(data, val_start, val_end),  
            'test': create_synthetic_sp500_benchmark(data, test_start, test_end)
        }
    
    return splits

class DDPGTradingEnv(gym.Env):
    def __init__(self, data, sp500_data=None, initial_balance=10000.0, transaction_fee_percent=0.001, 
                 window_size=50, benchmark_ticker='SPY', risk_free_rate=0.02):
        super(DDPGTradingEnv, self).__init__()
        
        self.data = data
        self.sp500_data = sp500_data
        self.tickers = list(data.keys())
        self.num_stocks = len(self.tickers)
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = window_size
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate  # 2% annual risk-free rate
        
        self.simple_align_dates()
        
        num_features_per_stock = 20
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_stocks + 1 + (self.num_stocks * num_features_per_stock),), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.num_stocks,), 
            dtype=np.float32
        )
        
        self.reset()
        
    def simple_align_dates(self):
        print("Simple date alignment...")
        
        common_dates = None
        for ticker in self.tickers:
            stock_dates = set(self.data[ticker].index)
            if common_dates is None:
                common_dates = stock_dates
            else:
                common_dates = common_dates.intersection(stock_dates)
        
        print(f"Stock intersection: {len(common_dates)} dates")
        
        if self.sp500_data is not None:
            sp500_dates = set(self.sp500_data.index)
            sp500_intersection = common_dates.intersection(sp500_dates)
            print(f"S&P 500 intersection: {len(sp500_intersection)} dates")
            
            if len(sp500_intersection) >= 100:
                common_dates = sp500_intersection
                print("Using S&P 500 benchmark")
            else:
                print("Disabling S&P 500 benchmark - insufficient overlap")
                self.sp500_data = None
        
        common_dates = sorted(list(common_dates))
        if len(common_dates) < self.window_size + 50:
            raise ValueError(f"Not enough data: {len(common_dates)} dates")
        
        for ticker in self.tickers:
            self.data[ticker] = self.data[ticker].loc[self.data[ticker].index.isin(common_dates)]
        
        if self.sp500_data is not None:
            self.sp500_data = self.sp500_data.loc[self.sp500_data.index.isin(common_dates)]
        
        self.dates = common_dates
        print(f"Final: {len(self.dates)} trading days")
        
    def _get_observation(self):
        total_value = self._calculate_portfolio_value()
        positions_balance = np.array(list(self.positions.values()) + [self.balance / total_value])
        
        market_features = []
        for ticker in self.tickers:
            stock_data = self.data[ticker].iloc[self.current_step]
            
            features = [
                stock_data['Close'] / stock_data['Open'] - 1,
                stock_data['High'] / stock_data['Low'] - 1,
                stock_data['Close'] / stock_data['MA_5'] - 1,
                stock_data['Close'] / stock_data['MA_10'] - 1,
                stock_data['Close'] / stock_data['MA_20'] - 1,
                stock_data['Close'] / stock_data['MA_50'] - 1,
                stock_data['RSI'] / 100,
                stock_data['MACD'] / stock_data['Close'],
                stock_data['MACD_Signal'] / stock_data['Close'],
                stock_data['MACD_Histogram'] / stock_data['Close'],
                stock_data['BB_Position'],
                stock_data['Price_to_BB_Middle'],
                stock_data['BB_Width'] / stock_data['Close'],
                stock_data['Volume_Ratio'],
                stock_data['Volatility_10d'],
                stock_data['Volatility_20d'],
                stock_data['Price_vs_Support'],
                stock_data['Price_vs_Resistance'],
                float(stock_data['Trend_5d']),
                float(stock_data['Trend_20d'])
            ]
            
            market_features.extend(features)
        
        observation = np.concatenate([positions_balance, np.array(market_features)])
        return observation.astype(np.float32)
    
    def _calculate_portfolio_value(self):
        portfolio_value = self.balance
        
        for ticker in self.tickers:
            current_price = self.data[ticker]['Close'].iloc[self.current_step]
            portfolio_value += self.positions[ticker] * current_price
        
        return portfolio_value
    
    def _execute_trades(self, actions):
        total_cost = 0
        trades_executed = []
        
        for i, ticker in enumerate(self.tickers):
            current_price = self.data[ticker]['Close'].iloc[self.current_step]
            current_position = self.positions[ticker]
            
            portfolio_value = self._calculate_portfolio_value()
            target_value = (actions[i] + 1) / 2 * portfolio_value
            target_shares = target_value / current_price
            
            shares_to_trade = target_shares - current_position
            
            if abs(shares_to_trade) > 0.01:
                trade_value = abs(shares_to_trade) * current_price
                transaction_cost = trade_value * self.transaction_fee_percent
                
                if shares_to_trade > 0:
                    total_needed = trade_value + transaction_cost
                    if total_needed <= self.balance:
                        self.positions[ticker] = target_shares
                        self.balance -= total_needed
                        total_cost += transaction_cost
                        trades_executed.append({
                            'ticker': ticker,
                            'action': 'buy',
                            'shares': shares_to_trade,
                            'price': current_price,
                            'cost': transaction_cost
                        })
                
                else:
                    if current_position >= abs(shares_to_trade):
                        self.positions[ticker] = target_shares
                        proceeds = trade_value - transaction_cost
                        self.balance += proceeds
                        total_cost += transaction_cost
                        trades_executed.append({
                            'ticker': ticker,
                            'action': 'sell',
                            'shares': shares_to_trade,
                            'price': current_price,
                            'cost': transaction_cost
                        })
        
        return total_cost, trades_executed
    
    def _calculate_reward(self):
        current_value = self._calculate_portfolio_value()
        self.portfolio_history.append(current_value)
        
        if len(self.portfolio_history) < 2:
            return 0
        
        portfolio_return = (current_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
        
        # Get benchmark return
        benchmark_return = 0
        if (self.sp500_data is not None and 
            self.current_step < len(self.sp500_data) and
            'Daily_Return' in self.sp500_data.columns):
            try:
                benchmark_return = self.sp500_data['Daily_Return'].iloc[self.current_step]
                if pd.isna(benchmark_return):
                    benchmark_return = 0
            except (IndexError, KeyError):
                benchmark_return = 0
        
        # Calculate differential return (portfolio vs benchmark)
        differential_return = portfolio_return - benchmark_return
        
        # Calculate differential Sharpe ratio with 2% risk-free rate
        if (self.sp500_data is not None and len(self.portfolio_history) >= 20):
            recent_portfolio_returns = []
            recent_benchmark_returns = []
            
            # Collect recent returns for differential Sharpe calculation
            for i in range(-20, 0):
                if i + len(self.portfolio_history) > 0:
                    port_ret = (self.portfolio_history[i] - self.portfolio_history[i-1]) / self.portfolio_history[i-1]
                    recent_portfolio_returns.append(port_ret)
                    
                    step_idx = self.current_step + i
                    if (step_idx >= 0 and 
                        step_idx < len(self.sp500_data) and
                        'Daily_Return' in self.sp500_data.columns):
                        try:
                            bench_ret = self.sp500_data['Daily_Return'].iloc[step_idx]
                            if pd.isna(bench_ret):
                                bench_ret = 0
                            recent_benchmark_returns.append(bench_ret)
                        except (IndexError, KeyError):
                            recent_benchmark_returns.append(0)
                    else:
                        recent_benchmark_returns.append(0)
            
            if len(recent_portfolio_returns) > 1 and len(recent_benchmark_returns) > 1:
                # Calculate excess returns over risk-free rate
                daily_risk_free_rate = self.risk_free_rate / 252  # Convert annual to daily
                
                portfolio_excess_returns = np.array(recent_portfolio_returns) - daily_risk_free_rate
                benchmark_excess_returns = np.array(recent_benchmark_returns) - daily_risk_free_rate
                
                # Calculate differential excess returns
                diff_excess_returns = portfolio_excess_returns - benchmark_excess_returns
                
                mean_diff_excess = np.mean(diff_excess_returns)
                std_diff_excess = np.std(diff_excess_returns)
                
                if std_diff_excess > 1e-8:
                    diff_sharpe = mean_diff_excess / std_diff_excess
                    reward = differential_return + 0.1 * diff_sharpe
                else:
                    reward = differential_return
            else:
                reward = differential_return
        else:
            # Fallback when no S&P 500 data or insufficient history
            if self.sp500_data is None and len(self.portfolio_history) >= 20:
                recent_returns = []
                for i in range(-20, 0):
                    if i + len(self.portfolio_history) > 0:
                        ret = (self.portfolio_history[i] - self.portfolio_history[i-1]) / self.portfolio_history[i-1]
                        recent_returns.append(ret)
                
                if len(recent_returns) > 1:
                    # Standard Sharpe ratio with 2% risk-free rate
                    daily_risk_free_rate = self.risk_free_rate / 252
                    excess_returns = np.array(recent_returns) - daily_risk_free_rate
                    mean_excess = np.mean(excess_returns)
                    std_excess = np.std(excess_returns)
                    
                    if std_excess > 1e-8:
                        sharpe = mean_excess / std_excess
                        reward = portfolio_return + 0.1 * sharpe
                    else:
                        reward = portfolio_return
                else:
                    reward = portfolio_return
            else:
                reward = differential_return
        
        return reward
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_history = [self.initial_balance]
        self.trade_history = []
        self.transaction_costs = []
        
        self.current_step = self.window_size
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        transaction_cost, trades = self._execute_trades(action)
        self.transaction_costs.append(transaction_cost)
        self.trade_history.extend(trades)
        
        self.current_step += 1
        reward = self._calculate_reward()
        done = self.current_step >= len(self.dates) - 1
        observation = self._get_observation()
        
        info = {
            'portfolio_value': self._calculate_portfolio_value(),
            'balance': self.balance,
            'positions': self.positions.copy(),
            'transaction_cost': transaction_cost,
            'trades': trades
        }
        
        return observation, reward, done, False, info

class PerformanceAnalyzer:
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_all_metrics(self, portfolio_values, benchmark_returns=None, dates=None):
        portfolio_returns = np.array([
            (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            for i in range(1, len(portfolio_values))
        ])
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        calmar_ratio = self._calculate_calmar_ratio(portfolio_values, annual_return)
        
        diff_sharpe_ratio = self._calculate_differential_sharpe_ratio(portfolio_returns, benchmark_returns)
        alpha, beta = self._calculate_alpha_beta(portfolio_returns, benchmark_returns)
        max_drawdown, drawdown_duration = self._calculate_drawdown_metrics(portfolio_values)
        winning_percentage = self._calculate_winning_percentage(portfolio_returns)
        roi = total_return * 100
        information_ratio = self._calculate_information_ratio(portfolio_returns, benchmark_returns)
        
        return {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'differential_sharpe_ratio': diff_sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'alpha': alpha * 100 if alpha is not None else None,
            'beta': beta,
            'max_drawdown': max_drawdown * 100,
            'drawdown_duration': drawdown_duration,
            'winning_percentage': winning_percentage,
            'roi': roi,
            'portfolio_returns': portfolio_returns,
            'portfolio_values': portfolio_values
        }
    
    def _calculate_sharpe_ratio(self, returns):
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    def _calculate_differential_sharpe_ratio(self, portfolio_returns, benchmark_returns):
        if benchmark_returns is None or len(benchmark_returns) != len(portfolio_returns):
            return None
        
        # Calculate excess returns over risk-free rate for both portfolio and benchmark
        daily_risk_free_rate = self.risk_free_rate / 252
        portfolio_excess_returns = portfolio_returns - daily_risk_free_rate
        benchmark_excess_returns = benchmark_returns - daily_risk_free_rate
        
        # Calculate differential excess returns
        differential_excess_returns = portfolio_excess_returns - benchmark_excess_returns
        mean_diff = np.mean(differential_excess_returns)
        std_diff = np.std(differential_excess_returns)
        
        if std_diff > 1e-8:
            return mean_diff / std_diff * np.sqrt(252)
        else:
            return 0
    
    def _calculate_information_ratio(self, portfolio_returns, benchmark_returns):
        if benchmark_returns is None or len(benchmark_returns) != len(portfolio_returns):
            return None
        
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error > 1e-8:
            return np.mean(excess_returns) / tracking_error * np.sqrt(252)
        else:
            return 0
    
    def _calculate_sortino_ratio(self, returns):
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, portfolio_values, annual_return):
        max_dd = self._calculate_max_drawdown(portfolio_values)
        return annual_return / max_dd if max_dd > 0 else 0
    
    def _calculate_alpha_beta(self, portfolio_returns, benchmark_returns):
        if benchmark_returns is None or len(benchmark_returns) != len(portfolio_returns):
            return None, None
        
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        portfolio_mean = np.mean(portfolio_returns) * 252
        benchmark_mean = np.mean(benchmark_returns) * 252
        alpha = portfolio_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
        
        return alpha, beta
    
    def _calculate_max_drawdown(self, portfolio_values):
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_drawdown_metrics(self, portfolio_values):
        peak = portfolio_values[0]
        max_drawdown = 0
        current_drawdown_duration = 0
        max_drawdown_duration = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
                current_drawdown_duration = 0
            else:
                current_drawdown_duration += 1
                
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
        
        return max_drawdown, max_drawdown_duration
    
    def _calculate_winning_percentage(self, returns):
        positive_returns = np.sum(returns > 0)
        return positive_returns / len(returns) * 100 if len(returns) > 0 else 0

def train_ddpg(env, episodes=1000, save_path='./ddpg_model', validation_env=None):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = DDPG(state_dim, action_dim)
    episode_rewards = []
    patience = 200
    use_early_stopping = validation_env is not None
    
    os.makedirs(save_path, exist_ok=True)
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            
            if len(agent.replay_buffer) > 1000:
                agent.update()
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            agent.step_schedulers()
        
        if use_early_stopping and episode % 100 == 0 and episode > 0:
            val_score = evaluate_agent(agent, validation_env)
            improved = agent.update_validation_score(val_score)
            if improved:
                agent.save(f"{save_path}/ddpg_model_best.pth")
            
            if agent.episodes_without_improvement >= patience and episode > 400:
                print(f"Early stopping at episode {episode}")
                break
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.4f}")
    
    agent.save(f"{save_path}/ddpg_model_final.pth")
    
    if use_early_stopping and os.path.exists(f"{save_path}/ddpg_model_best.pth"):
        agent.load(f"{save_path}/ddpg_model_best.pth")
    
    return agent, episode_rewards

def evaluate_agent(agent, env, episodes=5):
    total_returns = []
    
    for _ in range(episodes):
        try:
            state, _ = env.reset()
            done = False
            initial_value = env._calculate_portfolio_value()
            
            while not done:
                action = agent.select_action(state, add_noise=False)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            final_value = info.get('portfolio_value', initial_value)
            episode_return = (final_value - initial_value) / initial_value
            total_returns.append(episode_return)
        except:
            total_returns.append(0.0)
    
    if len(total_returns) > 1:
        mean_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        return mean_return / (std_return + 1e-8)
    else:
        return total_returns[0] if total_returns else 0.0

def backtest_with_metrics(agent, env, period_name="", save_path="./results"):
    os.makedirs(save_path, exist_ok=True)
    
    state, _ = env.reset()
    done = False
    portfolio_values = []
    daily_returns = []
    benchmark_returns = []
    trade_history = []
    transaction_costs = []
    dates = []
    
    step = 0
    while not done:
        action = agent.select_action(state, add_noise=False)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info['portfolio_value'])
        trade_history.extend(info['trades'])
        transaction_costs.append(info['transaction_cost'])
        
        if step < len(env.dates):
            dates.append(env.dates[step])
        
        if len(portfolio_values) >= 2:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)
            
            if env.sp500_data is not None and step < len(env.sp500_data):
                sp_return = env.sp500_data['Daily_Return'].iloc[step]
                benchmark_returns.append(sp_return)
            else:
                benchmark_returns.append(0)
        
        step += 1
    
    if len(benchmark_returns) != len(daily_returns):
        benchmark_returns = benchmark_returns[:len(daily_returns)]
    
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    metrics = analyzer.calculate_all_metrics(
        portfolio_values, 
        benchmark_returns=np.array(benchmark_returns) if benchmark_returns else None
    )
    
    total_transaction_costs = sum(transaction_costs)
    num_trades = len(trade_history)
    
    if benchmark_returns:
        benchmark_total_return = (np.prod(1 + np.array(benchmark_returns)) - 1) * 100
    else:
        benchmark_total_return = 0
    
    # Calculate excess annual return
    excess_annual_return = metrics['annual_return'] - (env.risk_free_rate * 100)
    daily_risk_free_rate = env.risk_free_rate / 252
    
    print(f"\n{period_name} Results (with Sentiment Integration & 2.0% Risk-Free Rate):")
    print("=" * 80)
    print("=" * 15)
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Annual Return: {metrics['annual_return']:.2f}%")
    print(f"Risk-Free Rate Used: {env.risk_free_rate:.1%} annually ({daily_risk_free_rate:.4%} daily)")
    print(f"Excess Annual Return: {excess_annual_return:.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio (vs {env.risk_free_rate:.1%} RFR): {metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio (vs {env.risk_free_rate:.1%} RFR): {metrics['sortino_ratio']:.4f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Winning Percentage: {metrics['winning_percentage']:.2f}%")
    print(f"ROI: {metrics['total_return']:.2f}%")
    print(f"Total Transaction Costs: ${total_transaction_costs:.2f}")
    print(f"Number of Trades: {num_trades}")
    
    if metrics['differential_sharpe_ratio'] is not None:
        print(f"S&P 500 Total Return: {benchmark_total_return:.2f}%")
        print(f"Differential Sharpe Ratio: {metrics['differential_sharpe_ratio']:.4f}")
        if metrics['information_ratio'] is not None:
            print(f"Information Ratio: {metrics['information_ratio']:.4f}")
        if metrics['alpha'] is not None:
            print(f"Alpha: {metrics['alpha']:.2f}%")
        if metrics['beta'] is not None:
            print(f"Beta: {metrics['beta']:.4f}")
    
    results_df = pd.DataFrame({
        'Date': dates[:len(portfolio_values)],
        'Portfolio_Value': portfolio_values,
        'Daily_Return': [0] + daily_returns,
        'SP500_Return': [0] + benchmark_returns,
        'Transaction_Cost': transaction_costs
    })
    results_df.to_csv(f"{save_path}/{period_name.lower()}_results.csv", index=False)
    
    # Create a summary metrics dictionary
    summary_metrics = {
        'period': period_name,
        'total_return_pct': metrics['total_return'],
        'annual_return_pct': metrics['annual_return'],
        'risk_free_rate_annual': env.risk_free_rate * 100,
        'risk_free_rate_daily': daily_risk_free_rate * 100,
        'excess_annual_return_pct': excess_annual_return,
        'volatility_pct': metrics['volatility'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'sortino_ratio': metrics['sortino_ratio'],
        'calmar_ratio': metrics['calmar_ratio'],
        'max_drawdown_pct': metrics['max_drawdown'],
        'winning_percentage': metrics['winning_percentage'],
        'roi_pct': metrics['total_return'],
        'total_transaction_costs': total_transaction_costs,
        'number_of_trades': num_trades,
        'benchmark_total_return_pct': benchmark_total_return,
        'differential_sharpe_ratio': metrics['differential_sharpe_ratio'],
        'information_ratio': metrics['information_ratio'],
        'alpha_pct': metrics['alpha'],
        'beta': metrics['beta']
    }
    
    # Save summary metrics to CSV
    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv(f"{save_path}/{period_name.lower()}_summary_metrics.csv", index=False)
    
    return metrics, results_df

def load_all_data():
    data = {}
    
    available_files = [
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/AAPL_2015-01-01_to_2020-01-01_with_indicators_20250831_104931.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/AMT_2015-01-01_to_2020-01-01_with_indicators_20250831_104424.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/AMZN_2015-01-01_to_2020-01-01_with_indicators_20250831_104508.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/CAT_2015-01-01_to_2020-01-01_with_indicators_20250831_104535.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/JPM_2015-01-01_to_2020-01-01_with_indicators_20250831_104607.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/LIN_2015-01-01_to_2020-01-01_with_indicators_20250831_104627.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/NEE_2015-01-01_to_2020-01-01_with_indicators_20250831_104649.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/PG_2015-01-01_to_2020-01-01_with_indicators_20250831_104703.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/UNH_2015-01-01_to_2020-01-01_with_indicators_20250831_104745.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/XOM_2015-01-01_to_2020-01-01_with_indicators_20250831_104808.csv'
    ]
    
    for filepath in available_files:
        filename = os.path.basename(filepath)
        ticker = filename.split('_')[0]
        
        try:
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue
                
            stock_data = pd.read_csv(filepath)
            
            if 'Date' not in stock_data.columns:
                print(f"Warning: {ticker} missing Date column")
                continue
                
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce', utc=True)
            stock_data = stock_data.dropna(subset=['Date'])
            stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)
            stock_data['Date'] = stock_data['Date'].dt.normalize()
            stock_data.set_index('Date', inplace=True)
            
            required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
            if all(col in stock_data.columns for col in required_cols):
                data[ticker] = preprocess_data(stock_data)
                print(f"Successfully loaded {ticker}: {len(data[ticker])} rows")
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    sp500_data = load_sp500_data('2015-01-01', '2020-01-01')
    return data, sp500_data

def verify_data_integrity(data):
    """Verify that all required columns exist in the data"""
    required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    optional_columns = ['MA_5', 'MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 
                       'MACD_Histogram', 'BB_Position', 'Price_to_BB_Middle', 'BB_Width',
                       'Volume_Ratio', 'Volatility_10d', 'Volatility_20d', 'Price_vs_Support',
                       'Price_vs_Resistance', 'Trend_5d', 'Trend_20d']
    
    for ticker, df in data.items():
        print(f"Checking {ticker}...")
        
        # Check required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            print(f"  ERROR: Missing required columns: {missing_required}")
            return False
        
        # Report missing optional columns
        missing_optional = [col for col in optional_columns if col not in df.columns]
        if missing_optional:
            print(f"  WARNING: Missing optional columns (will use defaults): {missing_optional}")
        
        # Check for sufficient data
        if len(df) < 100:
            print(f"  ERROR: Insufficient data for {ticker}: {len(df)} rows")
            return False
        
        print(f"  OK: {len(df)} rows, {len(df.columns)} columns")
    
    return True

def run_complete_ddpg_experiment():
    print("Loading CSV data and S&P 500 benchmark...")
    try:
        all_data, sp500_data = load_all_data()
        
        if not all_data:
            print("Error: No data loaded successfully")
            return None, None, None, None
        
        print(f"Loaded data for {len(all_data)} stocks: {list(all_data.keys())}")
        
        # Verify data integrity
        if not verify_data_integrity(all_data):
            print("Data integrity check failed. Please check your data files.")
            return None, None, None, None
        
        data_splits = split_data_by_periods(all_data, sp500_data)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = f"./ddpg_differential_sharpe_experiment_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        
        print("\nTraining DDPG agent...")
        
        train_data = {ticker: splits['train'] for ticker, splits in data_splits.items() if ticker != 'SP500'}
        val_data = {ticker: splits['validation'] for ticker, splits in data_splits.items() if ticker != 'SP500'}
        test_data = {ticker: splits['test'] for ticker, splits in data_splits.items() if ticker != 'SP500'}
        
        train_sp500 = data_splits['SP500']['train']
        val_sp500 = data_splits['SP500']['validation']
        test_sp500 = data_splits['SP500']['test']
        
        # Initialize environments with 2% risk-free rate
        print("Creating training environment...")
        train_env = DDPGTradingEnv(train_data, sp500_data=train_sp500, 
                                 initial_balance=10000.0, transaction_fee_percent=0.001, risk_free_rate=0.02)
        
        print("Creating validation environment...")
        val_env = DDPGTradingEnv(val_data, sp500_data=val_sp500,
                               initial_balance=10000.0, transaction_fee_percent=0.001, risk_free_rate=0.02)
        
        print("Creating test environment...")
        test_env = DDPGTradingEnv(test_data, sp500_data=test_sp500,
                                initial_balance=10000.0, transaction_fee_percent=0.001, risk_free_rate=0.02)
        
        model_save_path = f"{experiment_dir}/model"
        agent, training_rewards = train_ddpg(
            train_env, 
            episodes=1000,
            save_path=model_save_path,
            validation_env=val_env
        )
        
        train_metrics, _ = backtest_with_metrics(agent, train_env, "Training", f"{experiment_dir}/training_results")
        val_metrics, _ = backtest_with_metrics(agent, val_env, "Validation", f"{experiment_dir}/validation_results")
        test_metrics, _ = backtest_with_metrics(agent, test_env, "Testing", f"{experiment_dir}/testing_results")
        
        training_df = pd.DataFrame({
            'Episode': range(len(training_rewards)),
            'Episode_Reward': training_rewards
        })
        training_df.to_csv(f"{experiment_dir}/training_progress.csv", index=False)
        
        plt.figure(figsize=(12, 6))
        plt.plot(training_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('DDPG Differential Sharpe Training Progress (2% Risk-Free Rate)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{experiment_dir}/training_progress.png")
        plt.close()
        
        print(f"\nExperiment completed! Results saved to: {experiment_dir}")
        return train_metrics, val_metrics, test_metrics, experiment_dir
        
    except Exception as e:
        print(f"Critical error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def main():
    print("="*80)
    print("DDPG TRADING ALGORITHM - DIFFERENTIAL SHARPE RATIO VERSION (2% Risk-Free Rate)")
    print("="*80)
    print("Dataset: 10 sector-leading stocks (2015-2020)")
    print("Algorithm: Deep Deterministic Policy Gradient (DDPG)")
    print("Benchmark: S&P 500 Index (^GSPC)")
    print("Key Innovation: Differential Sharpe Ratio Optimization with 2% Risk-Free Rate")
    print("="*80)
    
    try:
        train_metrics, val_metrics, test_metrics, experiment_dir = run_complete_ddpg_experiment()
        
        if experiment_dir:
            print(f"\nDDPG DIFFERENTIAL SHARPE RATIO EXPERIMENT COMPLETED!")
            print(f"Results directory: {experiment_dir}")
            print("Key Features:")
            print("✓ Real-time S&P 500 benchmark integration")
            print("✓ Differential Sharpe ratio reward function with 2% risk-free rate") 
            print("✓ Alpha/Beta calculation vs S&P 500")
            print("✓ Information ratio analysis")
            
        return experiment_dir
        
    except Exception as e:
        print(f"Error during execution: {e}")
        return None

if __name__ == "__main__":
    print("Starting DDPG Differential Sharpe Ratio Trading Experiment (2% Risk-Free Rate)...")
    result_dir = main()
    
    if result_dir:
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {result_dir}")
    else:
        print("\nExperiment failed. Check error messages above.")