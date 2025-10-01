# PPO Trading Algorithm with Sentiment Features and Differential Sharpe Ratio Reward Function

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
from torch.distributions import Normal
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

# Define sector leaders
sector_leaders = [
    'AAPL',  # Tech
    'UNH',   # Healthcare
    'JPM',   # Finance
    'XOM',   # Energy
    'AMZN',  # Consumer
    'CAT',   # Industrial
    'AMT',   # Real Estate
    'NEE',   # Utilities
    'LIN',   # Materials
    'PG'     # Consumer Staples
]

# PPO Network Components
class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with regularization to prevent overfitting"""
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction layers with stronger regularization
        # Increased network size to handle sentiment features
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 512),  # Increased from 256 to handle more features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Actor head - sentiment-aware policy
        self.actor_mean = nn.Linear(128, action_dim)
        # More conservative initial exploration
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * 0.2)
        
        # Critic head - sentiment-aware value function
        self.critic = nn.Linear(128, 1)
        
        self.max_action = max_action
        
        # Initialize weights with smaller variance
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent extreme values"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain
                nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        shared_features = self.shared_layers(state)
        
        # Actor output with constrained std
        action_mean = self.actor_mean(shared_features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std.clamp(-2, 1))  # More conservative range
        
        # Critic output
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def act(self, state):
        """Sample action from policy with controlled exploration"""
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(-1)
        
        return action, action_log_prob, value
    
    def evaluate(self, state, action):
        """Evaluate action log probabilities and state values"""
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        
        action_log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return action_log_prob, value.squeeze(), entropy

class PPOBuffer:
    """Experience buffer for PPO with robust shape handling"""
    def __init__(self, capacity=2048):
        self.capacity = capacity
        self.clear()
        
    def store(self, state, action, reward, value, log_prob, done):
        """Store experience with type conversion and validation"""
        # Convert to numpy arrays and ensure consistent shapes
        if isinstance(state, (list, tuple)):
            state = np.array(state, dtype=np.float32)
        elif isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy().astype(np.float32)
        else:
            state = np.array(state, dtype=np.float32)
        
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.float32)
        elif isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().astype(np.float32)
        else:
            action = np.array(action, dtype=np.float32)
        
        # Ensure scalar values are properly formatted
        reward = float(reward)
        
        if isinstance(value, (list, tuple, np.ndarray)):
            if hasattr(value, '__len__') and len(value) > 0:
                value = float(value[0])  # Take first element if array
            else:
                value = float(value)
        elif isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        else:
            value = float(value)
        
        if isinstance(log_prob, (list, tuple, np.ndarray)):
            if hasattr(log_prob, '__len__') and len(log_prob) > 0:
                log_prob = float(log_prob[0])  # Take first element if array
            else:
                log_prob = float(log_prob)
        elif isinstance(log_prob, torch.Tensor):
            log_prob = float(log_prob.detach().cpu().item())
        else:
            log_prob = float(log_prob)
        
        done = bool(done)
        
        # Store with consistent shapes
        self.states.append(state.flatten())  # Flatten to ensure 1D
        self.actions.append(action.flatten())  # Flatten to ensure 1D
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def finish_path(self, last_value=0):
        """Compute advantages and returns using GAE with robust handling"""
        if len(self.rewards) == 0:
            return
            
        gamma = 0.99
        lam = 0.95
        
        # Convert last_value to float if needed
        if isinstance(last_value, (torch.Tensor, np.ndarray)):
            if hasattr(last_value, 'item'):
                last_value = float(last_value.item())
            else:
                last_value = float(last_value)
        else:
            last_value = float(last_value)
        
        # Ensure we have consistent arrays
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=bool)
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - float(dones[t])
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - float(dones[t])
                nextvalues = values[t + 1]
            
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        
        # Compute returns
        returns = advantages + values[:-1]
        
        # Store as lists of floats
        self.advantages = [float(adv) for adv in advantages]
        self.returns = [float(ret) for ret in returns]
    
    def get(self):
        """Get all stored experiences with shape validation"""
        if len(self.states) == 0:
            # Return empty tensors with correct shapes
            return {
                'states': torch.empty(0, 0),
                'actions': torch.empty(0, 0),
                'log_probs': torch.empty(0),
                'advantages': torch.empty(0),
                'returns': torch.empty(0)
            }
        
        # Validate that all sequences have the same length
        lengths = [len(self.states), len(self.actions), len(self.rewards), 
                  len(self.values), len(self.log_probs), len(self.dones)]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError(f"Inconsistent buffer lengths: {lengths}")
        
        if len(self.advantages) != len(self.states) or len(self.returns) != len(self.states):
            raise ValueError("Advantages/returns not computed. Call finish_path() first.")
        
        # Convert to numpy arrays with shape validation
        try:
            states_array = np.array(self.states, dtype=np.float32)
            actions_array = np.array(self.actions, dtype=np.float32)
            log_probs_array = np.array(self.log_probs, dtype=np.float32)
            advantages_array = np.array(self.advantages, dtype=np.float32)
            returns_array = np.array(self.returns, dtype=np.float32)
        except ValueError as e:
            print(f"Error creating arrays: {e}")
            print(f"States shapes: {[s.shape if hasattr(s, 'shape') else len(s) for s in self.states[:3]]}")
            print(f"Actions shapes: {[a.shape if hasattr(a, 'shape') else len(a) for a in self.actions[:3]]}")
            raise
        
        # Normalize advantages
        if len(advantages_array) > 1:
            advantages_array = (advantages_array - advantages_array.mean()) / (advantages_array.std() + 1e-8)
        
        # Convert to tensors
        data = {
            'states': torch.FloatTensor(states_array),
            'actions': torch.FloatTensor(actions_array),
            'log_probs': torch.FloatTensor(log_probs_array),
            'advantages': torch.FloatTensor(advantages_array),
            'returns': torch.FloatTensor(returns_array)
        }
        
        return data
    
    def clear(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def __len__(self):
        return len(self.states)

class PPO:
    """Proximal Policy Optimization Algorithm with overfitting prevention"""
    def __init__(self, state_dim, action_dim, max_action=1.0, lr=5e-4, 
                 clip_ratio=0.15, value_coef=0.5, entropy_coef=0.01,
                 update_epochs=3, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network with weight decay for regularization
        self.actor_critic = ActorCritic(state_dim, action_dim, max_action).to(self.device)
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), 
            lr=lr, 
            eps=1e-5,
            weight_decay=1e-4  # L2 regularization
        )
        
        # More conservative PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.max_action = max_action
        
        # Learning rate scheduler for adaptive learning
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        # Experience buffer
        self.buffer = PPOBuffer()
        
    def select_action(self, state):
        """Select action using actor network"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.act(state)
            
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy(), value.cpu().numpy()
    
    def store_experience(self, state, action, reward, value, log_prob, done):
        """Store experience in buffer"""
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def update(self, last_value=0):
        """Update policy using PPO with robust error handling"""
        try:
            # Finish current trajectory
            self.buffer.finish_path(last_value)
            
            # Check if we have enough data
            if len(self.buffer) < self.batch_size:
                print(f"Warning: Not enough data for update. Buffer size: {len(self.buffer)}, required: {self.batch_size}")
                self.buffer.clear()
                return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
            
            # Get data from buffer
            data = self.buffer.get()
            
            # Check data validity
            if data['states'].shape[0] == 0:
                print("Warning: Empty data batch")
                self.buffer.clear()
                return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
            
            # Update for multiple epochs
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            update_count = 0
            
            for epoch in range(self.update_epochs):
                try:
                    # Create random mini-batches
                    indices = torch.randperm(len(data['states']))
                    
                    for start in range(0, len(data['states']), self.batch_size):
                        end = start + self.batch_size
                        batch_indices = indices[start:end]
                        
                        if len(batch_indices) < self.batch_size:
                            continue
                        
                        # Get batch data
                        batch_states = data['states'][batch_indices].to(self.device)
                        batch_actions = data['actions'][batch_indices].to(self.device)
                        batch_old_log_probs = data['log_probs'][batch_indices].to(self.device)
                        batch_advantages = data['advantages'][batch_indices].to(self.device)
                        batch_returns = data['returns'][batch_indices].to(self.device)
                        
                        # Validate batch data
                        if torch.isnan(batch_states).any() or torch.isnan(batch_actions).any():
                            print("Warning: NaN detected in batch data, skipping batch")
                            continue
                        
                        # Evaluate current policy
                        try:
                            new_log_probs, values, entropy = self.actor_critic.evaluate(batch_states, batch_actions)
                        except Exception as e:
                            print(f"Error in policy evaluation: {e}")
                            continue
                        
                        # Compute policy loss
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        
                        # Clamp ratio to prevent extreme values
                        ratio = torch.clamp(ratio, 0.1, 10.0)
                        
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Compute value loss
                        value_loss = F.mse_loss(values, batch_returns)
                        
                        # Compute entropy loss
                        entropy_loss = -entropy.mean()
                        
                        # Total loss
                        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                        
                        # Check for NaN in loss
                        if torch.isnan(loss):
                            print("Warning: NaN loss detected, skipping update")
                            continue
                        
                        # Update network
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                        self.optimizer.step()
                        
                        total_policy_loss += policy_loss.item()
                        total_value_loss += value_loss.item()
                        total_entropy_loss += entropy_loss.item()
                        update_count += 1
                        
                except Exception as e:
                    print(f"Error in epoch {epoch}: {e}")
                    continue
            
            # Clear buffer
            self.buffer.clear()
            
            # Return average losses
            if update_count > 0:
                return {
                    'policy_loss': total_policy_loss / update_count,
                    'value_loss': total_value_loss / update_count,
                    'entropy_loss': total_entropy_loss / update_count
                }
            else:
                return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
            
        except Exception as e:
            print(f"Critical error in PPO update: {e}")
            self.buffer.clear()
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
    
    def save(self, filename):
        """Save model"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        """Load model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

# Data loading and preprocessing functions
def preprocess_data_with_sentiment(data):
    """Enhanced preprocessing for sentiment-aware data"""
    # Handle missing values for sentiment features
    sentiment_columns = [
        'sentiment_score', 'sentiment_confidence', 'sentiment_momentum_3d',
        'sentiment_momentum_7d', 'sentiment_volatility_5d', 'sentiment_trend',
        'days_since_news', 'coverage_quality', 'article_density_7d',
        'article_density_20d', 'news_frequency_weekly', 'sentiment_raw',
        'article_count_daily'
    ]
    
    # Fill missing sentiment data with neutral/zero values
    for col in sentiment_columns:
        if col in data.columns:
            if col == 'sentiment_score':
                data[col] = data[col].fillna(0.0)  # Neutral sentiment
            elif col == 'sentiment_confidence':
                data[col] = data[col].fillna(0.5)  # Medium confidence
            elif col == 'days_since_news':
                data[col] = data[col].fillna(30)   # Default to 30 days
            else:
                data[col] = data[col].fillna(0.0)  # Default to zero for other metrics
    
    return data.dropna()

def split_data_by_periods(data):
    """Split data into training, validation, and testing periods"""
    # Training: January 2015 - December 2017
    train_start = pd.Timestamp('2015-01-01')
    train_end = pd.Timestamp('2017-12-31')
    
    # Validation: January 2018 - December 2018
    val_start = pd.Timestamp('2018-01-01')
    val_end = pd.Timestamp('2018-12-31')
    
    # Testing: January 2019 - December 2019
    test_start = pd.Timestamp('2019-01-01')
    test_end = pd.Timestamp('2019-12-31')
    
    splits = {}
    for ticker, ticker_data in data.items():
        try:
            # Create a copy to avoid modifying original data
            data_copy = ticker_data.copy()
            
            # Reset index to work with Date as column
            data_copy = data_copy.reset_index()
            
            # Ensure Date column is timezone-naive
            if 'Date' in data_copy.columns:
                # Convert to datetime without timezone
                data_copy['Date'] = pd.to_datetime(data_copy['Date'], utc=False, errors='coerce')
                # Remove timezone info if present
                if data_copy['Date'].dt.tz is not None:
                    data_copy['Date'] = data_copy['Date'].dt.tz_localize(None)
                # Set as index
                data_copy = data_copy.set_index('Date')
            
            # Perform the splits using boolean indexing instead of .loc with timestamps
            train_mask = (data_copy.index >= train_start) & (data_copy.index <= train_end)
            val_mask = (data_copy.index >= val_start) & (data_copy.index <= val_end)
            test_mask = (data_copy.index >= test_start) & (data_copy.index <= test_end)
            
            splits[ticker] = {
                'train': data_copy[train_mask].copy(),
                'validation': data_copy[val_mask].copy(),
                'test': data_copy[test_mask].copy()
            }
            
            # Verify we have data for each period
            for period, period_data in splits[ticker].items():
                if len(period_data) == 0:
                    print(f"Warning: No data found for {ticker} in {period} period")
                else:
                    print(f"{ticker} {period}: {len(period_data)} rows from {period_data.index[0].date()} to {period_data.index[-1].date()}")
                    
        except Exception as e:
            print(f"Error splitting data for {ticker}: {e}")
            # Fallback: use integer-based splitting if datetime slicing fails
            total_rows = len(ticker_data)
            train_end_idx = int(total_rows * 0.6)  # 60% for training
            val_end_idx = int(total_rows * 0.8)    # 20% for validation
            
            splits[ticker] = {
                'train': ticker_data.iloc[:train_end_idx].copy(),
                'validation': ticker_data.iloc[train_end_idx:val_end_idx].copy(),
                'test': ticker_data.iloc[val_end_idx:].copy()
            }
            
            print(f"Used fallback splitting for {ticker}")
    
    return splits

def load_all_data_with_sentiment():
    """Load all CSV files with sentiment data"""
    data = {}
    
    # Define the file paths for sentiment-enhanced data
    available_files = [
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/AAPL_complete_processed.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/AMT_complete_processed.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/AMZN_complete_processed.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/CAT_complete_processed.csv', 
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/JPM_complete_processed.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/LIN_complete_processed.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/NEE_complete_processed.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/PG_complete_processed.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/UNH_complete_processed.csv',
        '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/Processed_data/XOM_complete_processed.csv'
    ]
    
    for filepath in available_files:
        # Extract ticker from filepath
        try:
            filename = filepath.split('/')[-1]
            ticker = filename.split('_')[0]
        except Exception as e:
            print(f"Error extracting ticker from filepath {filepath}: {e}")
            continue
        
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue
                
            # Read CSV
            stock_data = pd.read_csv(filepath)
            
            # Ensure Date column exists and convert to datetime
            if 'Date' not in stock_data.columns:
                print(f"Warning: {ticker} missing Date column")
                continue
                
            # Handle datetime conversion
            try:
                stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
                stock_data = stock_data.dropna(subset=['Date'])
                
                # Remove timezone info if present
                if hasattr(stock_data['Date'].dtype, 'tz') and stock_data['Date'].dtype.tz is not None:
                    stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)
                    
                stock_data.set_index('Date', inplace=True)
            except Exception as date_error:
                print(f"Error processing dates for {ticker}: {date_error}")
                continue
            
            # Verify required columns exist
            required_cols = ['close_price', 'sentiment_score']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            
            if missing_cols:
                print(f"Warning: {ticker} missing required columns: {missing_cols}")
                continue
                
            # Process the data if we have minimum requirements
            processed_data = preprocess_data_with_sentiment(stock_data)
            if len(processed_data) > 0:
                data[ticker] = processed_data
                print(f"{ticker} data loaded successfully: {len(data[ticker])} rows from {data[ticker].index[0].date()} to {data[ticker].index[-1].date()}")
            else:
                print(f"Warning: No valid data remaining after preprocessing for {ticker}")
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return data

# Enhanced Trading Environment for PPO with Sentiment Integration and Differential Sharpe Ratio Reward
class PPOTradingEnvWithDifferentialSharpe(gym.Env):
    """Enhanced Trading Environment for PPO with sentiment data integration and differential Sharpe ratio reward function"""
    
    def __init__(self, data, initial_balance=10000.0, transaction_fee_percent=0.001, 
                 window_size=20, benchmark_ticker='SPY', risk_free_rate=0.02):
        super(PPOTradingEnvWithDifferentialSharpe, self).__init__()
        
        self.data = data
        self.tickers = list(data.keys())
        self.num_stocks = len(self.tickers)
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = window_size
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
        # Align dates across all tickers
        self.align_dates()
        
        # State space: enhanced with sentiment indicators
        # Technical features (20) + Sentiment features (13) = 33 features per stock
        num_features_per_stock = 33  # Increased to include sentiment features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_stocks + 1 + (self.num_stocks * num_features_per_stock),), 
            dtype=np.float32
        )
        
        # Action space: continuous portfolio weights
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.num_stocks,), 
            dtype=np.float32
        )
        
        # Performance tracking for differential Sharpe ratio
        self.reset()
        
    def align_dates(self):
        """Align all stock data to have the same trading dates"""
        common_dates = None
        
        for ticker in self.tickers:
            if common_dates is None:
                common_dates = set(self.data[ticker].index)
            else:
                common_dates = common_dates.intersection(set(self.data[ticker].index))
        
        common_dates = sorted(list(common_dates))
        
        if len(common_dates) < self.window_size + 50:
            raise ValueError(f"Not enough common trading dates. Found only {len(common_dates)} common dates.")
        
        for ticker in self.tickers:
            self.data[ticker] = self.data[ticker].loc[common_dates]
        
        self.dates = common_dates
        
    def _get_observation(self):
        """Get current state observation with enhanced features including sentiment"""
        # Current positions and balance (normalized)
        total_value = self._calculate_portfolio_value()
        positions_balance = np.array(list(self.positions.values()) + [self.balance / total_value])
        
        # Market data features with sentiment integration
        market_features = []
        for ticker in self.tickers:
            stock_data = self.data[ticker].iloc[self.current_step]
            
            # Original technical features (20)
            technical_features = [
                # Price-based features
                stock_data['return_1d'],
                stock_data['return_5d'], 
                stock_data['return_10d'],
                stock_data['price_to_ma5'],
                stock_data['price_to_ma20'],
                stock_data['price_to_ma50'],
                
                # Technical indicators (normalized)
                stock_data['rsi_normalized'],
                stock_data['macd_normalized'],
                stock_data['macd_signal_normalized'],
                stock_data['macd_histogram_normalized'],
                
                # Bollinger Bands
                stock_data['bb_position'],
                stock_data['bb_width_normalized'],
                
                # Volume and volatility
                stock_data['volume_ratio_normalized'],
                stock_data['volatility_10d_normalized'],
                stock_data['volatility_20d_normalized'],
                
                # Support/Resistance
                stock_data['price_vs_support'],
                stock_data['price_vs_resistance'],
                
                # Trend indicators
                float(stock_data['trend_5d']),
                float(stock_data['trend_20d']),
                
                # Additional technical features
                stock_data['ma20_slope']
            ]
            
            # New sentiment features (13)
            sentiment_features = [
                stock_data['sentiment_score'],
                stock_data['sentiment_confidence'],
                stock_data['sentiment_momentum_3d'],
                stock_data['sentiment_momentum_7d'],
                stock_data['sentiment_volatility_5d'],
                stock_data['sentiment_trend'],
                stock_data['days_since_news'] / 100.0,  # Normalize days
                stock_data['coverage_quality'],
                stock_data['article_density_7d'],
                stock_data['article_density_20d'],
                stock_data['news_frequency_weekly'],
                stock_data['sentiment_raw'],
                stock_data['article_count_daily'] / 10.0  # Normalize article count
            ]
            
            # Combine technical and sentiment features
            features = technical_features + sentiment_features
            market_features.extend(features)
        
        # Combine all features
        observation = np.concatenate([positions_balance, np.array(market_features)])
        return observation.astype(np.float32)
    
    def _calculate_portfolio_value(self):
        """Calculate current portfolio value"""
        portfolio_value = self.balance
        
        for ticker in self.tickers:
            current_price = self.data[ticker]['close_price'].iloc[self.current_step]
            portfolio_value += self.positions[ticker] * current_price
        
        return portfolio_value
    
    def _execute_trades(self, actions):
        """Execute trades based on PPO actions with transaction costs"""
        total_cost = 0
        trades_executed = []
        
        for i, ticker in enumerate(self.tickers):
            current_price = self.data[ticker]['close_price'].iloc[self.current_step]
            current_position = self.positions[ticker]
            
            # Convert action to target position (as fraction of portfolio value)
            portfolio_value = self._calculate_portfolio_value()
            target_value = (actions[i] + 1) / 2 * portfolio_value  # Scale to [0, portfolio_value]
            target_shares = target_value / current_price
            
            # Calculate shares to trade
            shares_to_trade = target_shares - current_position
            
            if abs(shares_to_trade) > 0.01:  # Minimum trade threshold
                trade_value = abs(shares_to_trade) * current_price
                transaction_cost = trade_value * self.transaction_fee_percent
                
                if shares_to_trade > 0:  # Buying
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
                
                else:  # Selling
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
    
    def _calculate_differential_sharpe_reward(self):
        """Calculate differential Sharpe ratio reward"""
        current_value = self._calculate_portfolio_value()
        self.portfolio_history.append(current_value)
        
        # Need at least two values to calculate return
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Calculate current portfolio return
        portfolio_return = (current_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
        self.returns_history.append(portfolio_return)
        
        # Need at least a window of returns to calculate differential Sharpe
        if len(self.returns_history) < self.differential_window:
            # Return simple excess return for the initial period
            return portfolio_return - self.risk_free_rate
        
        # Calculate current window statistics
        current_window_returns = self.returns_history[-self.differential_window:]
        current_mean = np.mean(current_window_returns)
        current_std = np.std(current_window_returns)
        
        # Calculate previous window statistics (shifted by 1)
        if len(self.returns_history) >= self.differential_window + 1:
            previous_window_returns = self.returns_history[-(self.differential_window + 1):-1]
            previous_mean = np.mean(previous_window_returns)
            previous_std = np.std(previous_window_returns)
        else:
            # If we don't have enough previous data, use current as baseline
            previous_mean = current_mean
            previous_std = current_std
        
        # Calculate differential Sharpe ratio
        # Difference in mean excess returns
        current_excess_mean = current_mean - self.risk_free_rate
        previous_excess_mean = previous_mean - self.risk_free_rate
        delta_mean = current_excess_mean - previous_excess_mean
        
        # Difference in volatility (standard deviation)
        delta_std = current_std - previous_std
        
        # Differential Sharpe ratio calculation
        if abs(delta_std) > 1e-8:  # Avoid division by zero
            differential_sharpe = delta_mean / abs(delta_std)
        else:
            # If volatility difference is near zero, use sign of mean difference
            differential_sharpe = np.sign(delta_mean) * abs(delta_mean) * 100
        
        # Scale the reward to make it more actionable for RL
        reward = differential_sharpe * self.reward_scale
        
        # Add a baseline return component to maintain profitability incentive
        baseline_reward = portfolio_return - self.risk_free_rate
        
        # Combine differential Sharpe with baseline return (weighted)
        final_reward = 0.7 * reward + 0.3 * baseline_reward
        
        # Clip reward to prevent extreme values
        final_reward = np.clip(final_reward, -0.1, 0.1)
        
        return final_reward
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_history = [self.initial_balance]
        self.returns_history = []
        self.trade_history = []
        self.transaction_costs = []
        
        # Parameters for differential Sharpe ratio
        self.differential_window = 20  # Window size for calculating Sharpe ratio
        self.reward_scale = 10.0  # Scale factor for the differential Sharpe reward
        
        self.current_step = self.window_size
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Take a step in the environment with differential Sharpe ratio reward"""
        # Execute trades
        transaction_cost, trades = self._execute_trades(action)
        self.transaction_costs.append(transaction_cost)
        self.trade_history.extend(trades)
        
        # Move to next day
        self.current_step += 1
        
        # Calculate differential Sharpe ratio reward
        reward = self._calculate_differential_sharpe_reward()
        
        # Check if episode is done
        done = self.current_step >= len(self.dates) - 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': self._calculate_portfolio_value(),
            'balance': self.balance,
            'positions': self.positions.copy(),
            'transaction_cost': transaction_cost,
            'trades': trades,
            'differential_sharpe_reward': reward
        }
        
        return observation, reward, done, False, info

class PerformanceAnalyzer:
    """Comprehensive performance analysis with all requested metrics"""
    
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate  # 2% annual risk-free rate
        
    def calculate_all_metrics(self, portfolio_values, benchmark_returns=None, dates=None):
        """Calculate all performance metrics"""
        
        # Calculate daily returns
        portfolio_returns = np.array([
            (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            for i in range(1, len(portfolio_values))
        ])
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        calmar_ratio = self._calculate_calmar_ratio(portfolio_values, annual_return)
        
        # Alpha and Beta (if benchmark provided)
        alpha, beta = self._calculate_alpha_beta(portfolio_returns, benchmark_returns)
        
        # Drawdown metrics
        max_drawdown, drawdown_duration = self._calculate_drawdown_metrics(portfolio_values)
        
        # Trading metrics
        winning_percentage = self._calculate_winning_percentage(portfolio_returns)
        roi = total_return * 100
        
        return {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
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
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns):
        """Calculate Sortino ratio (only penalizes downside deviation)"""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, portfolio_values, annual_return):
        """Calculate Calmar ratio (annual return / max drawdown)"""
        max_dd = self._calculate_max_drawdown(portfolio_values)
        return annual_return / max_dd if max_dd > 0 else 0
    
    def _calculate_alpha_beta(self, portfolio_returns, benchmark_returns):
        """Calculate Alpha and Beta relative to benchmark"""
        if benchmark_returns is None or len(benchmark_returns) != len(portfolio_returns):
            return None, None
        
        # Calculate beta using covariance
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha (Jensen's alpha)
        portfolio_mean = np.mean(portfolio_returns) * 252
        benchmark_mean = np.mean(benchmark_returns) * 252
        alpha = portfolio_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
        
        return alpha, beta
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_drawdown_metrics(self, portfolio_values):
        """Calculate drawdown metrics including duration"""
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
        """Calculate percentage of profitable periods"""
        positive_returns = np.sum(returns > 0)
        return positive_returns / len(returns) * 100 if len(returns) > 0 else 0

def train_ppo_with_differential_sharpe(env, episodes=1000, save_path='./ppo_model', validation_env=None):
    """Train PPO agent with differential Sharpe ratio reward"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize PPO agent
    agent = PPO(state_dim, action_dim, batch_size=32, lr=1e-4, entropy_coef=0.015)
    
    # Training tracking
    episode_rewards = []
    portfolio_values = []
    validation_scores = []
    best_validation_score = -float('inf')
    episodes_without_improvement = 0
    patience = 200
    
    os.makedirs(save_path, exist_ok=True)
    
    # Update frequency
    episodes_per_update = 3
    episode_count = 0
    
    # Reduced warmup period to encourage faster learning
    warmup_episodes = 80
    
    print("Starting PPO training with Differential Sharpe Ratio reward...")
    
    for episode in range(episodes):
        try:
            state, _ = env.reset()
            episode_reward = 0
            done = False
            episode_steps = 0
            
            # More aggressive exploration that decreases more slowly
            exploration_noise = max(0.15 * (warmup_episodes - episode) / warmup_episodes, 0.02)
            
            while not done:
                if state is None or len(state) == 0:
                    break
                
                try:
                    action, log_prob, value = agent.select_action(state)
                    
                    # Enhanced exploration during warmup
                    if episode < warmup_episodes:
                        noise = np.random.normal(0, exploration_noise, action.shape)
                        action = action + noise
                    # Continue some exploration even after warmup
                    elif episode < warmup_episodes * 2:
                        noise = np.random.normal(0, exploration_noise * 0.3, action.shape)
                        action = action + noise
                    
                except Exception as e:
                    print(f"Error selecting action at episode {episode}: {e}")
                    break
                
                try:
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                except Exception as e:
                    print(f"Error taking step at episode {episode}: {e}")
                    break
                
                # Validate reward
                if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                try:
                    agent.store_experience(state, action, reward, value, log_prob, done)
                except Exception as e:
                    continue
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if episode_steps > 1000:
                    done = True
            
            episode_count += 1
            episode_rewards.append(episode_reward)
            
            if 'portfolio_value' in info:
                portfolio_values.append(info['portfolio_value'])
            else:
                portfolio_values.append(10000.0)
            
            # Update policy
            if episode_count % episodes_per_update == 0:
                try:
                    last_value = 0
                    if not done and next_state is not None:
                        try:
                            _, _, last_value = agent.select_action(next_state)
                        except:
                            last_value = 0
                    
                    if len(agent.buffer) >= agent.batch_size:
                        losses = agent.update(last_value)
                        if episode_count % 30 == 0:
                            agent.scheduler.step()
                    else:
                        agent.buffer.clear()
                        
                except Exception as e:
                    print(f"Error updating agent at episode {episode}: {e}")
                    agent.buffer.clear()
                    continue
            
            # Validation checks
            if episode % 100 == 0 and episode > 0:
                if validation_env is not None:
                    val_score = evaluate_agent(agent, validation_env)
                    validation_scores.append(val_score)
                    
                    if val_score > best_validation_score:
                        best_validation_score = val_score
                        episodes_without_improvement = 0
                        try:
                            agent.save(f"{save_path}/ppo_model_best.pth")
                        except Exception as save_error:
                            print(f"Warning: Could not save best model: {save_error}")
                    else:
                        episodes_without_improvement += 100
                    
                    if episodes_without_improvement >= patience and episode > 400:
                        print(f"Early stopping at episode {episode}")
                        break
                
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                avg_portfolio = np.mean(portfolio_values[-100:]) if portfolio_values else 10000
                
                print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Portfolio Value: {avg_portfolio:.2f}")
                
        except Exception as e:
            print(f"Critical error in episode {episode}: {e}")
            continue
    
    # Save final model
    try:
        agent.save(f"{save_path}/ppo_model_final.pth")
        print("Training completed successfully")
        
        # Load best model if available
        if validation_env is not None and os.path.exists(f"{save_path}/ppo_model_best.pth"):
            try:
                agent.load(f"{save_path}/ppo_model_best.pth")
            except Exception:
                pass
        
    except Exception as e:
        print(f"Error saving model: {e}")
    
    return agent, episode_rewards

def evaluate_agent(agent, env, episodes=5):
    """Evaluate agent performance on validation environment"""
    total_returns = []
    
    for _ in range(episodes):
        try:
            state, _ = env.reset()
            done = False
            initial_value = env._calculate_portfolio_value()
            
            while not done:
                action, _, _ = agent.select_action(state)
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

def analyze_differential_sharpe_performance(reward_history, portfolio_values, dates):
    """Analyze the differential Sharpe ratio reward performance"""
    if not reward_history or len(reward_history) < 2:
        return {
            'avg_differential_sharpe_reward': 0,
            'reward_volatility': 0,
            'positive_reward_percentage': 0,
            'correlation_with_returns': 0
        }
    
    # Calculate portfolio returns
    returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
              for i in range(1, len(portfolio_values))]
    
    # Align lengths
    min_len = min(len(reward_history), len(returns))
    reward_history = reward_history[:min_len]
    returns = returns[:min_len]
    
    # Calculate metrics
    avg_reward = np.mean(reward_history)
    reward_volatility = np.std(reward_history)
    positive_reward_pct = (np.sum(np.array(reward_history) > 0) / len(reward_history)) * 100
    
    # Correlation between differential Sharpe rewards and actual returns
    correlation = 0
    if len(reward_history) > 1 and len(returns) > 1:
        try:
            correlation = np.corrcoef(reward_history, returns)[0, 1]
            if np.isnan(correlation):
                correlation = 0
        except:
            correlation = 0
    
    return {
        'avg_differential_sharpe_reward': avg_reward,
        'reward_volatility': reward_volatility,
        'positive_reward_percentage': positive_reward_pct,
        'correlation_with_returns': correlation,
        'reward_history': reward_history
    }

def generate_differential_sharpe_plots(portfolio_values, daily_returns, dates, period_name, save_path, reward_analysis):
    """Generate performance visualizations including differential Sharpe analysis"""
    
    # 1. Portfolio Value Over Time
    plt.figure(figsize=(12, 6))
    if dates and len(dates) == len(portfolio_values):
        plt.plot(pd.to_datetime(dates), portfolio_values)
        plt.xlabel('Date')
    else:
        plt.plot(portfolio_values)
        plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'{period_name} Portfolio Performance with Differential Sharpe Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{period_name.lower()}_portfolio_performance.png")
    plt.close()
    
    # 2. Daily Returns Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.title(f'{period_name} Daily Returns Distribution')
    plt.axvline(np.mean(daily_returns), color='red', linestyle='--', label=f'Mean: {np.mean(daily_returns):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{period_name.lower()}_returns_distribution.png")
    plt.close()
    
    # 3. Differential Sharpe Reward Analysis
    if reward_analysis and 'reward_history' in reward_analysis:
        reward_history = reward_analysis['reward_history']
        if reward_history:
            plt.figure(figsize=(12, 8))
            
            # Reward over time
            plt.subplot(2, 1, 1)
            if dates and len(dates) >= len(reward_history):
                plt.plot(pd.to_datetime(dates[1:len(reward_history)+1]), reward_history, alpha=0.7)
            else:
                plt.plot(reward_history, alpha=0.7)
            plt.ylabel('Differential Sharpe Reward')
            plt.title(f'{period_name} Differential Sharpe Ratio Reward Over Time')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Reward distribution
            plt.subplot(2, 1, 2)
            plt.hist(reward_history, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Differential Sharpe Reward')
            plt.ylabel('Frequency')
            plt.title(f'{period_name} Differential Sharpe Reward Distribution')
            plt.axvline(np.mean(reward_history), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(reward_history):.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/{period_name.lower()}_differential_sharpe_analysis.png")
            plt.close()
    
    # 4. Reward vs Returns Correlation
    if reward_analysis and 'reward_history' in reward_analysis:
        reward_history = reward_analysis['reward_history']
        if reward_history and len(daily_returns) > 0:
            # Align lengths
            min_len = min(len(reward_history), len(daily_returns))
            rewards_aligned = reward_history[:min_len]
            returns_aligned = daily_returns[:min_len]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(rewards_aligned, returns_aligned, alpha=0.6)
            plt.xlabel('Differential Sharpe Reward')
            plt.ylabel('Daily Returns')
            plt.title(f'{period_name} Differential Sharpe Reward vs Returns Correlation')
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            if len(rewards_aligned) > 1:
                try:
                    z = np.polyfit(rewards_aligned, returns_aligned, 1)
                    p = np.poly1d(z)
                    plt.plot(rewards_aligned, p(rewards_aligned), "r--", alpha=0.8)
                    
                    # Add correlation text
                    correlation = reward_analysis.get('correlation_with_returns', 0)
                    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                            transform=plt.gca().transAxes, fontsize=12, 
                            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
                except:
                    pass
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/{period_name.lower()}_reward_return_correlation.png")
            plt.close()
    
    # 5. Drawdown Chart
    peak = portfolio_values[0]
    drawdowns = []
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)
    
    plt.figure(figsize=(12, 6))
    if dates and len(dates) == len(drawdowns):
        plt.fill_between(pd.to_datetime(dates), drawdowns, alpha=0.3, color='red')
        plt.plot(pd.to_datetime(dates), drawdowns, color='red')
        plt.xlabel('Date')
    else:
        plt.fill_between(range(len(drawdowns)), drawdowns, alpha=0.3, color='red')
        plt.plot(drawdowns, color='red')
        plt.xlabel('Trading Days')
    plt.ylabel('Drawdown')
    plt.title(f'{period_name} Drawdown Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{period_name.lower()}_drawdown.png")
    plt.close()

def backtest_with_differential_sharpe(agent, env, period_name="", save_path="./results"):
    """Comprehensive backtesting with differential Sharpe analysis"""
    os.makedirs(save_path, exist_ok=True)
    
    # Run backtest
    state, _ = env.reset()
    done = False
    portfolio_values = []
    daily_returns = []
    trade_history = []
    transaction_costs = []
    dates = []
    reward_history = []
    
    step = 0
    while not done:
        action, _, _ = agent.select_action(state)  # No noise during testing
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info['portfolio_value'])
        trade_history.extend(info['trades'])
        transaction_costs.append(info['transaction_cost'])
        reward_history.append(info.get('differential_sharpe_reward', reward))
        
        if step < len(env.dates):
            dates.append(env.dates[step])
        
        if len(portfolio_values) >= 2:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)
        
        step += 1
    
    # Calculate comprehensive metrics
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_all_metrics(portfolio_values)
    
    # Differential Sharpe-specific analysis
    reward_analysis = analyze_differential_sharpe_performance(reward_history, portfolio_values, dates)
    
    # Additional calculations
    total_transaction_costs = sum(transaction_costs)
    num_trades = len(trade_history)
    
    # Print results with differential Sharpe insights
    print(f"\n{period_name} Results (with Differential Sharpe Ratio Reward):")
    print("=" * 60)
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Annual Return: {metrics['annual_return']:.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
    if metrics['alpha'] is not None:
        print(f"Alpha: {metrics['alpha']:.2f}%")
        print(f"Beta: {metrics['beta']:.4f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Winning Percentage: {metrics['winning_percentage']:.2f}%")
    print(f"ROI: {metrics['roi']:.2f}%")
    print(f"Total Transaction Costs: ${total_transaction_costs:.2f}")
    print(f"Number of Trades: {num_trades}")
    
    # Differential Sharpe-specific insights
    print("\nDifferential Sharpe Ratio Analysis:")
    print("-" * 40)
    print(f"Avg Differential Sharpe Reward: {reward_analysis['avg_differential_sharpe_reward']:.4f}")
    print(f"Reward Volatility: {reward_analysis['reward_volatility']:.4f}")
    print(f"Positive Reward Percentage: {reward_analysis['positive_reward_percentage']:.1f}%")
    print(f"Reward-Return Correlation: {reward_analysis['correlation_with_returns']:.3f}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Date': dates[:len(portfolio_values)],
        'Portfolio_Value': portfolio_values,
        'Daily_Return': [0] + daily_returns,
        'Transaction_Cost': transaction_costs,
        'Differential_Sharpe_Reward': [0] + reward_history[:len(portfolio_values)-1]
    })
    results_df.to_csv(f"{save_path}/{period_name.lower()}_results.csv", index=False)
    
    # Save differential Sharpe analysis
    try:
        if reward_analysis and reward_analysis['reward_history']:
            reward_df = pd.DataFrame({
                'reward_history': reward_analysis['reward_history'],
                'daily_returns': daily_returns[:len(reward_analysis['reward_history'])]
            })
            reward_df.to_csv(f"{save_path}/{period_name.lower()}_differential_sharpe_analysis.csv", index=False)
        else:
            print(f"Warning: No reward data to save for {period_name}")
    except Exception as e:
        print(f"Warning: Could not save differential Sharpe analysis for {period_name}: {e}")
    
    # Save metrics summary
    metrics_df = pd.DataFrame({
        'Metric': [
            'Total Return (%)', 'Annual Return (%)', 'Volatility (%)',
            'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
            'Alpha (%)', 'Beta', 'Max Drawdown (%)',
            'Winning Percentage (%)', 'ROI (%)', 'Total Transaction Costs ($)',
            'Number of Trades', 'Avg Differential Sharpe Reward', 'Reward-Return Correlation'
        ],
        'Value': [
            metrics['total_return'], metrics['annual_return'], metrics['volatility'],
            metrics['sharpe_ratio'], metrics['sortino_ratio'], metrics['calmar_ratio'],
            metrics['alpha'] if metrics['alpha'] is not None else 'N/A', 
            metrics['beta'] if metrics['beta'] is not None else 'N/A',
            metrics['max_drawdown'], metrics['winning_percentage'], metrics['roi'],
            total_transaction_costs, num_trades, reward_analysis['avg_differential_sharpe_reward'],
            reward_analysis['correlation_with_returns']
        ]
    })
    metrics_df.to_csv(f"{save_path}/{period_name.lower()}_metrics.csv", index=False)
    
    # Generate visualizations
    generate_differential_sharpe_plots(
        portfolio_values, daily_returns, dates, period_name, save_path, reward_analysis
    )
    
    return metrics, results_df, reward_analysis

def generate_comprehensive_differential_sharpe_report(train_metrics, val_metrics, test_metrics, 
                                                    train_reward, val_reward, test_reward, save_dir):
    """Generate HTML report with differential Sharpe ratio analysis insights"""
    
    def format_metric(value, is_percentage=False, decimals=2):
        if value is None or (isinstance(value, str) and value == 'N/A'):
            return 'N/A'
        if is_percentage:
            return f"{value:.{decimals}f}%"
        return f"{value:.{decimals}f}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PPO Trading Algorithm with Differential Sharpe Ratio Reward - Complete Results</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
            .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 40px; padding: 20px; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; border-radius: 10px; }}
            .header h1 {{ margin: 0; font-size: 2.5em; }}
            .header p {{ margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }}
            
            .comparison-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .comparison-table th, .comparison-table td {{ padding: 12px; text-align: center; border: 1px solid #dee2e6; }}
            .comparison-table th {{ background-color: #343a40; color: white; }}
            .comparison-table tr:nth-child(even) {{ background-color: #f8f9fa; }}
            
            .reward-section {{ background-color: #e8f5e8; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745; margin: 20px 0; }}
            .reward-section h3 {{ margin-top: 0; color: #155724; }}
            
            .summary {{ background-color: #fff3cd; padding: 20px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 20px 0; }}
            .summary h3 {{ margin-top: 0; color: #856404; }}
            
            .positive {{ color: #28a745; }}
            .negative {{ color: #dc3545; }}
            .neutral {{ color: #6c757d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>PPO Trading Algorithm with Differential Sharpe Ratio Reward</h1>
                <p>Proximal Policy Optimization with Dynamic Risk-Adjusted Reward Function</p>
                <p>Training: 2015-2017 | Validation: 2018 | Testing: 2019</p>
            </div>
            
            <div class="summary">
                <h3>Executive Summary</h3>
                <p><strong>Algorithm:</strong> Proximal Policy Optimization (PPO) with Differential Sharpe Ratio reward</p>
                <p><strong>Key Innovation:</strong> Dynamic reward based on changes in risk-adjusted returns rather than absolute returns</p>
                <p><strong>Transaction Cost Impact:</strong> 0.1% per trade</p>
                <p><strong>Methodology:</strong> Reward = 0.7 * (Δmean_excess_return / Δvolatility) + 0.3 * excess_return</p>
            </div>
            
            <div class="reward-section">
                <h3>Differential Sharpe Ratio Reward Details</h3>
                <p><strong>Window Size:</strong> 20-day rolling window for Sharpe ratio calculation</p>
                <p><strong>Formula:</strong> DSR = (μ_current - μ_previous) / |σ_current - σ_previous|</p>
                <p><strong>Training Reward-Return Correlation:</strong> {format_metric(train_reward['correlation_with_returns'] if train_reward else None, False, 3)}</p>
                <p><strong>Validation Reward-Return Correlation:</strong> {format_metric(val_reward['correlation_with_returns'] if val_reward else None, False, 3)}</p>
                <p><strong>Testing Reward-Return Correlation:</strong> {format_metric(test_reward['correlation_with_returns'] if test_reward else None, False, 3)}</p>
            </div>
            
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Training (2015-2017)</th>
                        <th>Validation (2018)</th>
                        <th>Testing (2019)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Total Return</strong></td>
                        <td class="{'positive' if train_metrics and train_metrics['total_return'] > 0 else 'negative'}">{format_metric(train_metrics['total_return'] if train_metrics else None, True)}</td>
                        <td class="{'positive' if val_metrics and val_metrics['total_return'] > 0 else 'negative'}">{format_metric(val_metrics['total_return'] if val_metrics else None, True)}</td>
                        <td class="{'positive' if test_metrics and test_metrics['total_return'] > 0 else 'negative'}">{format_metric(test_metrics['total_return'] if test_metrics else None, True)}</td>
                    </tr>
                    <tr>
                        <td><strong>Sharpe Ratio</strong></td>
                        <td class="{'positive' if train_metrics and train_metrics['sharpe_ratio'] > 0 else 'negative'}">{format_metric(train_metrics['sharpe_ratio'] if train_metrics else None, False, 4)}</td>
                        <td class="{'positive' if val_metrics and val_metrics['sharpe_ratio'] > 0 else 'negative'}">{format_metric(val_metrics['sharpe_ratio'] if val_metrics else None, False, 4)}</td>
                        <td class="{'positive' if test_metrics and test_metrics['sharpe_ratio'] > 0 else 'negative'}">{format_metric(test_metrics['sharpe_ratio'] if test_metrics else None, False, 4)}</td>
                    </tr>
                    <tr>
                        <td><strong>Avg Differential Sharpe Reward</strong></td>
                        <td class="neutral">{format_metric(train_reward['avg_differential_sharpe_reward'] if train_reward else None, False, 4)}</td>
                        <td class="neutral">{format_metric(val_reward['avg_differential_sharpe_reward'] if val_reward else None, False, 4)}</td>
                        <td class="neutral">{format_metric(test_reward['avg_differential_sharpe_reward'] if test_reward else None, False, 4)}</td>
                    </tr>
                    <tr>
                        <td><strong>Maximum Drawdown</strong></td>
                        <td class="negative">{format_metric(train_metrics['max_drawdown'] if train_metrics else None, True)}</td>
                        <td class="negative">{format_metric(val_metrics['max_drawdown'] if val_metrics else None, True)}</td>
                        <td class="negative">{format_metric(test_metrics['max_drawdown'] if test_metrics else None, True)}</td>
                    </tr>
                    <tr>
                        <td><strong>Positive Reward %</strong></td>
                        <td class="neutral">{format_metric(train_reward['positive_reward_percentage'] if train_reward else None, True, 1)}</td>
                        <td class="neutral">{format_metric(val_reward['positive_reward_percentage'] if val_reward else None, True, 1)}</td>
                        <td class="neutral">{format_metric(test_reward['positive_reward_percentage'] if test_reward else None, True, 1)}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(f"{save_dir}/comprehensive_differential_sharpe_report.html", 'w') as f:
        f.write(html_content)

def run_differential_sharpe_experiment():
    """Run complete PPO experiment with differential Sharpe ratio reward function"""
    
    print("="*80)
    print("PPO TRADING ALGORITHM WITH DIFFERENTIAL SHARPE RATIO REWARD")
    print("="*80)
    print("Dataset: Sector-leading stocks with sentiment data (2015-2020)")
    print("Algorithm: Proximal Policy Optimization (PPO) with Differential Sharpe Reward")
    print("Reward Function: Differential Sharpe Ratio (dynamic risk-adjusted reward)")
    print("Transaction Costs: 0.1% per trade")
    print("State Enhancement: 33 features per stock (20 technical + 13 sentiment)")
    print("="*80)
    
    try:
        # Load data
        print("\nLoading sentiment-enhanced CSV data...")
        all_data = load_all_data_with_sentiment()
        
        if not all_data:
            print("Error: No sentiment data loaded successfully")
            return None, None, None, None
        
        print(f"\nLoaded data for {len(all_data)} stocks: {list(all_data.keys())}")
        
        # Split data
        print("\nSplitting data into periods...")
        data_splits = split_data_by_periods(all_data)
        
        # Create timestamp for this experiment
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = f"./ppo_differential_sharpe_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create environments
        train_data = {ticker: splits['train'] for ticker, splits in data_splits.items()}
        val_data = {ticker: splits['validation'] for ticker, splits in data_splits.items()}
        test_data = {ticker: splits['test'] for ticker, splits in data_splits.items()}
        
        train_env = PPOTradingEnvWithDifferentialSharpe(train_data, initial_balance=10000.0)
        val_env = PPOTradingEnvWithDifferentialSharpe(val_data, initial_balance=10000.0)
        test_env = PPOTradingEnvWithDifferentialSharpe(test_data, initial_balance=10000.0)
        
        # Phase 1: Training
        print("\n" + "="*60)
        print("PHASE 1: TRAINING WITH DIFFERENTIAL SHARPE RATIO REWARD")
        print("="*60)
        
        model_save_path = f"{experiment_dir}/model"
        agent, training_rewards = train_ppo_with_differential_sharpe(
            train_env, 
            episodes=1000,
            save_path=model_save_path,
            validation_env=val_env
        )
        
        # Training performance evaluation
        print("\nEvaluating training performance with differential Sharpe analysis...")
        train_results_path = f"{experiment_dir}/training_results"
        train_metrics, train_df, train_reward = backtest_with_differential_sharpe(
            agent, train_env, "Training", train_results_path
        )
        
        # Phase 2: Validation
        print("\n" + "="*60)
        print("PHASE 2: VALIDATION EVALUATION")
        print("="*60)
        
        val_results_path = f"{experiment_dir}/validation_results"
        val_metrics, val_df, val_reward = backtest_with_differential_sharpe(
            agent, val_env, "Validation", val_results_path
        )
        
        # Phase 3: Testing
        print("\n" + "="*60)
        print("PHASE 3: OUT-OF-SAMPLE TESTING (2019)")
        print("="*60)
        
        test_results_path = f"{experiment_dir}/testing_results"
        test_metrics, test_df, test_reward = backtest_with_differential_sharpe(
            agent, test_env, "Testing", test_results_path
        )
        
        # Generate comprehensive report
        generate_comprehensive_differential_sharpe_report(
            train_metrics, val_metrics, test_metrics,
            train_reward, val_reward, test_reward,
            experiment_dir
        )
        
        # Save training progress
        training_df = pd.DataFrame({
            'Episode': range(len(training_rewards)),
            'Episode_Reward': training_rewards
        })
        training_df.to_csv(f"{experiment_dir}/training_progress.csv", index=False)
        
        # Plot training progress
        plt.figure(figsize=(12, 8))
        
        # Plot episode rewards
        plt.subplot(2, 1, 1)
        plt.plot(training_rewards, alpha=0.7, color='blue')
        plt.plot([np.mean(training_rewards[max(0, i-50):i+1]) for i in range(len(training_rewards))], 
                 color='red', linewidth=2, label='50-Episode Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward (Differential Sharpe)')
        plt.title('PPO Training Progress with Differential Sharpe Reward - Episode Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative performance
        plt.subplot(2, 1, 2)
        cumulative_rewards = np.cumsum(training_rewards)
        plt.plot(cumulative_rewards, color='green', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('PPO Training Progress with Differential Sharpe Reward - Cumulative Rewards')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{experiment_dir}/training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nDifferential Sharpe ratio experiment completed! Results saved to: {experiment_dir}")
        
        # Summary with differential Sharpe insights
        print("\n" + "="*60)
        print("FINAL SUMMARY WITH DIFFERENTIAL SHARPE ANALYSIS")
        print("="*60)
        if train_metrics:
            print(f"Training Sharpe: {train_metrics['sharpe_ratio']:.4f}")
            if train_reward:
                print(f"Training Avg Differential Sharpe Reward: {train_reward['avg_differential_sharpe_reward']:.4f}")
                print(f"Training Reward-Return Correlation: {train_reward['correlation_with_returns']:.3f}")
        if val_metrics:
            print(f"Validation Sharpe: {val_metrics['sharpe_ratio']:.4f}")
            if val_reward:
                print(f"Validation Avg Differential Sharpe Reward: {val_reward['avg_differential_sharpe_reward']:.4f}")
        if test_metrics:
            print(f"Testing Sharpe: {test_metrics['sharpe_ratio']:.4f}")
            if test_reward:
                print(f"Testing Avg Differential Sharpe Reward: {test_reward['avg_differential_sharpe_reward']:.4f}")
        
        return train_metrics, val_metrics, test_metrics, experiment_dir
        
    except Exception as e:
        print(f"Critical error during differential Sharpe experiment: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def main():
    """Main execution function for differential Sharpe ratio PPO"""
    print("="*80)
    print("PPO TRADING ALGORITHM WITH DIFFERENTIAL SHARPE RATIO REWARD")
    print("="*80)
    print("Dynamic risk-adjusted reward function that optimizes for improvements")
    print("in risk-adjusted returns rather than absolute returns")
    print("Reward = 0.7 * (Δmean_excess_return / Δvolatility) + 0.3 * excess_return")
    print("="*80)
    
    try:
        # Run the complete differential Sharpe experiment
        results = run_differential_sharpe_experiment()
        
        if results and len(results) == 4:
            train_metrics, val_metrics, test_metrics, experiment_dir = results
            if experiment_dir:
                print(f"\n{'='*80}")
                print("DIFFERENTIAL SHARPE RATIO EXPERIMENT COMPLETED SUCCESSFULLY!")
                print(f"{'='*80}")
                print(f"Results directory: {experiment_dir}")
                return experiment_dir
        
        print("Experiment completed with some issues - check output above")
        return None
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the complete PPO trading experiment with differential Sharpe ratio reward
    result_dir = main()