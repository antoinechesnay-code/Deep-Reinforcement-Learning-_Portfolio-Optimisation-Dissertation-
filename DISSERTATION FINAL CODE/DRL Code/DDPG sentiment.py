import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class ReplayBuffer:
    """Experience replay buffer for DDPG"""
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def size(self):
        return len(self.buffer)

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class Actor(nn.Module):
    """Actor network for DDPG"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Output between 0 and 1
        
        # Apply constraints: normalize to ensure sum <= 1 and max weight <= 0.2
        x = torch.clamp(x, 0, 0.2)  # Max 20% per stock
        x_sum = torch.sum(x, dim=-1, keepdim=True)
        x = torch.where(x_sum > 0, x / x_sum, x)  # Normalize only if sum > 0
        x = torch.clamp(x, 0, 0.2)  # Ensure constraint after normalization
        
        return x

class Critic(nn.Module):
    """Critic network for DDPG"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DDPG:
    """Deep Deterministic Policy Gradient Agent"""
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.99, tau=0.001, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        
        # Replay buffer and noise
        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(action_dim)
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = self.noise.sample()
            action += 0.1 * noise  # Scale noise
            action = np.clip(action, 0, 0.2)  # Ensure constraints
            action_sum = np.sum(action)
            if action_sum > 0:
                action = action / action_sum  # Normalize
                action = np.clip(action, 0, 0.2)  # Ensure constraint after normalization
        
        return action
    
    def update(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class PortfolioEnvironment:
    """Portfolio optimization environment"""
    def __init__(self, data_dict, initial_capital=100000, transaction_cost=0.001, risk_free_rate=0.02):
        self.data_dict = data_dict
        self.tickers = sorted(list(data_dict.keys()))  # Ensure consistent ordering
        self.n_assets = len(self.tickers)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
        print(f"Environment initialized with {self.n_assets} assets: {self.tickers}")
        
        # Data alignment
        self.align_data()
        self.reset()
        
    def align_data(self):
        """Align all ticker data to common dates"""
        if not self.data_dict:
            print("No data loaded!")
            return
            
        # Get common dates
        date_sets = [set(df['Date']) for df in self.data_dict.values()]
        if not date_sets:
            print("No date sets found!")
            return
            
        common_dates = date_sets[0]
        for date_set in date_sets[1:]:
            common_dates = common_dates.intersection(date_set)
        
        common_dates = sorted(list(common_dates))
        
        if not common_dates:
            print("No common dates found!")
            return
        
        # Filter data to common dates
        for ticker in self.tickers:
            self.data_dict[ticker] = self.data_dict[ticker][
                self.data_dict[ticker]['Date'].isin(common_dates)
            ].sort_values('Date').reset_index(drop=True)
        
        self.dates = common_dates
        self.n_steps = len(common_dates)
        print(f"Aligned data: {len(common_dates)} common dates from {min(common_dates)} to {max(common_dates)}")
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = np.zeros(self.n_assets)
        self.weights = np.zeros(self.n_assets)
        self.portfolio_returns = []
        self.portfolio_values = [self.initial_capital]
        
        # Initialize rolling windows for metrics
        self.return_window = deque(maxlen=20)
        self.benchmark_window = deque(maxlen=20)
        self.diff_return_window = deque(maxlen=20)
        
        return self.get_state()
    
    def get_state(self):
        """Get current state representation"""
        if self.current_step >= self.n_steps:
            return None
        
        state_features = []
        
        # Individual stock features (60 dimensions)
        for i, ticker in enumerate(self.tickers):
            try:
                data = self.data_dict[ticker].iloc[self.current_step]
                stock_features = [
                    float(data.get('return_1d', 0.0)),
                    float(data.get('rsi_normalized', 0.5)),
                    float(data.get('macd_normalized', 0.0)),
                    float(data.get('bb_position', 0.5)),
                    float(data.get('volume_ratio_normalized', 1.0)),
                    float(data.get('sentiment_score', 0.0))
                ]
                state_features.extend(stock_features)
            except Exception as e:
                print(f"Error getting features for {ticker} at step {self.current_step}: {e}")
                # Use default values if data is missing
                state_features.extend([0.0, 0.5, 0.0, 0.5, 1.0, 0.0])
        
        # Portfolio-level features (12 dimensions)
        cash_position = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0.0
        portfolio_vol = self.get_portfolio_volatility()
        
        portfolio_features = list(self.weights) + [cash_position, portfolio_vol]
        state_features.extend(portfolio_features)
        
        # Ensure we have exactly 72 features
        while len(state_features) < 72:
            state_features.append(0.0)
        
        state_features = state_features[:72]  # Truncate if too many
        
        return np.array(state_features, dtype=np.float32)
    
    def get_portfolio_volatility(self):
        """Calculate 20-day portfolio volatility"""
        if len(self.return_window) < 2:
            return 0.0
        returns = np.array(self.return_window)
        return float(np.std(returns) * np.sqrt(252))  # Annualized
    
    def step(self, action):
        """Execute one step in environment"""
        if self.current_step >= self.n_steps - 1:
            return None, 0, True, {}
        
        # Ensure action is valid
        action = np.array(action)
        action = np.clip(action, 0, 0.2)  # Max 20% per asset
        action_sum = np.sum(action)
        if action_sum > 1.0:
            action = action / action_sum  # Normalize if sum > 1
        
        # Current prices
        current_prices = []
        for ticker in self.tickers:
            try:
                price = float(self.data_dict[ticker].iloc[self.current_step]['close_price'])
                current_prices.append(max(price, 0.01))  # Ensure positive prices
            except:
                current_prices.append(100.0)  # Default price
        current_prices = np.array(current_prices)
        
        # Calculate transaction costs
        weight_changes = np.abs(action - self.weights)
        transaction_costs = self.transaction_cost * np.sum(weight_changes) * self.portfolio_value
        
        # Update positions
        available_capital = self.portfolio_value - transaction_costs
        new_positions = action * available_capital / current_prices
        self.positions = new_positions
        self.weights = action
        self.cash = available_capital - np.sum(self.positions * current_prices)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate returns
        portfolio_return = 0.0
        benchmark_return = 0.0
        
        if self.current_step < self.n_steps:
            next_prices = []
            benchmark_returns = []
            
            for ticker in self.tickers:
                try:
                    next_price = float(self.data_dict[ticker].iloc[self.current_step]['close_price'])
                    next_prices.append(max(next_price, 0.01))
                    
                    # Get benchmark return
                    bench_ret = float(self.data_dict[ticker].iloc[self.current_step].get('return_1d', 0.0))
                    benchmark_returns.append(bench_ret)
                except:
                    next_prices.append(current_prices[self.tickers.index(ticker)])
                    benchmark_returns.append(0.0)
            
            next_prices = np.array(next_prices)
            
            # Portfolio return
            new_portfolio_value = np.sum(self.positions * next_prices) + self.cash
            portfolio_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
            
            # Benchmark return (equal weight)
            benchmark_return = np.mean(benchmark_returns)
            
            # Update tracking
            self.portfolio_value = new_portfolio_value
            self.portfolio_returns.append(portfolio_return)
            self.portfolio_values.append(self.portfolio_value)
            self.return_window.append(portfolio_return)
            self.benchmark_window.append(benchmark_return)
            
            # Calculate Differential Sharpe Ratio
            diff_return = portfolio_return - benchmark_return
            self.diff_return_window.append(diff_return)
            
            reward = self.calculate_differential_sharpe_ratio()
        else:
            reward = 0.0
        
        next_state = self.get_state()
        done = self.current_step >= self.n_steps - 1
        
        info = {
            'portfolio_return': portfolio_return,
            'portfolio_value': self.portfolio_value,
            'transaction_costs': transaction_costs,
            'benchmark_return': benchmark_return
        }
        
        return next_state, reward, done, info
    
    def calculate_differential_sharpe_ratio(self):
        """Calculate Differential Sharpe Ratio as reward"""
        if len(self.diff_return_window) < 2:
            return 0.0
        
        diff_returns = np.array(self.diff_return_window)
        mean_diff = np.mean(diff_returns)
        std_diff = np.std(diff_returns)
        
        if std_diff == 0 or np.isnan(std_diff):
            return 0.0
        
        dsr = mean_diff / std_diff
        return float(np.clip(dsr, -10, 10))  # Clip extreme values

class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_metrics(portfolio_returns, benchmark_returns=None, risk_free_rate=0.02):
        """Calculate all performance metrics"""
        portfolio_returns = np.array(portfolio_returns)
        
        if len(portfolio_returns) == 0:
            return {}
        
        if benchmark_returns is None:
            benchmark_returns = np.zeros_like(portfolio_returns)
        else:
            benchmark_returns = np.array(benchmark_returns)
        
        # Basic statistics
        total_return = np.prod(1 + portfolio_returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Alpha and Beta
        daily_rf = risk_free_rate / 252
        excess_portfolio = portfolio_returns - daily_rf
        excess_benchmark = benchmark_returns - daily_rf
        
        if len(benchmark_returns) > 1 and np.var(excess_benchmark) > 0:
            beta = np.cov(excess_portfolio, excess_benchmark)[0, 1] / np.var(excess_benchmark)
            alpha = np.mean(excess_portfolio) - beta * np.mean(excess_benchmark)
            alpha_annualized = alpha * 252
        else:
            beta = 0
            alpha_annualized = 0
        
        # Consistency metrics
        winning_percentage = len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns)
        
        # Information Ratio
        if len(benchmark_returns) > 1:
            tracking_error = np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)
            information_ratio = (annualized_return - np.mean(benchmark_returns) * 252) / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0
        
        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Maximum Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Alpha (Annualized)': alpha_annualized,
            'Beta': beta,
            'Winning Percentage': winning_percentage,
            'Information Ratio': information_ratio
        }

def load_and_prepare_data(file_paths):
    """Load and prepare data from CSV files"""
    data_dict = {}
    
    print("Loading data files...")
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        try:
            # Extract ticker from filename
            ticker = os.path.basename(file_path).split('_')[0]
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Convert Date column, handling timezone issues
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                # Remove timezone information if present
                if df['Date'].dtype.name == 'datetime64[ns, UTC]' or 'datetime64[ns,' in str(df['Date'].dtype):
                    df['Date'] = df['Date'].dt.tz_localize(None)
            except Exception as date_error:
                print(f"Date conversion warning for {ticker}: {date_error}")
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                except:
                    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
            
            # Check for required columns
            required_columns = ['Date', 'close_price']
            optional_columns = ['return_1d', 'rsi_normalized', 'macd_normalized', 
                              'bb_position', 'volume_ratio_normalized', 'sentiment_score']
            
            missing_req = [col for col in required_columns if col not in df.columns]
            if missing_req:
                print(f"Skipping {ticker}: Missing required columns {missing_req}")
                continue
            
            # Add missing optional columns with default values
            for col in optional_columns:
                if col not in df.columns:
                    if col == 'return_1d':
                        df[col] = df['close_price'].pct_change().fillna(0)
                    elif col in ['rsi_normalized', 'bb_position']:
                        df[col] = 0.5
                    elif col == 'sentiment_score':
                        df[col] = 0.0
                    else:
                        df[col] = 1.0
            
            # Handle any remaining NaN values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Store in dictionary
            data_dict[ticker] = df
            print(f"Loaded {ticker}: {len(df)} rows, date range: {df['Date'].min()} to {df['Date'].max()}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(data_dict)} datasets")
    return data_dict

def train_ddpg(data_dict, episodes=500, split_date='2018-01-01', test_date='2019-01-01'):
    """Train DDPG agent"""
    if not data_dict:
        print("No data available for training!")
        return None, None, None, [], []
    
    # Convert string dates to pandas datetime for comparison
    split_date = pd.to_datetime(split_date)
    test_date = pd.to_datetime(test_date)
    
    # Split data
    train_data = {}
    val_data = {}
    test_data = {}
    
    for ticker, df in data_dict.items():
        train_mask = df['Date'] < split_date
        val_mask = (df['Date'] >= split_date) & (df['Date'] < test_date)
        test_mask = df['Date'] >= test_date
        
        train_data[ticker] = df[train_mask].reset_index(drop=True)
        val_data[ticker] = df[val_mask].reset_index(drop=True)
        test_data[ticker] = df[test_mask].reset_index(drop=True)
    
    print(f"Data split - Train: {len(train_data[list(train_data.keys())[0]])}, "
          f"Val: {len(val_data[list(val_data.keys())[0]])}, "
          f"Test: {len(test_data[list(test_data.keys())[0]])}")
    
    # Create environments
    train_env = PortfolioEnvironment(train_data)
    val_env = PortfolioEnvironment(val_data)
    test_env = PortfolioEnvironment(test_data)
    
    # Create agent
    state_dim = 72  # As per methodology
    action_dim = len(data_dict)  # Number of assets
    agent = DDPG(state_dim, action_dim)
    
    # Training loop
    training_rewards = []
    validation_rewards = []
    
    print(f"\nStarting DDPG Training for {episodes} episodes...")
    
    for episode in range(episodes):
        state = train_env.reset()
        episode_reward = 0
        steps = 0
        
        while state is not None:
            action = agent.select_action(state)
            next_state, reward, done, info = train_env.step(action)
            
            if next_state is not None:
                agent.replay_buffer.add(state, action, reward, next_state, done)
            
            if agent.replay_buffer.size() > 64:
                agent.update()
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        training_rewards.append(episode_reward)
        
        # Validation every 50 episodes
        if episode % 50 == 0:
            val_reward = evaluate_agent(agent, val_env)
            validation_rewards.append(val_reward)
            print(f"Episode {episode}: Training Reward: {episode_reward:.4f}, "
                  f"Validation Reward: {val_reward:.4f}, Steps: {steps}")
    
    print("Training completed!")
    
    # Final evaluation
    print("\nEvaluating on test set...")
    test_results = evaluate_agent_detailed(agent, test_env)
    
    # Calculate benchmark performance
    benchmark_results = evaluate_benchmark(test_env)
    
    return agent, test_results, benchmark_results, training_rewards, validation_rewards

def evaluate_agent(agent, env):
    """Simple evaluation for validation"""
    state = env.reset()
    total_reward = 0
    steps = 0
    
    while state is not None:
        action = agent.select_action(state, add_noise=False)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    return total_reward

def evaluate_agent_detailed(agent, env):
    """Detailed evaluation with metrics"""
    state = env.reset()
    portfolio_returns = []
    portfolio_values = []
    actions_taken = []
    benchmark_returns = []
    
    while state is not None:
        action = agent.select_action(state, add_noise=False)
        actions_taken.append(action.copy())
        state, reward, done, info = env.step(action)
        
        if 'portfolio_return' in info:
            portfolio_returns.append(info['portfolio_return'])
            portfolio_values.append(info['portfolio_value'])
            benchmark_returns.append(info.get('benchmark_return', 0.0))
        
        if done:
            break
    
    return {
        'portfolio_returns': portfolio_returns,
        'portfolio_values': portfolio_values,
        'actions': actions_taken,
        'benchmark_returns': benchmark_returns
    }

def evaluate_benchmark(env):
    """Evaluate equal-weight benchmark strategy"""
    state = env.reset()
    portfolio_returns = []
    portfolio_values = []
    n_assets = env.n_assets
    equal_weights = np.ones(n_assets) / n_assets  # Equal weight allocation
    
    while state is not None:
        state, reward, done, info = env.step(equal_weights)
        
        if 'portfolio_return' in info:
            portfolio_returns.append(info['portfolio_return'])
            portfolio_values.append(info['portfolio_value'])
        
        if done:
            break
    
    return {
        'portfolio_returns': portfolio_returns,
        'portfolio_values': portfolio_values
    }

def plot_results(ddpg_results, benchmark_results, training_rewards):
    """Plot comprehensive results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Portfolio value comparison
    if ddpg_results['portfolio_values'] and benchmark_results['portfolio_values']:
        axes[0, 0].plot(ddpg_results['portfolio_values'], label='DDPG Strategy', linewidth=2, color='blue')
        axes[0, 0].plot(benchmark_results['portfolio_values'], label='Equal Weight Benchmark', linewidth=2, color='red')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative returns
    if ddpg_results['portfolio_returns'] and benchmark_results['portfolio_returns']:
        ddpg_cumret = np.cumprod(1 + np.array(ddpg_results['portfolio_returns'])) - 1
        bench_cumret = np.cumprod(1 + np.array(benchmark_results['portfolio_returns'])) - 1
        
        axes[0, 1].plot(ddpg_cumret, label='DDPG Strategy', linewidth=2, color='blue')
        axes[0, 1].plot(bench_cumret, label='Equal Weight Benchmark', linewidth=2, color='red')
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Training progress
    if training_rewards:
        axes[1, 0].plot(training_rewards, color='green', alpha=0.7)
        # Add moving average for smoother visualization
        if len(training_rewards) > 10:
            window = min(50, len(training_rewards)//10)
            moving_avg = pd.Series(training_rewards).rolling(window).mean()
            axes[1, 0].plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window})')
            axes[1, 0].legend()
        axes[1, 0].set_title('Training Progress')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Reward')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Portfolio allocations over time
    if ddpg_results['actions']:
        actions = np.array(ddpg_results['actions'])
        colors = plt.cm.tab10(np.linspace(0, 1, actions.shape[1]))
        
        for i in range(min(actions.shape[1], 10)):  # Limit to 10 assets for readability
            axes[1, 1].plot(actions[:, i], label=f'Asset {i+1}', alpha=0.8, color=colors[i])
        
        axes[1, 1].set_title('Portfolio Weights Over Time')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 0.25)  # Show up to 25% allocation
    
    plt.tight_layout()
    plt.show()

def print_detailed_metrics(ddpg_results, benchmark_results, ddpg_metrics, benchmark_metrics):
    """Print comprehensive performance analysis"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Summary Statistics
    print("\n📊 SUMMARY STATISTICS")
    print("-" * 50)
    ddpg_total = ddpg_results['portfolio_values'][-1] - ddpg_results['portfolio_values'][0] if ddpg_results['portfolio_values'] else 0
    bench_total = benchmark_results['portfolio_values'][-1] - benchmark_results['portfolio_values'][0] if benchmark_results['portfolio_values'] else 0
    
    print(f"Initial Capital: ${100000:,.2f}")
    print(f"DDPG Final Value: ${ddpg_results['portfolio_values'][-1]:,.2f}" if ddpg_results['portfolio_values'] else "N/A")
    print(f"Benchmark Final Value: ${benchmark_results['portfolio_values'][-1]:,.2f}" if benchmark_results['portfolio_values'] else "N/A")
    print(f"DDPG Total P&L: ${ddpg_total:,.2f}")
    print(f"Benchmark Total P&L: ${bench_total:,.2f}")
    print(f"Outperformance: ${ddpg_total - bench_total:,.2f}")
    
    # Performance Metrics Comparison
    print(f"\n📈 PERFORMANCE METRICS COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<25} {'DDPG Strategy':<18} {'Equal Weight':<18} {'Difference':<15}")
    print("-" * 80)
    
    metrics_order = [
        'Total Return',
        'Annualized Return', 
        'Volatility',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Maximum Drawdown',
        'Calmar Ratio',
        'Alpha (Annualized)',
        'Beta',
        'Information Ratio',
        'Winning Percentage'
    ]
    
    for metric in metrics_order:
        if metric in ddpg_metrics and metric in benchmark_metrics:
            ddpg_val = ddpg_metrics[metric]
            bench_val = benchmark_metrics[metric]
            diff = ddpg_val - bench_val
            
            # Format percentages
            if metric in ['Total Return', 'Annualized Return', 'Volatility', 'Maximum Drawdown', 'Winning Percentage']:
                ddpg_str = f"{ddpg_val*100:.2f}%"
                bench_str = f"{bench_val*100:.2f}%"
                diff_str = f"{diff*100:+.2f}%"
            else:
                ddpg_str = f"{ddpg_val:.4f}"
                bench_str = f"{bench_val:.4f}"
                diff_str = f"{diff:+.4f}"
            
            print(f"{metric:<25} {ddpg_str:<18} {bench_str:<18} {diff_str:<15}")
    
    # Risk Analysis
    print(f"\n⚠️  RISK ANALYSIS")
    print("-" * 50)
    
    if ddpg_results['portfolio_returns']:
        returns = np.array(ddpg_results['portfolio_returns'])
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * 100
        
        print(f"Value at Risk (95%): {var_95:.2f}%")
        print(f"Value at Risk (99%): {var_99:.2f}%") 
        print(f"Conditional VaR (95%): {cvar_95:.2f}%")
        
        # Skewness and Kurtosis
        from scipy import stats
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        print(f"Skewness: {skewness:.4f}")
        print(f"Kurtosis: {kurtosis:.4f}")
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        print(f"Max Consecutive Loss Days: {max_consecutive_losses}")
    
    # Trading Statistics
    print(f"\n💼 TRADING STATISTICS")
    print("-" * 50)
    
    if ddpg_results['actions']:
        actions = np.array(ddpg_results['actions'])
        
        # Portfolio concentration
        avg_weights = np.mean(actions, axis=0)
        max_weight = np.max(avg_weights)
        min_weight = np.min(avg_weights)
        weight_std = np.std(avg_weights)
        
        print(f"Average Max Asset Weight: {max_weight:.2%}")
        print(f"Average Min Asset Weight: {min_weight:.2%}")
        print(f"Weight Distribution Std: {weight_std:.4f}")
        
        # Turnover analysis
        weight_changes = np.diff(actions, axis=0)
        daily_turnover = np.mean(np.sum(np.abs(weight_changes), axis=1))
        
        print(f"Average Daily Turnover: {daily_turnover:.2%}")
        
        # Asset utilization
        asset_usage = np.mean(actions > 0.001, axis=0)  # Assets with >0.1% weight
        avg_assets_used = np.mean(np.sum(actions > 0.001, axis=1))
        
        print(f"Average Assets Used: {avg_assets_used:.1f} out of {actions.shape[1]}")

def main():
    """Main execution function with comprehensive testing"""
    
    # File paths
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
    
    print("🚀 DDPG Portfolio Optimization with Sentiment Analysis")
    print("="*60)
    
    # Load data
    data_dict = load_and_prepare_data(available_files)
    
    if not data_dict:
        print("❌ No data could be loaded. Please check file paths.")
        return None, None, None
    
    # Train the DDPG agent
    print(f"\n🎯 Training DDPG agent on {len(data_dict)} assets...")
    agent, ddpg_results, benchmark_results, training_rewards, validation_rewards = train_ddpg(
        data_dict, episodes=300, split_date='2018-01-01', test_date='2019-01-01'
    )
    
    if agent is None:
        print("❌ Training failed!")
        return None, None, None
    
    # Calculate performance metrics
    print("\n📊 Calculating performance metrics...")
    
    # DDPG metrics
    ddpg_metrics = PerformanceMetrics.calculate_metrics(
        ddpg_results['portfolio_returns'], 
        ddpg_results.get('benchmark_returns', benchmark_results['portfolio_returns'])
    )
    
    # Benchmark metrics  
    benchmark_metrics = PerformanceMetrics.calculate_metrics(
        benchmark_results['portfolio_returns'],
        benchmark_results['portfolio_returns']
    )
    
    # Print detailed analysis
    print_detailed_metrics(ddpg_results, benchmark_results, ddpg_metrics, benchmark_metrics)
    
    # Plot results
    print(f"\n📈 Generating performance charts...")
    plot_results(ddpg_results, benchmark_results, training_rewards)
    
    # Save results summary
    results_summary = {
        'DDPG_Metrics': ddpg_metrics,
        'Benchmark_Metrics': benchmark_metrics,
        'Final_Portfolio_Value_DDPG': ddpg_results['portfolio_values'][-1] if ddpg_results['portfolio_values'] else 0,
        'Final_Portfolio_Value_Benchmark': benchmark_results['portfolio_values'][-1] if benchmark_results['portfolio_values'] else 0,
        'Training_Episodes': len(training_rewards),
        'Test_Period_Days': len(ddpg_results['portfolio_returns']) if ddpg_results['portfolio_returns'] else 0
    }
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📁 Results summary available in returned dictionary")
    
    return agent, ddpg_results, benchmark_results, results_summary

if __name__ == "__main__":
    try:
        # Install required packages if not available
        import scipy
    except ImportError:
        print("Installing required package: scipy")
        import subprocess
        subprocess.check_call(["pip", "install", "scipy"])
        import scipy
    
    agent, ddpg_results, benchmark_results, summary = main()