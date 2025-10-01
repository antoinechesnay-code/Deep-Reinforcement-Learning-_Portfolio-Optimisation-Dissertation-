# Deep Reinforcement Learning for Portfolio Optimization

A comprehensive research implementation comparing sentiment-enhanced and baseline Deep Reinforcement Learning approaches for automated portfolio management.

## Overview

This repository contains the complete implementation of a master's dissertation investigating whether integrating financial sentiment analysis improves Deep Reinforcement Learning (DRL) portfolio optimization strategies. The research compares two state-of-the-art DRL algorithms—Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG)—with and without FinBERT-derived sentiment features.

## Key Findings

- **Baseline DRL Performance**: Both algorithms significantly outperformed the S&P 500 benchmark (31.49% return)
  - DDPG (no sentiment): 41.29% return, 2.80 Sharpe ratio
  - PPO (no sentiment): 35.50% return, 3.02 Sharpe ratio

- **Sentiment Integration Impact**: Contrary to initial hypotheses, sentiment enhancement decreased performance
  - DDPG (with sentiment): 37.85% return, 2.33 Sharpe ratio
  - PPO (with sentiment): 28.24% return (with inflated risk metrics due to over-conservatism)

- **Key Insight**: Data quality and sentiment bias proved critical—negative sentiment bias caused excessive conservatism, particularly in PPO

## Repository Structure
```
├── DDPG no sentiment.py          # Baseline DDPG implementation
├── DDPG sentiment.py              # Sentiment-enhanced DDPG
├── PPO no sentiment.py            # Baseline PPO implementation
├── PPO sentiment.py               # Sentiment-enhanced PPO
├── Dissertation final.docx        # Complete research documentation
└── README.md

```
## Methodology

### Data
- **Assets**: 10 sector-leading stocks (AAPL, AMZN, JPM, XOM, etc.)
- **Period**: 2015-2019
- **Split**: Training (2015-2017), Validation (2018), Testing (2019)
- **Features**: 20 technical indicators + 13 sentiment features (when applicable)

### Technical Indicators
- RSI (14-day), MACD, Bollinger Bands
- Volume ratios, price returns (1/5/20-day)
- Moving averages, volatility measures

### Sentiment Analysis
- **Model**: FinBERT (fine-tuned BERT for financial text)
- **Source**: Historical financial news via Wayback Machine
- **Processing**: Daily aggregation with confidence weighting
- **Features**: Sentiment score, momentum (3/7-day), volatility, coverage quality

### DRL Architecture

**State Space (72 dimensions)**:
- Stock features: 6 features × 10 stocks (60 dims)
- Portfolio features: weights (10) + cash + volatility (12 dims)

**Action Space**:
- Continuous portfolio weights [0, 0.2] per stock
- Constraints: Long-only, max 20% per asset, full investment ≤ 100%

**Reward Function**: Differential Sharpe Ratio (DSR)
DSR = (portfolio_return - benchmark_return) / std(differential_returns)

**Transaction Costs**: 0.1% per trade

### Algorithms

**DDPG** (Off-policy, deterministic):
- Actor-critic architecture with target networks
- Experience replay buffer
- Deterministic policy gradient

**PPO** (On-policy, stochastic):
- Clipped surrogate objective
- Multiple epochs per batch
- Conservative policy updates

## Requirements

```bash
pip install numpy pandas torch gymnasium matplotlib yfinance
pip install transformers  # For FinBERT sentiment analysis
```


## Usage
### Running Baseline Models

```bash
# DDPG without sentiment
python "DDPG no sentiment.py"

# PPO without sentiment
python "PPO no sentiment.py"
```

### Running Sentiment-Enhanced Models

```bash
# DDPG with sentiment
python "DDPG sentiment.py"

# PPO with sentiment
python "PPO sentiment.py"
```

## Expected Outputs

Each script generates:

- Performance metrics (returns, Sharpe/Sortino/Calmar ratios)
- Portfolio value progression
- Trading history and transaction costs
- Visualization plots
- CSV results files

## Key Implementation Details

### DDPG Architecture

- **Actor Network**: 400-300-action_dim (with dropout 0.3)
- **Critic Network**: 400-300-1 (with dropout 0.2-0.3)
- **Learning Rates**: Actor $1 \times 10^{-4}$, Critic $1 \times 10^{-3}$
- **Target Update**: Soft update with $\tau=0.001$

### PPO Architecture

- **Shared Layers**: 512-256-128 (with dropout 0.2-0.3)
- **Actor Head**: Stochastic policy with learned std
- **Critic Head**: Single value output
- **Hyperparameters**: $\epsilon=0.15$, updates=3, batch=32

### Regularization Techniques

- Dropout layers (0.2-0.3)
- Gradient clipping (0.5)
- Weight decay ($1 \times 10^{-4}$)
- Learning rate scheduling

## Performance Metrics

- **Total Return**: Portfolio appreciation over test period
- **Sharpe Ratio**: Risk-adjusted returns vs 2% risk-free rate
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return/max drawdown
- **Maximum Drawdown**: Peak-to-trough decline
- **Winning Percentage**: Profitable trading days

## Research Contributions

- **Empirical Evidence**: First comprehensive comparison showing sentiment integration can degrade performance if data quality is poor.
- **Algorithm Comparison**: DDPG demonstrates greater robustness to sentiment noise than PPO.
- **Methodological Framework**: Rigorous evaluation with temporal splits, transaction costs, and realistic constraints.
- **Data Quality Insights**: Identifies critical challenges in news coverage irregularity and sentiment bias.

## Limitations

- **Data Bias**: Negative sentiment skew from news sources.
- **Coverage Gaps**: Irregular news flow across stocks.
- **Computational Cost**: Real-time sentiment processing latency.
- **Market Impact**: Not modeled for large portfolios.
- **Regulatory**: Black-box explainability challenges.

## Future Work

- **Multi-Source Sentiment**: Integrate social media, SEC filings, analyst reports.
- **Debiasing Techniques**: Address sentiment imbalances systematically.
- **Ensemble Methods**: Combine strengths of different DRL algorithms.
- **Adaptive Sensitivity**: Dynamic sentiment weighting based on market regime.
- **Regime Detection**: Identify bull/bear/volatile periods for context-aware decisions.
