import pandas as pd
import numpy as np
from transformers import pipeline
from datetime import datetime, timedelta
import warnings
import os
from scipy import stats

warnings.filterwarnings('ignore')

class SentimentAnalysisPipeline:
    def __init__(self):
        """Initialize FinBERT model for sentiment analysis"""
        print("Loading FinBERT model...")
        self.finbert = pipeline("sentiment-analysis", 
                               model="ProsusAI/finbert",
                               tokenizer="ProsusAI/finbert",
                               device=0 if self._check_gpu() else -1)
        print("FinBERT model loaded successfully!")
        
    def _check_gpu(self):
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of single text using FinBERT
        Returns: (score, confidence)
        """
        if pd.isna(text) or text == '':
            return 0.0, 0.0
            
        try:
            result = self.finbert(str(text))[0]
            
            # Convert to numerical score: positive=1, negative=-1, neutral=0
            if result['label'] == 'positive':
                return result['score'], result['score']
            elif result['label'] == 'negative':
                return -result['score'], result['score']
            else:
                return 0.0, result['score']
                
        except Exception as e:
            print(f"Error analyzing text: {text[:50]}... Error: {e}")
            return 0.0, 0.0
    
    def process_news_sentiment(self, news_file_path, ticker_symbol):
        """
        Stage 3: Sentiment Analysis Processing
        Process news articles and perform sentiment analysis
        """
        print(f"\nProcessing sentiment analysis for {ticker_symbol}...")
        
        # Load news data
        news_df = pd.read_csv(news_file_path)
        print(f"Loaded {len(news_df)} news articles")
        
        # Handle different date column names (Date vs date)
        if 'Date' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['Date'], errors='coerce')
        elif 'date' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
        else:
            print("Error: No date column found in news data")
            return None
        
        # Clean and prepare data
        news_df['headline_clean'] = news_df['headline'].fillna('').astype(str)
        
        # Remove rows with invalid dates
        news_df = news_df.dropna(subset=['date'])
        
        # Perform sentiment analysis on each headline
        print("Analyzing sentiment for each headline...")
        sentiment_results = []
        
        for idx, headline in enumerate(news_df['headline_clean']):
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(news_df)} headlines...")
                
            score, confidence = self.analyze_sentiment(headline)
            sentiment_results.append({
                'sentiment_score': score,
                'sentiment_confidence': confidence
            })
        
        # Add sentiment results to dataframe
        sentiment_df = pd.DataFrame(sentiment_results)
        news_df = pd.concat([news_df, sentiment_df], axis=1)
        
        # Daily aggregation with confidence weighting and outlier handling
        daily_sentiment = self.aggregate_daily_sentiment(news_df)
        
        return daily_sentiment
    
    def aggregate_daily_sentiment(self, news_df):
        """
        Aggregate sentiment scores to daily level with confidence weighting
        """
        print("Aggregating sentiment scores to daily level...")
        
        # Remove extreme outliers (>2 standard deviations)
        sentiment_mean = news_df['sentiment_score'].mean()
        sentiment_std = news_df['sentiment_score'].std()
        lower_bound = sentiment_mean - 2 * sentiment_std
        upper_bound = sentiment_mean + 2 * sentiment_std
        
        news_df['sentiment_score_capped'] = np.clip(
            news_df['sentiment_score'], lower_bound, upper_bound
        )
        
        # Daily aggregation with confidence weighting
        daily_agg = news_df.groupby('date').agg({
            'sentiment_score_capped': [
                'mean',  # Simple average
                lambda x: np.average(x, weights=news_df.loc[x.index, 'sentiment_confidence'])  # Confidence-weighted average
            ],
            'sentiment_confidence': ['mean', 'count'],
            'headline': 'count'
        }).round(4)
        
        # Flatten column names
        daily_agg.columns = [
            'sentiment_mean', 'sentiment_weighted', 
            'avg_confidence', 'confidence_count', 'article_count'
        ]
        
        daily_agg = daily_agg.reset_index()
        
        # Use confidence-weighted sentiment as primary score
        daily_agg['sentiment_score'] = daily_agg['sentiment_weighted']
        daily_agg['daily_confidence'] = daily_agg['avg_confidence'] * np.log1p(daily_agg['article_count'])
        
        return daily_agg
    
    def handle_missing_data(self, financial_df, daily_sentiment, decay_rate=0.9):
        """
        Stage 4: Missing Data and Unbalanced Coverage Handling
        """
        print("Handling missing sentiment data...")
        
        # Standardize date column names and formats
        date_columns = [col for col in financial_df.columns.str.lower() if 'date' in col]
        if date_columns:
            # Use the first date column found
            date_col = [col for col in financial_df.columns if col.lower() in date_columns][0]
            if date_col != 'Date':
                financial_df['Date'] = financial_df[date_col]
        elif 'Date' not in financial_df.columns:
            # If no date column found, create one from index
            print("Warning: No date column found, using index as dates")
            financial_df['Date'] = pd.date_range(start='2015-01-01', periods=len(financial_df), freq='D')
        
        # Convert to datetime with error handling
        financial_df['Date'] = pd.to_datetime(financial_df['Date'], errors='coerce')
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'], errors='coerce')
        
        # Remove any rows with invalid dates
        financial_df = financial_df.dropna(subset=['Date'])
        daily_sentiment = daily_sentiment.dropna(subset=['date'])
        
        # Create date strings for merging (avoid datetime issues)
        financial_df = financial_df.copy()
        daily_sentiment = daily_sentiment.copy()
        
        # Convert datetime to string format for consistent merging
        financial_df['date_key'] = financial_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
        daily_sentiment['date_key'] = daily_sentiment['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
        
        # Reset indices to avoid merge conflicts
        financial_df = financial_df.reset_index(drop=True)
        daily_sentiment = daily_sentiment.reset_index(drop=True)
        
        # Merge financial and sentiment data using string dates
        merged_df = financial_df.merge(daily_sentiment, 
                                     left_on='date_key', right_on='date_key', 
                                     how='left', suffixes=('', '_sentiment'))
        
        # Sort by date
        merged_df = merged_df.sort_values('Date').reset_index(drop=True)
        
        # Forward fill with exponential decay
        merged_df['sentiment_filled'] = merged_df['sentiment_score'].copy()
        merged_df['confidence_filled'] = merged_df['daily_confidence'].copy()
        merged_df['days_since_news'] = 0
        
        last_valid_sentiment = 0.0
        last_valid_confidence = 0.0
        days_since_last = 0
        
        for i in range(len(merged_df)):
            if pd.notna(merged_df.loc[i, 'sentiment_score']):
                # Fresh news available
                last_valid_sentiment = merged_df.loc[i, 'sentiment_score']
                last_valid_confidence = merged_df.loc[i, 'daily_confidence'] if pd.notna(merged_df.loc[i, 'daily_confidence']) else 0.0
                days_since_last = 0
            else:
                # No news, apply decay
                days_since_last += 1
                decayed_sentiment = last_valid_sentiment * (decay_rate ** days_since_last)
                decayed_confidence = last_valid_confidence * (decay_rate ** days_since_last)
                
                merged_df.loc[i, 'sentiment_filled'] = decayed_sentiment
                merged_df.loc[i, 'confidence_filled'] = decayed_confidence
            
            merged_df.loc[i, 'days_since_news'] = days_since_last
        
        # Calculate additional metadata features
        total_days = len(merged_df)
        days_with_news = merged_df['sentiment_score'].notna().sum()
        coverage_quality = days_with_news / total_days if total_days > 0 else 0
        
        merged_df['coverage_quality'] = coverage_quality
        merged_df['article_density_20d'] = merged_df['article_count'].fillna(0).rolling(20, min_periods=1).mean()
        
        return merged_df
    
    def normalize_features(self, df):
        """
        Stage 5: Feature Normalization
        """
        print("Normalizing features...")
        
        # Rolling Z-Score for financial features (252-day window)
        financial_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Price_Change', 'Price_Change_5d', 'Price_Change_10d',
            'Volatility_10d', 'Volatility_20d'
        ]
        
        for feature in financial_features:
            if feature in df.columns:
                rolling_mean = df[feature].rolling(252, min_periods=10).mean()
                rolling_std = df[feature].rolling(252, min_periods=10).std()
                df[f'{feature}_normalized'] = (df[feature] - rolling_mean) / rolling_std
        
        # Min-Max scaling for technical indicators (already bounded)
        technical_features = {
            'RSI': 100,  # RSI is 0-100
            'BB_Position': 1,  # Already 0-1
            'Volume_Ratio': None,  # Will cap at 3x average
            'Close_to_High': 1,  # Already 0-1
            'Close_to_Low': 1   # Already 0-1
        }
        
        for feature, max_val in technical_features.items():
            if feature in df.columns:
                if max_val:
                    df[f'{feature}_normalized'] = df[feature] / max_val
                else:  # Handle Volume_Ratio special case
                    df[f'{feature}_normalized'] = np.clip(df[feature], 0, 3) / 3
        
        # Sentiment features are already bounded [-1, 1] from FinBERT
        # Just ensure they're in the right range
        df['sentiment_normalized'] = np.clip(df['sentiment_filled'], -1, 1)
        df['confidence_normalized'] = np.clip(df['confidence_filled'], 0, 1)
        
        return df
    
    def create_final_features(self, df, ticker_symbol):
        """
        Create comprehensive feature set for DRL training
        """
        print("Creating comprehensive feature set...")
        
        # Calculate returns if not present
        if 'return_1d' not in df.columns:
            df['return_1d'] = df['Close'].pct_change()
            
        # Calculate additional features from available data
        if 'portfolio_volatility' not in df.columns:
            df['portfolio_volatility'] = df.get('Volatility_20d', df['Close'].pct_change().rolling(20, min_periods=1).std()).fillna(0)
        
        # Create comprehensive feature dictionary
        final_features = {
            # Identifiers
            'Date': df['Date'],
            'ticker': ticker_symbol,
            
            # Core financial features
            'return_1d': df['return_1d'].fillna(0),
            'return_5d': df.get('Price_Change_5d', df['Close'].pct_change(5)).fillna(0),
            'return_10d': df.get('Price_Change_10d', df['Close'].pct_change(10)).fillna(0),
            
            # Technical indicators - Momentum
            'rsi_normalized': (df.get('RSI', 50) / 100).fillna(0.5),
            'macd_normalized': ((df.get('MACD', 0) - df.get('MACD', 0).rolling(252, min_periods=10).mean()) / 
                               (df.get('MACD', 1).rolling(252, min_periods=10).std().replace(0, 1))).fillna(0),
            'macd_signal_normalized': ((df.get('MACD_Signal', 0) - df.get('MACD_Signal', 0).rolling(252, min_periods=10).mean()) / 
                                     (df.get('MACD_Signal', 1).rolling(252, min_periods=10).std().replace(0, 1))).fillna(0),
            'macd_histogram_normalized': ((df.get('MACD_Histogram', 0) - df.get('MACD_Histogram', 0).rolling(252, min_periods=10).mean()) / 
                                        (df.get('MACD_Histogram', 1).rolling(252, min_periods=10).std().replace(0, 1))).fillna(0),
            
            # Technical indicators - Volatility
            'bb_position': df.get('BB_Position', 0.5).fillna(0.5),
            'bb_width_normalized': (df.get('BB_Width', 0) / df.get('BB_Width', 1).rolling(252, min_periods=10).max().replace(0, 1)).fillna(0),
            'volatility_10d_normalized': (df.get('Volatility_10d', 0) / df.get('Volatility_10d', 1).rolling(252, min_periods=10).max().replace(0, 1)).fillna(0),
            'volatility_20d_normalized': (df.get('Volatility_20d', 0) / df.get('Volatility_20d', 1).rolling(252, min_periods=10).max().replace(0, 1)).fillna(0),
            
            # Technical indicators - Volume
            'volume_ratio_normalized': np.clip(df.get('Volume_Ratio', 1), 0, 3).fillna(1) / 3,
            'volume_normalized': ((df.get('Volume', 0) - df.get('Volume', 0).rolling(252, min_periods=10).mean()) / 
                                (df.get('Volume', 1).rolling(252, min_periods=10).std().replace(0, 1))).fillna(0),
            
            # Technical indicators - Price action
            'high_low_ratio_normalized': (df.get('High_Low_Ratio', 1) / df.get('High_Low_Ratio', 1).rolling(252, min_periods=10).max().replace(0, 1)).fillna(0),
            'close_to_high': df.get('Close_to_High', 0.5).fillna(0.5),
            'close_to_low': df.get('Close_to_Low', 0.5).fillna(0.5),
            
            # Moving average indicators
            'price_to_ma5': ((df.get('Close', 0) / df.get('MA_5', df.get('Close', 1)).replace(0, 1)) - 1).fillna(0),
            'price_to_ma20': ((df.get('Close', 0) / df.get('MA_20', df.get('Close', 1)).replace(0, 1)) - 1).fillna(0),
            'price_to_ma50': ((df.get('Close', 0) / df.get('MA_50', df.get('Close', 1)).replace(0, 1)) - 1).fillna(0),
            'ma20_slope': df.get('MA_20', 0).pct_change(5).fillna(0),
            
            # Trend indicators
            'trend_5d': df.get('Trend_5d', 0).fillna(0),
            'trend_20d': df.get('Trend_20d', 0).fillna(0),
            
            # Support and resistance
            'price_vs_support': df.get('Price_vs_Support', 0).fillna(0),
            'price_vs_resistance': df.get('Price_vs_Resistance', 0).fillna(0),
            
            # Enhanced sentiment features
            'sentiment_score': df['sentiment_normalized'],
            'sentiment_confidence': df['confidence_normalized'],
            'sentiment_momentum_3d': df['sentiment_normalized'].rolling(3, min_periods=1).mean(),
            'sentiment_momentum_7d': df['sentiment_normalized'].rolling(7, min_periods=1).mean(),
            'sentiment_volatility_5d': df['sentiment_normalized'].rolling(5, min_periods=1).std().fillna(0),
            'sentiment_trend': df['sentiment_normalized'].diff(5).fillna(0),
            'days_since_news': df['days_since_news'],
            'coverage_quality': df['coverage_quality'],
            'article_density_7d': df.get('article_count', 0).rolling(7, min_periods=1).mean(),
            'article_density_20d': df.get('article_density_20d', 0),
            'news_frequency_weekly': df.get('article_count', 0).rolling(5, min_periods=1).sum(),
            
            # Portfolio state features (placeholders for DRL environment)
            'current_portfolio_weight': 0.0,
            'portfolio_volatility': df['portfolio_volatility'],
            'cash_position': 1.0,
            'portfolio_return_1d': 0.0,
            'portfolio_sharpe_20d': 0.0,
            
            # Raw values for debugging and analysis
            'sentiment_raw': df.get('sentiment_score', 0).fillna(0),
            'article_count_daily': df.get('article_count', 0).fillna(0),
            'close_price': df.get('Close', 0),
            'volume_raw': df.get('Volume', 0),
        }
        
        # Create DataFrame and ensure all numeric
        final_df = pd.DataFrame(final_features)
        final_df['ticker'] = ticker_symbol
        
        # Fill any remaining NaN values with 0
        numeric_columns = final_df.select_dtypes(include=[np.number]).columns
        final_df[numeric_columns] = final_df[numeric_columns].fillna(0)
        
        print(f"Created feature set with {len(final_df.columns)} columns")
        return final_df
    
    def process_ticker(self, news_file_path, financial_file_path, ticker_symbol, 
                      output_dir='./', decay_rate=0.9):
        """
        Complete pipeline for processing a single ticker
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING TICKER: {ticker_symbol}")
        print(f"{'='*60}")
        
        try:
            # Stage 3: Sentiment Analysis
            daily_sentiment = self.process_news_sentiment(news_file_path, ticker_symbol)
            
            # Load financial data
            print(f"Loading financial data from {financial_file_path}...")
            financial_df = pd.read_csv(financial_file_path)
            print(f"Loaded financial data: {len(financial_df)} rows, {len(financial_df.columns)} columns")
            
            # Stage 4: Handle missing data
            merged_df = self.handle_missing_data(financial_df, daily_sentiment, decay_rate)
            
            # Stage 5: Normalize features
            normalized_df = self.normalize_features(merged_df)
            
            # Create final feature set
            final_df = self.create_final_features(normalized_df, ticker_symbol)
            
            # Save results
            output_file = os.path.join(output_dir, f'{ticker_symbol}_complete_processed.csv')
            final_df.to_csv(output_file, index=False)
            
            # Print summary statistics
            self.print_processing_summary(final_df, ticker_symbol)
            
            print(f"\n✅ Processing complete! Saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error processing {ticker_symbol}: {str(e)}")
            return None
    
    def print_processing_summary(self, df, ticker_symbol):
        """Print summary of processing results"""
        print(f"\n📊 PROCESSING SUMMARY FOR {ticker_symbol}:")
        print(f"   Total trading days: {len(df)}")
        print(f"   Days with news: {df['sentiment_raw'].notna().sum()}")
        print(f"   Coverage ratio: {df['sentiment_raw'].notna().mean():.1%}")
        print(f"   Average daily articles: {df['article_count_daily'].mean():.1f}")
        print(f"   Sentiment range: [{df['sentiment_score'].min():.3f}, {df['sentiment_score'].max():.3f}]")
        print(f"   Average confidence: {df['sentiment_confidence'].mean():.3f}")
        print(f"   Max days since news: {df['days_since_news'].max()}")


def main():
    """
    Main execution function
    Easily modify file paths here for different tickers
    """
    
    # Initialize pipeline
    pipeline = SentimentAnalysisPipeline()
    
    # Configuration - MODIFY THESE PATHS AS NEEDED
    configs = [
        {
            'ticker': 'XOM',
            'news_file': '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/temp_optimized_1500_XOM_2015_2020.csv',
            'financial_file': '/Users/antoinerithychesnay/Library/Mobile Documents/com~apple~CloudDocs/Desktop/dissertation/financial_data/XOM_2015-01-01_to_2020-01-01_with_indicators_20250831_104808.csv',
            'output_dir': './'
        },
        # Add more ticker configurations here:
        # {
        #     'ticker': 'AMZN',
        #     'news_file': 'path/to/AMZN_news.csv',
        #     'financial_file': 'path/to/AMZN_financial.csv',
        #     'output_dir': './'
        # },
    ]
    
    # Process each ticker
    processed_files = []
    for config in configs:
        result = pipeline.process_ticker(
            news_file_path=config['news_file'],
            financial_file_path=config['financial_file'],
            ticker_symbol=config['ticker'],
            output_dir=config['output_dir'],
            decay_rate=0.9  # Adjust decay rate as needed
        )
        
        if result:
            processed_files.append(result)
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"Processed files: {processed_files}")
    
    return processed_files


if __name__ == "__main__":
    # Run the complete pipeline
    processed_files = main()