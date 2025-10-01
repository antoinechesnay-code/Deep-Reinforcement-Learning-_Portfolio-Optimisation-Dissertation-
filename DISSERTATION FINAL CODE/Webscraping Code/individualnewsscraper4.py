import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime, timedelta
import csv
import logging
from typing import List, Dict, Optional
import random
from urllib.parse import quote, urljoin

class BalancedNewsScraper:
    """
    Balanced scraper targeting ~1200 articles with optimized speed
    Middle ground between fast (750) and comprehensive (1900) versions
    """
    
    def __init__(self, tickers: List[str], output_file: str = "historical_financial_news.csv"):
        self.tickers = [ticker.upper() for ticker in tickers]
        self.output_file = output_file
        self.wayback_base = "https://web.archive.org/web"
        self.session = requests.Session()
        
        # Set up headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('balanced_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Expanded search terms for all tickers
        self.search_terms = {
            'AAPL': ['Apple', 'AAPL', 'iPhone', 'Tim Cook', 'iPad'],
            'AMZN': ['Amazon', 'AMZN', 'Bezos', 'AWS', 'Prime'],
            'AMT': ['American Tower', 'AMT', 'REIT', 'cell tower'],
            'CAT': ['Caterpillar', 'CAT', 'construction equipment'],
            'JPM': ['JPMorgan', 'JPM', 'Chase', 'Jamie Dimon'],
            'LIN': ['Linde', 'LIN', 'industrial gases'],
            'NEE': ['NextEra', 'NEE', 'renewable energy', 'Florida Power'],
            'PG': ['Procter & Gamble', 'P&G', 'PG', 'consumer goods'],
            'UNH': ['UnitedHealth', 'UNH', 'Optum', 'health insurance'],
            'XOM': ['Exxon', 'XOM', 'ExxonMobil', 'oil', 'energy'],
            'GOOGL': ['Google', 'Alphabet', 'GOOGL', 'GOOG'],
            'MSFT': ['Microsoft', 'MSFT', 'Azure', 'Windows'],
            'TSLA': ['Tesla', 'TSLA', 'Musk', 'Model'],
            'META': ['Facebook', 'Meta', 'FB', 'Instagram'],
            'NVDA': ['NVIDIA', 'NVDA', 'GPU'],
        }
        
        # BALANCED: 6 sites (middle ground between 4 and 8)
        self.priority_sites = {
            'MarketWatch': {
                'base_url': 'https://www.marketwatch.com',
                'ticker_patterns': [
                    '/investing/stock/{ticker}',
                    '/quote/stock/us/{ticker}',
                ],
                'general_patterns': ['/story/', '/markets/', '/earnings/'],
                'selectors': ['h1', 'h2', 'h3', 'a[href*="/story/"]', '.headline'],
                'priority': 'high',
                'snapshots_per_url': 10  # Reduced from 12
            },
            'Yahoo Finance': {
                'base_url': 'https://finance.yahoo.com',
                'ticker_patterns': [
                    '/quote/{ticker}/news',
                    '/quote/{ticker}/',
                ],
                'general_patterns': ['/news/', '/stock-market-news/'],
                'selectors': ['h1', 'h2', 'h3', 'a[href*="/news/"]', '.title'],
                'priority': 'high',
                'snapshots_per_url': 10
            },
            'CNBC': {
                'base_url': 'https://www.cnbc.com',
                'ticker_patterns': [
                    '/quotes/{ticker}',
                ],
                'general_patterns': ['/stocks/', '/investing/', '/earnings/'],
                'selectors': ['h1', 'h2', 'h3', 'a[href*="/stocks/"]', '.headline'],
                'priority': 'high',
                'snapshots_per_url': 8
            },
            'Seeking Alpha': {
                'base_url': 'https://seekingalpha.com',
                'ticker_patterns': [
                    '/symbol/{ticker}',
                    '/symbol/{ticker}/news',
                ],
                'general_patterns': ['/news/', '/earnings/'],
                'selectors': ['h1', 'h2', 'h3', 'a[data-test-id*="post-title"]', '.title'],
                'priority': 'medium',
                'snapshots_per_url': 8
            },
            'The Motley Fool': {
                'base_url': 'https://www.fool.com',
                'ticker_patterns': [
                    '/quote/nasdaq/{ticker}/',
                ],
                'general_patterns': ['/investing/', '/earnings/'],
                'selectors': ['h1', 'h2', 'h3', '.headline', '.article-title'],
                'priority': 'medium',
                'snapshots_per_url': 6
            },
            'Benzinga': {
                'base_url': 'https://www.benzinga.com',
                'ticker_patterns': [
                    '/quote/{ticker}',
                    '/stock/{ticker}',
                ],
                'general_patterns': ['/news/', '/earnings/'],
                'selectors': ['h1', 'h2', 'h3', '.title', '.post-title'],
                'priority': 'medium',
                'snapshots_per_url': 6
            }
        }
        
        self.results = []
        
    def wait_between_requests(self, min_delay: float = 1.5, max_delay: float = 2.5):
        """Balanced delays - faster than scraper3 but safer than scraper2"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def get_balanced_snapshots(self, url: str, year: int, limit: int = 10) -> List[str]:
        """Balanced snapshot collection - more than scraper2, less than scraper3"""
        snapshots = []
        
        # Use quarterly approach but with more snapshots per quarter
        quarters = [
            (f"{year}0101", f"{year}0331"),  # Q1
            (f"{year}0401", f"{year}0630"),  # Q2  
            (f"{year}0701", f"{year}0930"),  # Q3
            (f"{year}1001", f"{year}1231"),  # Q4
        ]
        
        cdx_url = f"https://web.archive.org/cdx/search/cdx"
        
        for start_date, end_date in quarters:
            params = {
                'url': url,
                'from': start_date,
                'to': end_date,
                'output': 'json',
                'limit': 20,  # Increased from 15 (scraper2) but less than full monthly (scraper3)
                'filter': 'statuscode:200'
            }
            
            try:
                response = self.session.get(cdx_url, params=params, timeout=12)
                response.raise_for_status()
                
                data = response.json()
                if len(data) > 1:
                    # Take every 2nd snapshot for better coverage than scraper2
                    for i, row in enumerate(data[1:]):
                        if i % 2 == 0 and len(snapshots) < limit:
                            timestamp = row[1]
                            original_url = row[2]
                            wayback_url = f"{self.wayback_base}/{timestamp}/{original_url}"
                            snapshots.append(wayback_url)
                            
                time.sleep(0.8)  # Balanced delay between quarters
                        
            except Exception as e:
                self.logger.warning(f"Error fetching snapshots for {url}: {str(e)}")
                continue
                
        self.logger.info(f"Found {len(snapshots)} snapshots for {url} in {year}")
        return snapshots[:limit]
    
    def is_relevant_headline(self, headline: str, ticker: str) -> tuple[bool, str, float]:
        """Relevance check with scoring"""
        headline_upper = headline.upper()
        search_terms = self.search_terms.get(ticker.upper(), [ticker.upper()])
        
        # Must contain ticker or company name
        matched_term = None
        for term in search_terms:
            if term.upper() in headline_upper:
                matched_term = term
                break
                
        if not matched_term:
            return False, "", 0.0
            
        # Filter out irrelevant content
        irrelevant_keywords = [
            'HOROSCOPE', 'WEATHER', 'SPORTS SCORES', 'LOTTERY', 
            'RECIPE', 'CELEBRITY', 'FASHION', 'TRAVEL TIPS',
            'DATING', 'FITNESS', 'DIET', 'REAL ESTATE TIPS',
            'CAR REVIEW', 'MOVIE REVIEW', 'TV SHOW'
        ]
        
        for keyword in irrelevant_keywords:
            if keyword in headline_upper:
                return False, "", 0.0
        
        # Calculate quality score
        score = 2.0  # Base score
        
        # Financial keywords boost
        financial_keywords = [
            'EARNINGS', 'STOCK', 'SHARES', 'REVENUE', 'PROFIT', 'LOSS',
            'MARKET', 'TRADING', 'INVESTMENT', 'ANALYST', 'PRICE',
            'QUARTERLY', 'ANNUAL', 'SEC', 'FILING', 'ACQUISITION',
            'MERGER', 'IPO', 'DIVIDEND', 'GUIDANCE', 'FORECAST'
        ]
        
        for keyword in financial_keywords:
            if keyword in headline_upper:
                score += 0.5
                break
        
        # Length check
        if len(headline) < 20 or len(headline) > 200:
            return False, "", 0.0
            
        return True, matched_term, score
    
    def scrape_balanced_snapshot(self, wayback_url: str, ticker: str, site_info: dict, 
                                year: int, site_name: str) -> List[Dict]:
        """Balanced scraping approach"""
        articles = []
        
        try:
            self.wait_between_requests(1.5, 2.5)
            
            response = self.session.get(wayback_url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            if len(response.content) < 500:
                return articles
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            selectors = site_info.get('selectors', ['h1', 'h2', 'h3'])
            found_articles = set()
            
            for selector in selectors:
                try:
                    elements = soup.select(selector)[:55]  # Middle ground between 50 and 60
                    
                    for element in elements:
                        headline_text = element.get_text(strip=True)
                        
                        is_relevant, matched_term, score = self.is_relevant_headline(headline_text, ticker)
                        
                        if is_relevant and score > 0:
                            # Get article URL
                            article_url = ""
                            if element.name == 'a' and element.get('href'):
                                href = element.get('href')
                                if href.startswith('http'):
                                    article_url = href
                                elif href.startswith('/'):
                                    domain = self.extract_original_domain(wayback_url)
                                    if domain:
                                        article_url = domain + href
                            
                            date_str = self.extract_simple_date(wayback_url, year)
                            
                            article_key = headline_text.lower().strip()
                            if article_key not in found_articles:
                                found_articles.add(article_key)
                                
                                articles.append({
                                    'headline': headline_text,
                                    'date': date_str,
                                    'url': article_url or wayback_url,
                                    'relevant_tickers': [ticker],
                                    'summary': '',
                                    'source': site_name,
                                    'wayback_url': wayback_url,
                                    'matched_term': matched_term,
                                    'relevance_score': score
                                })
                                
                except Exception as e:
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error scraping {wayback_url}: {str(e)}")
            
        return articles
    
    def extract_simple_date(self, wayback_url: str, year: int) -> str:
        """Simple date extraction from Wayback URL"""
        try:
            parts = wayback_url.split('/')
            for part in parts:
                if len(part) >= 8 and part.isdigit():
                    timestamp = part[:8]
                    if len(timestamp) == 8:
                        year_part = timestamp[:4]
                        month_part = timestamp[4:6]
                        day_part = timestamp[6:8]
                        return f"{month_part}/{day_part}/{year_part}"
        except:
            pass
        return f"01/01/{year}"
    
    def extract_original_domain(self, wayback_url: str) -> Optional[str]:
        """Extract original domain from Wayback URL"""
        try:
            parts = wayback_url.split('/')
            if len(parts) >= 7:
                protocol = parts[5] if parts[5] in ['http:', 'https:'] else 'https:'
                domain = parts[6] if parts[5] in ['http:', 'https:'] else parts[5]
                return f"{protocol}//{domain}"
        except:
            pass
        return None
    
    def search_ticker_balanced(self, ticker: str, year: int) -> List[Dict]:
        """Balanced search strategy"""
        articles = []
        
        self.logger.info(f"🎯 Balanced search for {ticker} in {year}")
        
        for site_name, site_info in self.priority_sites.items():
            self.logger.info(f"Searching {site_name}...")
            
            # Build target URLs with ticker replacement
            target_urls = []
            
            # Ticker-specific URLs
            if 'ticker_patterns' in site_info:
                for pattern in site_info['ticker_patterns']:
                    # Replace {ticker} placeholder with actual ticker
                    url = site_info['base_url'] + pattern.replace('{ticker}', ticker.lower())
                    target_urls.append((url, 'high'))
            
            # Add general patterns (fewer than scraper3)
            for pattern in site_info['general_patterns'][:2]:  # Only first 2 general patterns
                url = site_info['base_url'] + pattern
                target_urls.append((url, 'medium'))
                
            # Process URLs
            for url, priority in target_urls:
                try:
                    snapshots_limit = site_info.get('snapshots_per_url', 8)
                    snapshots = self.get_balanced_snapshots(url, year, snapshots_limit)
                    
                    for snapshot_url in snapshots:
                        try:
                            snapshot_articles = self.scrape_balanced_snapshot(
                                snapshot_url, ticker, site_info, year, site_name
                            )
                            articles.extend(snapshot_articles)
                            
                        except Exception as e:
                            continue
                            
                    # Balanced delays
                    time.sleep(random.uniform(2.5, 4.0))
                    
                except Exception as e:
                    self.logger.warning(f"URL error {url}: {str(e)}")
                    continue
            
            # Moderate delay between sites
            time.sleep(random.uniform(5.0, 8.0))
            
        self.logger.info(f"Found {len(articles)} articles for {ticker} in {year}")
        return articles
    
    def deduplicate_and_rank(self, articles: List[Dict]) -> List[Dict]:
        """Simple and fast deduplication"""
        if not articles:
            return articles
            
        # Remove exact duplicates
        seen_headlines = set()
        unique_articles = []
        
        for article in articles:
            headline_key = article['headline'].lower().strip()
            if headline_key not in seen_headlines:
                seen_headlines.add(headline_key)
                unique_articles.append(article)
        
        # Sort by relevance score
        unique_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return unique_articles
    
    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            self.logger.warning("No results to save")
            return
            
        # Deduplicate and rank
        self.results = self.deduplicate_and_rank(self.results)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Clean data
        df['relevant_tickers'] = df['relevant_tickers'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        df['headline'] = df['headline'].str.strip()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Sort by relevance score and date
        df = df.sort_values(['relevance_score', 'date'], ascending=[False, False])
        
        # Save to CSV
        df.to_csv(self.output_file, index=False)
        self.logger.info(f"Saved {len(df)} articles to {self.output_file}")
        
        # Print summary
        print(f"\n=== BALANCED SCRAPING SUMMARY ===")
        print(f"Total articles: {len(df)}")
        
        if not df.empty:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Average relevance score: {df['relevance_score'].mean():.2f}")
            print(f"Sources: {df['source'].value_counts().to_dict()}")
            
            if 'matched_term' in df.columns:
                print(f"Top matched terms: {df['matched_term'].value_counts().head(3).to_dict()}")
                
            print(f"\nTop 5 articles by relevance:")
            for i, (_, row) in enumerate(df.head().iterrows()):
                print(f"  {i+1}. {row['headline'][:60]}... (Score: {row['relevance_score']:.1f})")
                
        print(f"Results saved to: {self.output_file}")
    
    def run_balanced_scrape(self, start_year: int = 2015, end_year: int = 2020):
        """Run balanced scraping for ~1200 articles"""
        self.logger.info(f"🚀 Starting BALANCED scrape for: {self.tickers}")
        self.logger.info(f"Year range: {start_year} to {end_year}")
        self.logger.info(f"Target: ~1200 articles in 60-75 minutes")
        
        all_articles = []
        total_start_time = time.time()
        
        for year in range(start_year, end_year + 1):
            year_start_time = time.time()
            self.logger.info(f"=== Scraping {year} ===")
            
            for ticker in self.tickers:
                try:
                    articles = self.search_ticker_balanced(ticker, year)
                    all_articles.extend(articles)
                    
                    self.logger.info(f"✔ {ticker} {year}: {len(articles)} articles")
                    
                    # Balanced delay between tickers
                    time.sleep(random.uniform(6.0, 9.0))
                    
                except Exception as e:
                    self.logger.error(f"Error processing {ticker} {year}: {str(e)}")
                    continue
                    
            year_time = time.time() - year_start_time
            self.logger.info(f"Year {year} completed in {year_time:.1f} seconds")
            
            # Save intermediate results every 2 years
            if year % 2 == 0:
                self.results = all_articles
                temp_filename = f"temp_{self.output_file}"
                temp_df = pd.DataFrame(self.deduplicate_and_rank(all_articles))
                if not temp_df.empty:
                    temp_df.to_csv(temp_filename, index=False)
                    self.logger.info(f"Intermediate save: {len(temp_df)} articles")
            
            # Balanced delay between years
            if year < end_year:
                time.sleep(random.uniform(8.0, 12.0))
        
        total_time = time.time() - total_start_time
        self.logger.info(f"Total scraping time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Store and save results
        self.results = all_articles
        self.save_results()
        
        return self.results


def main():
    """Main function - balanced scraping for multiple tickers"""
    
    # You can change this to any ticker from your list
    # Options: 'AMT', 'AMZN', 'CAT', 'JPM', 'LIN', 'NEE', 'PG', 'UNH', 'XOM'
    target_tickers = ['JPM']  # Change as needed
    
    print(f"⚖️ BALANCED MODE: {target_tickers[0]}")
    print("Optimizations:")
    print("- 6 news sources (middle ground between 4 and 8)")
    print("- Balanced snapshot sampling (10 per URL vs 15 vs 12)")
    print("- Moderate delays (1.5-2.5s vs 1.5-3.0s vs 2.0-4.0s)")
    print("- Simplified processing for speed")
    print("- Target: ~1200 articles in 60-75 minutes")
    print("- Balance: Speed + Volume\n")
    
    # Initialize balanced scraper
    scraper = BalancedNewsScraper(
        tickers=target_tickers,
        output_file=f"balanced_{target_tickers[0]}_2015_2020.csv"
    )
    
    # Run balanced scraping
    try:
        start_time = time.time()
        results = scraper.run_balanced_scrape(
            start_year=2015,
            end_year=2020
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n=== BALANCED SCRAPING COMPLETED ===")
        print(f"Time taken: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Articles found: {len(results)}")
        print(f"Rate: {len(results)/duration*60:.1f} articles/minute")
        print("\nOptimal balance achieved! ⚖️")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        scraper.save_results()
        
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        scraper.save_results()


if __name__ == "__main__":
    main()