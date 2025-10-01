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
# this one took about an hour to run and got 750 
class FastTargetedNewsScraper:
    """
    Fast, targeted scraper optimized for high-profile tickers
    Focuses on quality and relevance over exhaustive coverage
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
                logging.FileHandler('fast_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Focused search terms - only the most relevant ones
        self.search_terms = {
            'AAPL': ['Apple', 'AAPL', 'iPhone', 'Tim Cook'],
            'AMZN': ['Amazon', 'AMZN', 'Bezos', 'AWS'],
            'GOOGL': ['Google', 'Alphabet', 'GOOGL'],
            'MSFT': ['Microsoft', 'MSFT'],
            'TSLA': ['Tesla', 'TSLA', 'Musk'],
            'META': ['Facebook', 'Meta', 'FB'],
            'NVDA': ['NVIDIA', 'NVDA'],
            'UNH': ['UnitedHealth', 'UNH'],
        }
        
        # Only the most reliable, fast-loading news sources
        self.priority_sites = {
            'MarketWatch': {
                'base_url': 'https://www.marketwatch.com',
                'ticker_patterns': [
                    f'/investing/stock/{self.tickers[0].lower()}',
                    f'/quote/stock/us/{self.tickers[0].lower()}',
                ],
                'general_patterns': ['/story/', '/markets/', '/earnings/'],
                'selectors': ['h1', 'h2', 'h3', 'a[href*="/story/"]', '.headline']
            },
            'Yahoo Finance': {
                'base_url': 'https://finance.yahoo.com',
                'ticker_patterns': [
                    f'/quote/{self.tickers[0]}/news',
                    f'/quote/{self.tickers[0]}/',
                ],
                'general_patterns': ['/news/', '/stock-market-news/'],
                'selectors': ['h1', 'h2', 'h3', 'a[href*="/news/"]', '.title']
            },
            'CNBC': {
                'base_url': 'https://www.cnbc.com',
                'ticker_patterns': [
                    f'/quotes/{self.tickers[0]}',
                ],
                'general_patterns': ['/stocks/', '/investing/', '/earnings/'],
                'selectors': ['h1', 'h2', 'h3', 'a[href*="/stocks/"]', '.headline']
            },
            'Seeking Alpha': {
                'base_url': 'https://seekingalpha.com',
                'ticker_patterns': [
                    f'/symbol/{self.tickers[0]}',
                    f'/symbol/{self.tickers[0]}/news',
                ],
                'general_patterns': ['/news/', '/earnings/'],
                'selectors': ['h1', 'h2', 'h3', 'a[data-test-id*="post-title"]', '.title']
            }
        }
        
        self.results = []
        
    def wait_between_requests(self, min_delay: float = 1.5, max_delay: float = 3.0):
        """Shorter delays for faster execution"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def get_strategic_snapshots(self, url: str, year: int) -> List[str]:
        """Get strategic snapshots - quarterly sampling for better coverage with fewer requests"""
        snapshots = []
        
        # Get snapshots for each quarter to ensure temporal coverage
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
                'limit': 15,  # Reduced from 50+ to 15 per quarter
                'filter': 'statuscode:200'
            }
            
            try:
                response = self.session.get(cdx_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if len(data) > 1:
                    # Take every 3rd snapshot to spread coverage
                    for i, row in enumerate(data[1:]):
                        if i % 3 == 0:  # Sample every 3rd
                            timestamp = row[1]
                            original_url = row[2]
                            wayback_url = f"{self.wayback_base}/{timestamp}/{original_url}"
                            snapshots.append(wayback_url)
                            
                time.sleep(0.5)  # Short delay between quarter requests
                        
            except Exception as e:
                self.logger.warning(f"Error fetching quarterly snapshots for {url}: {str(e)}")
                continue
                
        self.logger.info(f"Found {len(snapshots)} strategic snapshots for {url} in {year}")
        return snapshots
    
    def is_relevant_headline(self, headline: str, ticker: str) -> tuple[bool, str]:
        """Fast relevance check with strict criteria"""
        headline_upper = headline.upper()
        search_terms = self.search_terms.get(ticker.upper(), [ticker.upper()])
        
        # Must contain ticker or company name
        matched_term = None
        for term in search_terms:
            if term.upper() in headline_upper:
                matched_term = term
                break
                
        if not matched_term:
            return False, ""
            
        # Filter out clearly irrelevant content
        irrelevant_keywords = [
            'HOROSCOPE', 'WEATHER', 'SPORTS SCORES', 'LOTTERY', 
            'RECIPE', 'CELEBRITY', 'FASHION', 'TRAVEL TIPS',
            'DATING', 'FITNESS', 'DIET', 'REAL ESTATE TIPS',
            'CAR REVIEW', 'MOVIE REVIEW', 'TV SHOW'
        ]
        
        for keyword in irrelevant_keywords:
            if keyword in headline_upper:
                return False, ""
        
        # Boost financial/business relevance
        financial_keywords = [
            'EARNINGS', 'STOCK', 'SHARES', 'REVENUE', 'PROFIT', 'LOSS',
            'MARKET', 'TRADING', 'INVESTMENT', 'ANALYST', 'PRICE',
            'QUARTERLY', 'ANNUAL', 'SEC', 'FILING', 'ACQUISITION',
            'MERGER', 'IPO', 'DIVIDEND', 'GUIDANCE', 'FORECAST'
        ]
        
        has_financial_context = any(keyword in headline_upper for keyword in financial_keywords)
        
        # Must be reasonable length
        if len(headline) < 20 or len(headline) > 200:
            return False, ""
            
        return True, matched_term
    
    def scrape_targeted_snapshot(self, wayback_url: str, ticker: str, site_info: dict, 
                                year: int, site_name: str) -> List[Dict]:
        """Fast, targeted scraping focused on relevant content"""
        articles = []
        
        try:
            self.wait_between_requests(2.0, 4.0)
            
            response = self.session.get(wayback_url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            
            if len(response.content) < 500:
                return articles
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Use site-specific selectors for better targeting
            selectors = site_info.get('selectors', ['h1', 'h2', 'h3'])
            found_articles = set()
            
            for selector in selectors:
                try:
                    elements = soup.select(selector)[:50]  # Limit elements to check
                    
                    for element in elements:
                        headline_text = element.get_text(strip=True)
                        
                        # Quick relevance check
                        is_relevant, matched_term = self.is_relevant_headline(headline_text, ticker)
                        
                        if is_relevant:
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
                            
                            # Simple date extraction
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
                                    'relevance_score': self.calculate_relevance_score(headline_text, ticker)
                                })
                                
                                self.logger.info(f"✓ Found: {headline_text[:50]}... ({matched_term})")
                                
                except Exception as e:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error scraping {wayback_url}: {str(e)}")
            
        return articles
    
    def calculate_relevance_score(self, headline: str, ticker: str) -> float:
        """Simple relevance scoring for ranking results"""
        score = 0.0
        headline_upper = headline.upper()
        
        # Base score for containing ticker
        if ticker.upper() in headline_upper:
            score += 2.0
            
        # Bonus for financial keywords
        financial_keywords = [
            'EARNINGS', 'STOCK', 'REVENUE', 'PROFIT', 'QUARTERLY',
            'ANALYST', 'PRICE', 'SHARES', 'MARKET', 'TRADING'
        ]
        
        for keyword in financial_keywords:
            if keyword in headline_upper:
                score += 0.5
                
        # Length bonus (not too short, not too long)
        if 30 <= len(headline) <= 100:
            score += 0.3
            
        return score
    
    def extract_simple_date(self, wayback_url: str, year: int) -> str:
        """Simple date extraction from Wayback URL"""
        try:
            # Extract timestamp from Wayback URL
            parts = wayback_url.split('/')
            for part in parts:
                if len(part) >= 8 and part.isdigit():
                    timestamp = part[:8]  # YYYYMMDD
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
    
    def search_ticker_strategically(self, ticker: str, year: int) -> List[Dict]:
        """Strategic search focusing on ticker-specific and high-value pages"""
        articles = []
        
        self.logger.info(f"🎯 Strategic search for {ticker} in {year}")
        
        for site_name, site_info in self.priority_sites.items():
            self.logger.info(f"Searching {site_name}...")
            
            # Priority 1: Ticker-specific pages (most likely to have relevant content)
            target_urls = []
            
            # Add ticker-specific URLs first
            if 'ticker_patterns' in site_info and self.tickers:
                for pattern in site_info['ticker_patterns']:
                    url = site_info['base_url'] + pattern
                    target_urls.append((url, 'high'))
            
            # Add general financial pages second
            for pattern in site_info['general_patterns']:
                url = site_info['base_url'] + pattern
                target_urls.append((url, 'medium'))
                
            # Process URLs by priority
            for url, priority in target_urls:
                try:
                    snapshots = self.get_strategic_snapshots(url, year)
                    
                    # Limit snapshots based on priority
                    max_snapshots = 8 if priority == 'high' else 4
                    selected_snapshots = snapshots[:max_snapshots]
                    
                    for snapshot_url in selected_snapshots:
                        try:
                            snapshot_articles = self.scrape_targeted_snapshot(
                                snapshot_url, ticker, site_info, year, site_name
                            )
                            articles.extend(snapshot_articles)
                            
                        except Exception as e:
                            self.logger.warning(f"Snapshot error: {str(e)}")
                            continue
                            
                    # Short delay between URLs
                    time.sleep(random.uniform(2.0, 4.0))
                    
                except Exception as e:
                    self.logger.warning(f"URL error {url}: {str(e)}")
                    continue
            
            # Delay between sites
            time.sleep(random.uniform(3.0, 6.0))
            
        self.logger.info(f"Found {len(articles)} articles for {ticker} in {year}")
        return articles
    
    def deduplicate_and_rank(self, articles: List[Dict]) -> List[Dict]:
        """Fast deduplication and ranking by relevance"""
        if not articles:
            return articles
            
        # Remove exact duplicates first
        seen_headlines = set()
        unique_articles = []
        
        for article in articles:
            headline_key = article['headline'].lower().strip()
            if headline_key not in seen_headlines:
                seen_headlines.add(headline_key)
                unique_articles.append(article)
        
        # Sort by relevance score (highest first)
        unique_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return unique_articles
    
    def save_results(self):
        """Save results with quality metrics"""
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
        print(f"\n=== FAST TARGETED SCRAPING SUMMARY ===")
        print(f"Total relevant articles: {len(df)}")
        
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
    
    def run_fast_targeted_scrape(self, start_year: int = 2015, end_year: int = 2020):
        """Run fast, targeted scraping optimized for speed and relevance"""
        self.logger.info(f"🚀 Starting FAST targeted scrape for: {self.tickers}")
        self.logger.info(f"Year range: {start_year} to {end_year}")
        
        all_articles = []
        total_start_time = time.time()
        
        for year in range(start_year, end_year + 1):
            year_start_time = time.time()
            self.logger.info(f"=== Scraping {year} ===")
            
            for ticker in self.tickers:
                try:
                    articles = self.search_ticker_strategically(ticker, year)
                    all_articles.extend(articles)
                    
                    self.logger.info(f"✓ {ticker} {year}: {len(articles)} articles")
                    
                    # Short delay between tickers
                    time.sleep(random.uniform(4.0, 7.0))
                    
                except Exception as e:
                    self.logger.error(f"Error processing {ticker} {year}: {str(e)}")
                    continue
                    
            year_time = time.time() - year_start_time
            self.logger.info(f"Year {year} completed in {year_time:.1f} seconds")
            
            # Short delay between years
            if year < end_year:
                time.sleep(random.uniform(5.0, 8.0))
        
        total_time = time.time() - total_start_time
        self.logger.info(f"Total scraping time: {total_time:.1f} seconds")
        
        # Store and save results
        self.results = all_articles
        self.save_results()
        
        return self.results


def main():
    """Main function - fast targeted scraping"""
    
    target_tickers = ['AMZN']  # Change to ['AMZN'] for Amazon
    
    print(f"⚡ FAST TARGETED MODE: {target_tickers[0]}")
    print("Optimizations:")
    print("- Strategic quarterly sampling (not exhaustive)")
    print("- Strict relevance filtering")
    print("- Only 4 high-quality news sources")
    print("- Fast deduplication and ranking")
    print("- Target: 200+ relevant articles in <15 minutes")
    print("- Focus: Quality over quantity\n")
    
    # Initialize fast scraper
    scraper = FastTargetedNewsScraper(
        tickers=target_tickers,
        output_file=f"fast_targeted_{target_tickers[0]}_2015_2020.csv"
    )
    
    # Run fast targeted scraping
    try:
        start_time = time.time()
        results = scraper.run_fast_targeted_scrape(
            start_year=2015,
            end_year=2020
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n=== FAST TARGETED SCRAPING COMPLETED ===")
        print(f"Time taken: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Articles found: {len(results)}")
        print(f"Rate: {len(results)/duration*60:.1f} articles/minute")
        print("\nQuality over quantity achieved! ✨")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        scraper.save_results()
        
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        scraper.save_results()


if __name__ == "__main__":
    main()