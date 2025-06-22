# Web Scraping & ETL Pipeline

## Project Overview
This project demonstrates how to build a complete ETL (Extract, Transform, Load) pipeline using web scraping. We'll extract data from a public website, clean and transform it, and load it into a structured format.

## Project Structure
```
02_web_scraping_etl/
├── data/
│   └── raw/                 # Raw scraped data
│   └── processed/           # Processed data
├── notebooks/
│   └── etl_pipeline.ipynb  # Jupyter notebook with ETL process
├── src/
│   ├── __init__.py
│   ├── scraper.py          # Web scraping functions
│   ├── transformer.py      # Data transformation functions
│   └── loader.py           # Data loading functions
├── requirements.txt         # Project dependencies
└── README.md
```

## Setup Instructions
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p data/raw data/processed notebooks src
   ```

## Project Tasks
1. **Extract**: Scrape book information from a public website (e.g., books.toscrape.com)
2. **Transform**: Clean and structure the scraped data
3. **Load**: Save the processed data to CSV/JSON
4. **Analysis**: Perform basic analysis on the collected data
5. **Visualization**: Create visualizations of the data

## Implementation Steps

### 1. Set up the scraper (src/scraper.py)
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import json

class BookScraper:
    BASE_URL = 'http://books.toscrape.com/'
    
    def __init__(self):
        self.books = []
    
    def scrape_books(self, num_pages: int = 5) -> List[Dict]:
        """Scrape book information from multiple pages."""
        for page in range(1, num_pages + 1):
            print(f"Scraping page {page}...")
            url = f"{self.BASE_URL}/catalogue/page-{page}.html"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for book in soup.select('article.product_pod'):
                title = book.h3.a['title']
                price = book.select_one('p.price_color').text[1:]  # Remove £ symbol
                rating = book.p['class'][1]  # Get rating from class name
                available = 'in stock' in book.select_one('p.availability').text.lower()
                
                self.books.append({
                    'title': title,
                    'price': float(price),
                    'rating': rating,
                    'available': available
                })
            
            time.sleep(1)  # Be nice to the server
        
        return self.books
    
    def save_to_json(self, filename: str = 'data/raw/books_raw.json'):
        """Save scraped data to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.books, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {filename}")
```

### 2. Create the transformer (src/transformer.py)
```python
import pandas as pd
from typing import Dict, Any, List
import json

class DataTransformer:
    @staticmethod
    def clean_data(books: List[Dict[str, Any]]) -> pd.DataFrame:
        """Clean and transform the scraped book data."""
        df = pd.DataFrame(books)
        
        # Convert rating to numerical values
        rating_map = {
            'One': 1, 'Two': 2, 'Three': 3, 
            'Four': 4, 'Five': 5
        }
        df['rating'] = df['rating'].map(rating_map)
        
        # Add a price category
        df['price_category'] = pd.cut(
            df['price'],
            bins=[0, 10, 20, 50, float('inf')],
            labels=['Cheap', 'Affordable', 'Expensive', 'Very Expensive']
        )
        
        return df
    
    @staticmethod
    def save_processed_data(df: pd.DataFrame, filename: str = 'data/processed/books_clean.csv'):
        """Save processed data to CSV."""
        df.to_csv(filename, index=False)
        print(f"Processed data saved to {filename}")
```

### 3. Create the loader (src/loader.py)
```python
import pandas as pd
from typing import Optional

class DataLoader:
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """Load data from a CSV or JSON file."""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
    
    @staticmethod
    def get_summary_stats(df: pd.DataFrame) -> dict:
        """Generate summary statistics for the dataset."""
        return {
            'total_books': len(df),
            'avg_price': df['price'].mean(),
            'min_price': df['price'].min(),
            'max_price': df['price'].max(),
            'avg_rating': df['rating'].mean(),
            'availability_rate': df['available'].mean() * 100
        }
```

### 4. Create the main ETL script (notebooks/etl_pipeline.ipynb)
```python
# Import necessary libraries
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from scraper import BookScraper
from transformer import DataTransformer
from loader import DataLoader

# Step 1: Extract
print("Starting data extraction...")
scraper = BookScraper()
books_data = scraper.scrape_books(num_pages=3)  # Scrape first 3 pages
scraper.save_to_json('../data/raw/books_raw.json')

# Step 2: Transform
transformer = DataTransformer()
df_clean = transformer.clean_data(books_data)
transformer.save_processed_data(df_clean, '../data/processed/books_clean.csv')

# Step 3: Load and Analyze
data_loader = DataLoader()
df = data_loader.load_data('../data/processed/books_clean.csv')

# Display basic info
print("\nDataset Info:")
print(df.info())

# Show summary statistics
print("\nSummary Statistics:")
stats = data_loader.get_summary_stats(df)
for key, value in stats.items():
    print(f"{key.replace('_', ' ').title()}: {value:.2f}")

# Data Visualization
print("\nGenerating visualizations...")
plt.figure(figsize=(12, 6))

# Price distribution
plt.subplot(1, 2, 1)
sns.histplot(df['price'], bins=20, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price (£)')

# Rating distribution
plt.subplot(1, 2, 2)
sns.countplot(y='rating', data=df, order=sorted(df['rating'].unique()))
plt.title('Rating Distribution')
plt.xlabel('Count')

plt.tight_layout()
plt.savefig('../data/processed/price_rating_distribution.png')
plt.show()

# Price by rating
plt.figure(figsize=(10, 6))
sns.boxplot(x='rating', y='price', data=df)
plt.title('Price Distribution by Rating')
plt.xlabel('Rating')
plt.ylabel('Price (£)')
plt.tight_layout()
plt.savefig('../data/processed/price_by_rating.png')
plt.show()

print("\nETL pipeline completed successfully!")
```

### 5. Create requirements.txt
```
requests==2.31.0
beautifulsoup4==4.12.2
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
```

## Running the Project
1. Activate your virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Open Jupyter Notebook: `jupyter notebook notebooks/etl_pipeline.ipynb`
4. Run all cells in the notebook

## Expected Output
- Raw JSON data in `data/raw/books_raw.json`
- Processed CSV data in `data/processed/books_clean.csv`
- Visualization images in `data/processed/`
- Console output with summary statistics

## Project Extensions
1. Add error handling and retries for the web scraper
2. Implement database storage (e.g., SQLite, PostgreSQL)
3. Create a command-line interface for the ETL pipeline
4. Add unit tests for the transformation functions
5. Containerize the application with Docker

## Learning Objectives
- Web scraping with BeautifulSoup
- Data cleaning and transformation
- ETL pipeline design
- Data visualization
- Project structure and organization

## Next Steps
1. Try scraping a different website
2. Add more data processing steps
3. Schedule the ETL pipeline to run automatically
4. Deploy the pipeline to a cloud service
