# Simple Data Analysis Project

## Project Overview
This project will help you practice Python basics while performing simple data analysis on a dataset of movies.

## Project Structure
```
01_simple_data_analysis/
├── data/
│   └── movies.csv
├── analysis.ipynb
└── README.md
```

## Setup Instructions
1. Create a new Jupyter Notebook named `analysis.ipynb`
2. Download the sample dataset:
   ```python
   import pandas as pd
   
   # Sample dataset
   data = {
       'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction'],
       'year': [1994, 1972, 2008, 1994],
       'genre': ['Drama', 'Crime', 'Action', 'Crime'],
       'rating': [9.3, 9.2, 9.0, 8.9],
       'votes': [2600000, 1800000, 2500000, 2000000]
   }
   
   df = pd.DataFrame(data)
   df.to_csv('data/movies.csv', index=False)
   ```

## Project Tasks
1. Load the dataset into a pandas DataFrame
2. Display basic information about the dataset
3. Find the average rating of all movies
4. Find the most common genre
5. Create a new column called 'decade' based on the year
6. (Bonus) Create a simple visualization of the ratings

## Learning Objectives
- Practice reading and writing CSV files
- Basic data manipulation with pandas
- Simple data analysis techniques
- Basic data visualization

## Next Steps
Once you complete this project, try finding a larger dataset (e.g., from Kaggle) and perform similar analysis.
