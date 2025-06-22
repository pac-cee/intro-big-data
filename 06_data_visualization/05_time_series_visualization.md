# Time Series Visualization

## Table of Contents
1. [Introduction](#introduction)
2. [Time Series Data in Python](#time-series-data)
3. [Basic Time Series Plots](#basic-plots)
4. [Decomposition](#decomposition)
5. [Seasonal Analysis](#seasonal-analysis)
6. [Moving Averages and Smoothing](#smoothing)
7. [Interactive Time Series](#interactive)
8. [Anomaly Detection](#anomaly-detection)
9. [Forecasting Visualization](#forecasting)
10. [Case Study: Stock Market Data](#case-study)
11. [Best Practices](#best-practices)
12. [Further Reading](#further-reading)

## Introduction

Time series visualization is crucial for:
- Identifying trends and patterns
- Detecting seasonality and cycles
- Spotting anomalies and outliers
- Communicating insights effectively

Common applications include:
- Financial market analysis
- Weather forecasting
- Sales and demand forecasting
- IoT sensor monitoring
- Website traffic analysis

## Time Series Data in Python

### Key Libraries

```bash
pip install pandas numpy matplotlib seaborn plotly statsmodels
```

### Loading Time Series Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create date range
dates = pd.date_range('2023-01-01', periods=365, freq='D')

# Create sample data
np.random.seed(42)
trend = np.linspace(0, 10, 365)
seasonality = 5 * np.sin(2 * np.pi * np.arange(365) / 30)
noise = np.random.normal(0, 1, 365)

# Combine components
data = trend + seasonality + noise

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'value': data,
    'day_of_week': dates.dayofweek,
    'month': dates.month
})

# Set date as index
df.set_index('date', inplace=True)

print(df.head())
```

## Basic Time Series Plots

### Line Plot

```python
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'])
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```

### Multiple Time Series

```python
# Create multiple time series
df['rolling_7'] = df['value'].rolling(window=7).mean()
df['rolling_30'] = df['value'].rolling(window=30).mean()

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['value'], label='Daily', alpha=0.5)
plt.plot(df.index, df['rolling_7'], label='7-day MA', linewidth=2)
plt.plot(df.index, df['rolling_30'], label='30-day MA', linewidth=2)
plt.title('Time Series with Moving Averages')
plt.legend()
plt.grid(True)
plt.show()
```

### Seasonal Plot

```python
import seaborn as sns

# Add week number and day of week
df['week'] = df.index.isocalendar().week
df['day_of_week'] = df.index.dayofweek

# Pivot for heatmap
pivot_df = df.pivot_table(
    index='day_of_week',
    columns='week',
    values='value',
    aggfunc='mean'
)

# Plot heatmap
plt.figure(figsize=(15, 5))
sns.heatmap(pivot_df, cmap='viridis')
plt.title('Seasonal Heatmap')
plt.xlabel('Week of Year')
plt.ylabel('Day of Week')
plt.yticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()
```

## Decomposition

### Additive Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
result = seasonal_decompose(df['value'], model='additive', period=30)

# Plot decomposition
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(result.observed)
plt.title('Observed')

plt.subplot(4, 1, 2)
plt.plot(result.trend)
plt.title('Trend')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal)
plt.title('Seasonal')

plt.subplot(4, 1, 4)
plt.plot(result.resid)
plt.title('Residual')

plt.tight_layout()
plt.show()
```

### Multiplicative Decomposition

```python
# Multiplicative decomposition
result_mul = seasonal_decompose(df['value'], model='multiplicative', period=30)

# Plot multiplicative decomposition
result_mul.plot()
plt.suptitle('Multiplicative Decomposition', y=1.02)
plt.tight_layout()
plt.show()
```

## Seasonal Analysis

### Autocorrelation and Partial Autocorrelation

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12, 6))

# ACF
plt.subplot(1, 2, 1)
plot_acf(df['value'], lags=30, ax=plt.gca())
plt.title('Autocorrelation')

# PACF
plt.subplot(1, 2, 2)
plot_pacf(df['value'], lags=30, ax=plt.gca())
plt.title('Partial Autocorrelation')

plt.tight_layout()
plt.show()
```

### Seasonal Decomposition of Time (STL)

```python
from statsmodels.tsa.seasonal import STL

# STL decomposition
stl = STL(df['value'], period=30)
result = stl.fit()

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(result.observed)
plt.title('Observed')

plt.subplot(4, 1, 2)
plt.plot(result.trend)
plt.title('Trend')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal)
plt.title('Seasonal')

plt.subplot(4, 1, 4)
plt.plot(result.resid)
plt.title('Residual')

plt.tight_layout()
plt.show()
```

## Moving Averages and Smoothing

### Simple Moving Average

```python
# Calculate moving averages
df['sma_7'] = df['value'].rolling(window=7).mean()
df['sma_30'] = df['value'].rolling(window=30).mean()

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['value'], label='Original', alpha=0.5)
plt.plot(df.index, df['sma_7'], label='7-day SMA', linewidth=2)
plt.plot(df.index, df['sma_30'], label='30-day SMA', linewidth=2)
plt.title('Simple Moving Averages')
plt.legend()
plt.grid(True)
plt.show()
```

### Exponential Moving Average

```python
# Calculate EMAs
df['ema_7'] = df['value'].ewm(span=7, adjust=False).mean()
df['ema_30'] = df['value'].ewm(span=30, adjust=False).mean()

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['value'], label='Original', alpha=0.5)
plt.plot(df.index, df['ema_7'], label='7-day EMA', linewidth=2)
plt.plot(df.index, df['ema_30'], label='30-day EMA', linewidth=2)
plt.title('Exponential Moving Averages')
plt.legend()
plt.grid(True)
plt.show()
```

### LOESS Smoothing

```python
import statsmodels.api as sm

# Apply LOWESS smoothing
lowess = sm.nonparametric.lowess(df['value'], df.index, frac=0.1)

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['value'], label='Original', alpha=0.5)
plt.plot(lowess[:, 0], lowess[:, 1], label='LOWESS', color='red', linewidth=2)
plt.title('LOWESS Smoothing')
plt.legend()
plt.grid(True)
plt.show()
```

## Interactive Time Series

### Plotly Express

```python
import plotly.express as px

# Interactive line plot
fig = px.line(df, x=df.index, y='value', 
              title='Interactive Time Series',
              labels={'value': 'Value', 'date': 'Date'},
              template='plotly_white')

# Add range slider
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()
```

### Candlestick Chart (OHLC)

```python
import yfinance as yf
import plotly.graph_objects as go

# Get stock data
df_stock = yf.download('AAPL', start='2022-01-01', end='2023-01-01')

# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df_stock.index,
                open=df_stock['Open'],
                high=df_stock['High'],
                low=df_stock['Low'],
                close=df_stock['Close'])])

fig.update_layout(
    title='AAPL Stock Price',
    yaxis_title='Price ($)',
    xaxis_title='Date',
    template='plotly_white'
)

fig.show()
```

## Anomaly Detection

### Z-Score Method

```python
# Calculate Z-scores
df['z_score'] = (df['value'] - df['value'].mean()) / df['value'].std()

# Define threshold
threshold = 3
df['is_anomaly'] = df['z_score'].abs() > threshold

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['value'], label='Value', alpha=0.5)
plt.scatter(df[df['is_anomaly']].index, 
            df[df['is_anomaly']]['value'], 
            color='red', 
            label='Anomaly')
plt.title('Anomaly Detection using Z-Score')
plt.legend()
plt.grid(True)
plt.show()
```

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Prepare data for model
X = df['value'].values.reshape(-1, 1)

# Fit model
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X)

# Predict anomalies
df['is_anomaly_iso'] = model.predict(X) == -1

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['value'], label='Value', alpha=0.5)
plt.scatter(df[df['is_anomaly_iso']].index, 
            df[df['is_anomaly_iso']]['value'], 
            color='red', 
            label='Anomaly')
plt.title('Anomaly Detection using Isolation Forest')
plt.legend()
plt.grid(True)
plt.show()
```

## Forecasting Visualization

### ARIMA Forecast

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Split data
train_size = int(len(df) * 0.8)
train, test = df['value'][:train_size], df['value'][train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'RMSE: {rmse:.2f}')

# Plot
plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.title('ARIMA Forecast')
plt.legend()
plt.grid(True)
plt.show()
```

### Prophet Forecast

```python
from prophet import Prophet

# Prepare data for Prophet
prophet_df = df[['value']].copy()
prophet_df.reset_index(inplace=True)
prophet_df.columns = ['ds', 'y']

# Split data
train_size = int(len(prophet_df) * 0.8)
train = prophet_df[:train_size]
test = prophet_df[train_size:]

# Fit model
model = Prophet(yearly_seasonality=True)
model.fit(train)

# Create future dates
future = model.make_future_dataframe(periods=len(test))

# Make predictions
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

# Plot components
fig = model.plot_components(forecast)
plt.show()
```

## Case Study: Stock Market Data

### Loading and Preparing Data

```python
import yfinance as yf

# Download stock data
df_aapl = yf.download('AAPL', start='2018-01-01', end='2023-01-01')

# Calculate daily returns
df_aapl['Daily_Return'] = df_aapl['Adj Close'].pct_change()

# Calculate moving averages
df_aapl['MA50'] = df_aapl['Adj Close'].rolling(window=50).mean()
df_aapl['MA200'] = df_aapl['Adj Close'].rolling(window=200).mean()

# Calculate Bollinger Bands
window = 20
num_std = 2
df_aapl['MA20'] = df_aapl['Adj Close'].rolling(window=window).mean()
df_aapl['Upper'] = df_aapl['MA20'] + (df_aapl['Adj Close'].rolling(window=window).std() * num_std)
df_aapl['Lower'] = df_aapl['MA20'] - (df_aapl['Adj Close'].rolling(window=window).std() * num_std)
```

### Interactive Stock Analysis

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.1, 
                   subplot_titles=('Price and Moving Averages', 'Volume'),
                   row_heights=[0.7, 0.3])

# Add traces for price and moving averages
fig.add_trace(
    go.Scatter(x=df_aapl.index, y=df_aapl['Adj Close'], name='Price'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_aapl.index, y=df_aapl['MA50'], name='50-day MA'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_aapl.index, y=df_aapl['MA200'], name='200-day MA'),
    row=1, col=1
)

# Add candlestick trace
fig.add_trace(
    go.Candlestick(x=df_aapl.index,
                  open=df_aapl['Open'],
                  high=df_aapl['High'],
                  low=df_aapl['Low'],
                  close=df_aapl['Close'],
                  name='OHLC',
                  showlegend=False),
    row=1, col=1
)

# Add volume bars
colors = ['green' if close >= open_ else 'red' 
          for close, open_ in zip(df_aapl['Close'], df_aapl['Open'])]

fig.add_trace(
    go.Bar(x=df_aapl.index, y=df_aapl['Volume'], name='Volume', 
          marker_color=colors, opacity=0.5),
    row=2, col=1
)

# Update layout
fig.update_layout(
    title='Apple Stock Analysis',
    yaxis_title='Price ($)',
    xaxis_title='Date',
    template='plotly_white',
    height=800,
    showlegend=True,
    legend=dict(orientation='h', y=1.02, yanchor='bottom', x=0.5, xanchor='center')
)

# Update y-axes
fig.update_yaxes(title_text='Volume', row=2, col=1)

# Hide weekends and non-trading hours
fig.update_xaxes(
    rangebreaks=[
        dict(bounds=['sat', 'mon']),  # hide weekends
        dict(bounds=[20, 13.5], pattern='hour'),  # hide hours outside of 9:30am-4pm
    ],
    rangeslider_visible=False,
    range=[df_aapl.index[0], df_aapl.index[-1]]
)

fig.show()
```

## Best Practices

1. **Data Quality**
   - Handle missing values appropriately
   - Check for and remove duplicates
   - Ensure consistent time intervals

2. **Visual Clarity**
   - Label axes clearly with units
   - Use appropriate date formatting
   - Include a legend for multiple series
   - Choose appropriate scales (linear/log)

3. **Performance**
   - Downsample large datasets for visualization
   - Use aggregation for high-frequency data
   - Consider using specialized time series databases for large datasets

4. **Interpretation**
   - Add reference lines for important events
   - Highlight significant patterns or anomalies
   - Provide context with annotations

5. **Accessibility**
   - Use colorblind-friendly palettes
   - Provide text alternatives for visualizations
   - Ensure sufficient contrast

## Further Reading

- [Pandas Time Series Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
- [Statsmodels Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)
- [Plotly Time Series and Date Axes](https://plotly.com/python/time-series/)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Time Series Analysis with Python](https://www.oreilly.com/library/view/python-time-series/9781492041644/)
