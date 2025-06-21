# Data Visualization with Python

## Table of Contents
1. [Introduction to Data Visualization](#introduction-to-data-visualization)
2. [Matplotlib Fundamentals](#matplotlib-fundamentals)
3. [Seaborn for Statistical Visualization](#seaborn-for-statistical-visualization)
4. [Plotly for Interactive Visualizations](#plotly-for-interactive-visualizations)
5. [Dashboarding with Dash](#dashboarding-with-dash)
6. [Big Data Visualization Techniques](#big-data-visualization-techniques)
7. [Best Practices](#best-practices)
8. [Practice Exercises](#practice-exercises)

## Introduction to Data Visualization

Data visualization is the graphical representation of data to help people understand patterns, trends, and insights.

### Why Visualize Data?
- Identify patterns and trends
- Communicate findings effectively
- Detect outliers and anomalies
- Support decision-making
- Tell compelling data stories

### Types of Visualizations
1. **Comparison**: Bar charts, line charts, radar charts
2. **Distribution**: Histograms, box plots, violin plots
3. **Composition**: Pie charts, stacked bars, treemaps
4. **Relationship**: Scatter plots, bubble charts, heatmaps
5. **Geospatial**: Choropleth maps, point maps, flow maps

## Matplotlib Fundamentals

### Basic Plotting
```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
ax.plot(x, y, label='sin(x)', color='blue', linewidth=2)

# Customize plot
ax.set_title('Sine Wave')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()
ax.grid(True)

# Show plot
plt.tight_layout()
plt.show()
```

### Multiple Plots
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot
ax1.plot(x, np.sin(x), 'r-')
ax1.set_title('Sine')

# Second subplot
ax2.plot(x, np.cos(x), 'b--')
ax2.set_title('Cosine')

plt.tight_layout()
plt.show()
```

## Seaborn for Statistical Visualization

### Distribution Plots
```python
import seaborn as sns
import pandas as pd

# Load sample dataset
tips = sns.load_dataset('tips')

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', kde=True, bins=30)
plt.title('Distribution of Total Bill')
plt.show()
```

### Categorical Plots
```python
# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Total Bill by Day')
plt.show()

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, split=True)
plt.title('Total Bill by Day and Gender')
plt.show()
```

### Heatmaps
```python
# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

## Plotly for Interactive Visualizations

### Interactive Scatter Plot
```python
import plotly.express as px

# Create interactive scatter plot
fig = px.scatter(tips, x='total_bill', y='tip', 
                 color='sex', size='size', 
                 hover_data=['day', 'time'],
                 title='Total Bill vs Tip')
fig.show()
```

### 3D Scatter Plot
```python
# 3D scatter plot
fig = px.scatter_3d(tips, x='total_bill', y='tip', z='size',
                    color='sex', symbol='smoker',
                    title='3D Scatter Plot')
fig.show()
```

### Animated Plots
```python
# Load sample data
gapminder = px.data.gapminder()

# Create animated scatter plot
fig = px.scatter(gapminder, x='gdpPercap', y='lifeExp', 
                 size='pop', color='continent',
                 hover_name='country', 
                 animation_frame='year',
                 size_max=60,
                 range_x=[300, 120000],
                 range_y=[25, 90],
                 log_x=True)

fig.update_layout(title='Life Expectancy vs GDP per Capita (1952-2007)')
fig.show()
```

## Dashboarding with Dash

### Basic Dash App
```python
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Load data
df = px.data.iris()

# Initialize the app
app = Dash(__name__)


# App layout
app.layout = html.Div([
    html.H1('Iris Dashboard'),
    
    dcc.Dropdown(
        id='x-axis',
        options=[{'label': col, 'value': col} for col in df.columns[:4]],
        value='sepal_width',
        clearable=False
    ),
    
    dcc.Dropdown(
        id='y-axis',
        options=[{'label': col, 'value': col} for col in df.columns[:4]],
        value='sepal_length',
        clearable=False
    ),
    
    dcc.Graph(id='scatter-plot')
])

# Callback to update graph
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis', 'value'),
     Input('y-axis', 'value')]
)
def update_graph(x_axis, y_axis):
    fig = px.scatter(df, x=x_axis, y=y_axis, color='species')
    fig.update_layout(transition_duration=500)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
```

## Big Data Visualization Techniques

### Sampling
```python
# Random sampling
df_sample = df.sample(frac=0.1)  # 10% random sample
```

### Aggregation
```python
# Group by and aggregate
df_agg = df.groupby('species').agg({
    'sepal_width': 'mean',
    'sepal_length': 'mean'
}).reset_index()
```

### Hexbin Plots
```python
# Hexbin for large datasets
plt.hexbin(x=df['sepal_width'], y=df['sepal_length'], 
           gridsize=20, cmap='viridis')
plt.colorbar(label='Count')
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.show()
```

## Best Practices

### Design Principles
1. **Clarity**: Make sure your visual is easy to understand
2. **Accuracy**: Represent data truthfully
3. **Efficiency**: Convey maximum information with minimum ink
4. **Aesthetics**: Use color and design elements purposefully

### Color Choices
- Use colorblind-friendly palettes
- Be consistent with color meanings
- Use color to highlight important information
- Consider grayscale printing

### Accessibility
- Add alt text for screen readers
- Ensure sufficient color contrast
- Use patterns or textures in addition to color
- Provide text descriptions of key insights

## Practice Exercises

1. **Exploratory Data Analysis**
   - Load the Titanic dataset
   - Create visualizations to explore relationships between features
   - Identify patterns in survival rates

2. **Interactive Dashboard**
   - Build a Dash app with multiple interactive components
   - Include at least 3 different types of plots
   - Add filters to control the data displayed

3. **Time Series Visualization**
   - Visualize stock price data over time
   - Add moving averages and technical indicators
   - Create an interactive plot with zoom and hover features

4. **Geospatial Visualization**
   - Create a choropleth map showing COVID-19 cases by country
   - Add interactive tooltips with additional information
   - Include a time slider to show progression over time

5. **Big Data Challenge**
   - Work with a dataset containing >1 million rows
   - Implement appropriate sampling or aggregation
   - Create meaningful visualizations that reveal insights

## Resources
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Plotly Documentation](https://plotly.com/python/)
- [Dash Documentation](https://dash.plotly.com/)
- [ColorBrewer](https://colorbrewer2.org/) - Color scheme advisor

---
Next: [Advanced Visualization Techniques](./02_advanced_visualization.md)
