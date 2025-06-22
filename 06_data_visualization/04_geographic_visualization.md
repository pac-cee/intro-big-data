# Geographic Data Visualization

## Table of Contents
1. [Introduction](#introduction)
2. [Working with Geospatial Data](#geospatial-data)
3. [Static Maps with Matplotlib](#static-maps)
4. [Interactive Maps with Folium](#folium)
5. [Choropleth Maps](#choropleth)
6. [Geographic Heatmaps](#heatmaps)
7. [3D Terrain Maps](#3d-maps)
8. [Case Study: COVID-19 Map](#case-study)
9. [Best Practices](#best-practices)
10. [Further Reading](#further-reading)

## Introduction

Geographic visualization helps in understanding spatial patterns and relationships in data. Common use cases include:

- Population density analysis
- Sales territory mapping
- Environmental monitoring
- Transportation and logistics
- Real estate and urban planning

## Working with Geospatial Data

### Key Libraries

```bash
pip install geopandas folium plotly mapclassify contextily
```

### Loading Geographic Data

```python
import geopandas as gpd
import pandas as pd

# Load shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# View first few rows
print(world.head())

# Basic plot
world.plot(figsize=(10, 6))
```

### Coordinate Reference Systems (CRS)

```python
# Check current CRS
print(world.crs)

# Convert to Web Mercator (common for web maps)
world_web_mercator = world.to_crs(epsg=3857)

# Convert to WGS84 (lat/lon)
world_wgs84 = world.to_crs(epsg=4326)
```

## Static Maps with Matplotlib

### Basic Map

```python
import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 10))

# Plot countries
world.boundary.plot(ax=ax, linewidth=1)

# Customize
ax.set_title('World Map')
ax.set_axis_off()
plt.tight_layout()
plt.show()
```

### Thematic Map

```python
# Plot population
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(column='pop_est', 
           ax=ax, 
           legend=True,
           legend_kwds={'label': 'Population',
                       'orientation': 'horizontal'})

# Add country labels
for x, y, label in zip(world.geometry.centroid.x, 
                      world.geometry.centroid.y,
                      world['name']):
    ax.text(x, y, label, fontsize=8)

plt.title('World Population')
plt.axis('off')
plt.show()
```

## Interactive Maps with Folium

### Basic Map

```python
import folium

# Create map centered on a location
m = folium.Map(location=[51.5074, -0.1278],  # London
               zoom_start=10,
               tiles='OpenStreetMap')

# Add a marker
folium.Marker(
    [51.5074, -0.1278],
    popup='London',
    tooltip='Click me!'
).add_to(m)


# Save to HTML
m.save('london_map.html')
```

### Multiple Markers

```python
# Sample data
cities = {
    'London': {'lat': 51.5074, 'lon': -0.1278, 'pop': 8908081},
    'Paris': {'lat': 48.8566, 'lon': 2.3522, 'pop': 2148327},
    'Berlin': {'lat': 52.5200, 'lon': 13.4050, 'pop': 3769495}
}

# Create map
m = folium.Map(location=[51.1657, 10.4515], zoom_start=5)

# Add markers
for city, info in cities.items():
    folium.CircleMarker(
        location=[info['lat'], info['lon']],
        radius=5 + (info['pop'] / 1000000),  # Scale by population
        popup=f"{city}<br>Population: {info['pop']:,}",
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

m.save('europe_cities.html')
```

## Choropleth Maps

### Country-level Choropleth

```python
# Create a sample dataframe
data = {
    'country': ['United States', 'Canada', 'Mexico', 'Brazil', 'Argentina'],
    'value': [100, 85, 70, 55, 40]
}
df = pd.DataFrame(data)

# Merge with world data
merged = world.merge(df, left_on='name', right_on='country', how='left')

# Plot
fig, ax = plt.subplots(figsize=(15, 10))
merged.plot(column='value', 
           ax=ax, 
           legend=True,
           legend_kwds={'label': 'Value',
                       'orientation': 'horizontal'},
           cmap='viridis',
           missing_kwds={"color": "lightgrey"})

plt.title('Sample Choropleth Map')
plt.axis('off')
plt.show()
```

### Interactive Choropleth with Folium

```python
# Sample data
data = {
    'country': ['United States', 'Canada', 'Mexico', 'Brazil', 'Argentina'],
    'value': [100, 85, 70, 55, 40]
}
df = pd.DataFrame(data)

# Create base map
m = folium.Map(location=[0, 0], zoom_start=2)

# Add choropleth layer
folium.Choropleth(
    geo_data=world,
    name='choropleth',
    data=df,
    columns=['country', 'value'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Value'
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)


m.save('interactive_choropleth.html')
```

## Geographic Heatmaps

### Static Heatmap

```python
import numpy as np
from scipy.stats import multivariate_normal

# Generate sample data
np.random.seed(42)
lats = np.random.normal(40.7128, 1, 1000)
lons = np.random.normal(-74.0060, 1, 1000)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.hexbin(lons, lats, gridsize=30, cmap='YlOrRd')
ax.set_title('Heatmap of Points')
plt.show()
```

### Interactive Heatmap with Folium

```python
from folium.plugins import HeatMap

# Sample data (lat, lon, weight)
data = [[np.random.uniform(35, 45), 
          np.random.uniform(-100, -70), 
          np.random.random()] for _ in range(1000)]

# Create map
m = folium.Map([40, -85], zoom_start=4)

# Add heatmap
HeatMap(data, radius=15).add_to(m)

m.save('heatmap.html')
```

## 3D Terrain Maps

### 3D Surface Plot

```python
import numpy as np
import plotly.graph_objects as go

# Sample elevation data
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Create 3D surface
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

# Update layout
fig.update_layout(
    title='3D Terrain',
    autosize=False,
    width=800,
    height=600,
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.show()
```

### 3D Map with Real Terrain

```python
import plotly.express as px

# Sample data with elevation
df = px.data.election()

fig = px.scatter_3d(
    df, 
    x="Joly", 
    y="Coderre", 
    z="Bergeron",
    color="winner",
    size="total",
    hover_name="district",
    size_max=20,
    opacity=0.7
)

fig.update_layout(
    title="3D Election Results",
    scene=dict(
        xaxis_title="Joly",
        yaxis_title="Coderre",
        zaxis_title="Bergeron"
    )
)

fig.show()
```

## Case Study: COVID-19 Map

```python
import pandas as pd
import plotly.express as px

# Load COVID-19 data
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df = pd.read_csv(url)

# Process data
df = df.melt(
    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
    var_name='Date',
    value_name='Confirmed Cases'
)
df['Date'] = pd.to_datetime(df['Date'])

# Create animation
fig = px.scatter_geo(
    df,
    lat='Lat',
    lon='Long',
    size='Confirmed Cases',
    color='Confirmed Cases',
    hover_name='Country/Region',
    animation_frame=df['Date'].dt.strftime('%Y-%m-%d'),
    projection='natural earth',
    title='COVID-19 Confirmed Cases Over Time',
    size_max=50,
    color_continuous_scale='OrRd'
)

fig.update_geos(
    showcoastlines=True,
    coastlinecolor="Black",
    showland=True,
    landcolor="lightgray",
    showocean=True,
    oceancolor="LightBlue"
)

fig.update_layout(
    height=600,
    margin=dict(l=0, r=0, t=50, b=0)
)

fig.show()
```

## Best Practices

1. **Choose Appropriate Projections**
   - Use Web Mercator for web maps
   - Use equal-area projections for area comparisons
   - Consider your audience's familiarity with projections

2. **Color Choices**
   - Use colorblind-friendly palettes
   - Consider cultural implications of colors
   - Use sequential color schemes for ordered data
   - Use diverging schemes for data with a meaningful mid-point

3. **Performance Optimization**
   - Simplify geometries for web maps
   - Use appropriate zoom levels
   - Consider using vector tiles for large datasets

4. **Interactivity**
   - Add tooltips with relevant information
   - Include a legend
   - Provide context (scale bar, north arrow)
   - Allow users to toggle layers

5. **Accessibility**
   - Add alt text for images
   - Ensure sufficient color contrast
   - Provide text alternatives for visualizations

## Further Reading

- [Geopandas Documentation](https://geopandas.org/)
- [Folium Documentation](https://python-visualization.github.io/folium/)
- [Plotly Python Maps](https://plotly.com/python/maps/)
- [Mapbox GL JS](https://docs.mapbox.com/mapbox-gl-js/api/)
- [CartoPy Documentation](https://scitools.org.uk/cartopy/docs/latest/)
