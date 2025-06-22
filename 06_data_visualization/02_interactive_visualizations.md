# Interactive Data Visualizations

## Table of Contents
1. [Introduction to Interactive Visualizations](#introduction)
2. [Bokeh for Interactive Web Visualizations](#bokeh)
3. [D3.js for Custom Visualizations](#d3js)
4. [Plotly Dash for Interactive Dashboards](#plotly-dash)
5. [Interactive Maps with Folium and GeoPandas](#interactive-maps)
6. [Real-time Data Visualization](#real-time-visualization)
7. [Best Practices for Interactive Visualizations](#best-practices)
8. [Case Studies](#case-studies)
9. [Further Reading](#further-reading)

## Introduction to Interactive Visualizations

Interactive visualizations allow users to explore data dynamically, providing a more engaging and insightful experience than static charts. They enable users to:

- Zoom and pan through data
- Hover to see details
- Filter and sort information
- Animate transitions
- Drill down into specific data points

### When to Use Interactive Visualizations

- **Exploratory Data Analysis**: When discovering patterns and insights
- **Dashboards**: For monitoring and business intelligence
- **Storytelling**: To guide users through a data narrative
- **Complex Data**: When multiple dimensions need exploration
- **Web Applications**: For data-driven web apps

## Bokeh for Interactive Web Visualizations

Bokeh is a Python library for creating interactive visualizations that target modern web browsers for presentation.

### Basic Bokeh Plot

```python
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool
from bokeh.sampledata.iris import flowers
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral3

# Prepare data
x = flowers["petal_length"]
y = flowers["petal_width"]
species = flowers["species"]

# Create figure
p = figure(title="Iris Morphology", 
           x_axis_label='Petal Length', 
           y_axis_label='Petal Width',
           tools="pan,box_zoom,reset,save")

# Add scatter renderer with color mapping
p.scatter(x, y, fill_alpha=0.6, size=8, 
          color=factor_cmap('species', palette=Spectral3, 
                          factors=flowers['species'].unique()))

# Add hover tool
tooltips = [
    ("Species", "@species"),
    ("Petal Length", "@petal_length"),
    ("Petal Width", "@petal_width"),
]
p.add_tools(HoverTool(tooltips=tooltips))

# Output to static HTML file
output_file("iris.html")
show(p)
```

### Interactive Widgets

```python
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.plotting import curdoc

# Create data
source = ColumnDataSource(data=dict(
    x=list(range(10)),
    y=list(range(10))
))

# Create plot
plot = figure(plot_height=400, plot_width=600, tools="")
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

# Create widgets
slider = Slider(start=1, end=10, value=1, step=1, title="Multiplier")
select = Select(title="Function:", options=["x", "x^2", "x^3"], value="x")

def update_data(attrname, old, new):
    # Get current widget values
    m = slider.value
    func = select.value
    
    # Generate new data
    x = list(range(11))
    if func == "x":
        y = [m * i for i in x]
    elif func == "x^2":
        y = [m * (i**2) for i in x]
    else:  # x^3
        y = [m * (i**3) for i in x]
    
    # Update data source
    source.data = dict(x=x, y=y)

# Set up callbacks
slider.on_change('value', update_data)
select.on_change('value', update_data)

# Arrange plots and widgets in layouts
layout = column(row(select, slider, width=400), plot)

# Initialize
update_data(None, None, None)

# Serve the plot
curdoc().add_root(layout)
```

### Bokeh Server for Real-time Updates

```python
# Save as app.py
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from random import random
import numpy as np

# Create data source
source = ColumnDataSource(data=dict(x=[0], y=[0]))

# Create plot
plot = figure(plot_height=400, plot_width=800)
plot.line('x', 'y', source=source, line_width=2)

# Add periodic callback
def update():
    new_data = dict(
        x=np.append(source.data['x'], source.data['x'][-1] + 1),
        y=np.append(source.data['y'], random())
    )
    source.data = new_data
    
    # Keep only the last 100 points
    if len(source.data['x']) > 100:
        source.data = {k: v[-100:] for k, v in source.data.items()}
    
    # Update x-range
    plot.x_range.end = source.data['x'][-1] + 1
    plot.x_range.start = max(0, plot.x_range.end - 20)

# Add to document and set up periodic updates
curdoc().add_root(column(plot))
curdoc().add_periodic_callback(update, 500)  # Update every 500ms
```

## D3.js for Custom Visualizations

D3.js is a powerful JavaScript library for creating custom data visualizations in the browser.

### Basic D3.js Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>D3.js Bar Chart</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .bar {
            fill: steelblue;
        }
        .bar:hover {
            fill: orange;
        }
    </style>
</head>
<body>
    <div id="chart"></div>
    
    <script>
        // Data
        const data = [
            {category: 'A', value: 40},
            {category: 'B', value: 90},
            {category: 'C', value: 30},
            {category: 'D', value: 50},
            {category: 'E', value: 70}
        ];
        
        // Set dimensions and margins
        const margin = {top: 20, right: 30, bottom: 40, left: 40};
        const width = 600 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;
        
        // Create SVG
        const svg = d3.select("#chart")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);
        
        // Create scales
        const x = d3.scaleBand()
            .domain(data.map(d => d.category))
            .range([0, width])
            .padding(0.1);
            
        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.value)])
            .nice()
            .range([height, 0]);
        
        // Add axes
        svg.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(x));
            
        svg.append("g")
            .attr("class", "y-axis")
            .call(d3.axisLeft(y));
        
        // Add bars
        svg.selectAll(".bar")
            .data(data)
            .enter().append("rect")
            .attr("class", "bar")
            .attr("x", d => x(d.category))
            .attr("y", d => y(d.value))
            .attr("width", x.bandwidth())
            .attr("height", d => height - y(d.value))
            .on("mouseover", function(event, d) {
                d3.select(this).style("fill", "orange");
                // Show tooltip
                svg.append("text")
                    .attr("class", "tooltip")
                    .attr("x", x(d.category) + x.bandwidth() / 2)
                    .attr("y", y(d.value) - 5)
                    .attr("text-anchor", "middle")
                    .text(d.value);
            })
            .on("mouseout", function() {
                d3.select(this).style("fill", "steelblue");
                // Remove tooltip
                svg.select(".tooltip").remove();
            });
    </script>
</body>
</html>
```

### Interactive Network Graph with D3.js

```html
<!DOCTYPE html>
<html>
<head>
    <title>D3.js Force-Directed Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        .node {
            stroke: #fff;
            stroke-width: 1.5px;
        }
        .node text {
            pointer-events: none;
            font: 10px sans-serif;
        }
    </style>
</head>
<body>
    <div id="graph"></div>
    
    <script>
        // Sample graph data
        const graph = {
            nodes: [
                {id: 1, name: "Node 1"},
                {id: 2, name: "Node 2"},
                {id: 3, name: "Node 3"},
                {id: 4, name: "Node 4"},
                {id: 5, name: "Node 5"}
            ],
            links: [
                {source: 1, target: 2},
                {source: 2, target: 3},
                {source: 3, target: 4},
                {source: 4, target: 5},
                {source: 5, target: 1},
                {source: 1, target: 3},
                {source: 2, target: 4}
            ]
        };
        
        // Set up SVG
        const width = 800, height = 600;
        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create a group for the graph
        const g = svg.append("g");
        
        // Set up zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
            
        svg.call(zoom);
        
        // Create a color scale
        const color = d3.scaleOrdinal(d3.schemeCategory10);
        
        // Create the force simulation
        const simulation = d3.forceSimulation(graph.nodes)
            .force("link", d3.forceLink(graph.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));
        
        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(graph.links)
            .enter().append("line")
            .attr("class", "link");
        
        // Create node groups
        const node = g.append("g")
            .selectAll(".node")
            .data(graph.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // Add circles to nodes
        node.append("circle")
            .attr("r", 10)
            .attr("fill", (d, i) => color(i));
            
        // Add labels to nodes
        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .text(d => d.name);
        
        // Update positions on each tick
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
                
            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });
        
        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    </script>
</body>
</html>
```

## Best Practices for Interactive Visualizations

1. **Performance Optimization**
   - Use data aggregation for large datasets
   - Implement data sampling when possible
   - Use WebGL for rendering large datasets (e.g., with deck.gl or plotly.js)
   - Implement virtual scrolling for long lists

2. **Responsive Design**
   - Make visualizations adapt to different screen sizes
   - Use relative units (%, vh, vw) instead of fixed pixels
   - Implement mobile-friendly interactions

3. **Accessibility**
   - Add proper ARIA labels
   - Ensure keyboard navigation
   - Provide text alternatives for visual elements
   - Use sufficient color contrast

4. **User Experience**
   - Provide clear instructions
   - Add loading indicators for slow operations
   - Implement smooth transitions
   - Offer reset/undo functionality

5. **Testing**
   - Test across different browsers and devices
   - Test with different data volumes
   - Get user feedback early and often

## Further Reading

1. [Bokeh Documentation](https://docs.bokeh.org/)
2. [D3.js Documentation](https://d3js.org/)
3. [Interactive Data Visualization for the Web](https://www.oreilly.com/library/view/interactive-data-visualization/9781449340223/)
4. [Data Visualization with Python and JavaScript](https://www.oreilly.com/library/view/data-visualization-with/9781491920941/)
5. [Visualization Analysis and Design](https://www.cs.ubc.ca/~tmm/vadbook/)
