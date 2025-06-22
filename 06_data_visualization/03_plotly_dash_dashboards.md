# Building Interactive Dashboards with Plotly Dash

## Table of Contents
1. [Introduction](#introduction)
2. [Basic App Structure](#basic-app)
3. [Core Components](#core-components)
4. [Layout and Styling](#layout-styling)
5. [Callbacks](#callbacks)
6. [Deployment](#deployment)
7. [Example Dashboard](#example-dashboard)

## Introduction

Plotly Dash is a Python framework for building analytical web applications. It's built on top of Flask, React.js, and Plotly.js.

### Installation

```bash
pip install dash pandas plotly dash-bootstrap-components
```

## Basic App Structure

```python
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

# Sample data
df = pd.DataFrame({
    'Fruit': ['Apples', 'Oranges', 'Bananas'],
    'Amount': [4, 1, 2],
    'City': ['SF', 'SF', 'SF']
})

# Create figure
fig = px.bar(df, x='Fruit', y='Amount', color='City', barmode='group')

# App layout
app.layout = html.Div([
    html.H1('Hello Dash'),
    html.Div('Dash: A web application framework for your data.'),
    dcc.Graph(id='example-graph', figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
```

## Core Components

### HTML Components

```python
import dash_html_components as html

layout = html.Div([
    html.H1('Heading'),
    html.Div([
        html.P('A paragraph with text.'),
        html.A('Link', href='https://dash.plotly.com/')
    ])
])
```

### Interactive Components

```python
from dash import dcc

layout = html.Div([
    # Dropdown
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'New York', 'value': 'NYC'},
            {'label': 'Montreal', 'value': 'MTL'}
        ],
        value='MTL'
    ),
    
    # Slider
    dcc.Slider(
        min=0,
        max=10,
        step=0.5,
        value=5,
        marks={i: str(i) for i in range(11)}
    ),
    
    # Graph
    dcc.Graph(id='graph')
])
```

## Layout and Styling

### Using Bootstrap

```python
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='graph1'), md=6),
        dbc.Col(dcc.Graph(id='graph2'), md=6)
    ])
])
```

## Callbacks

```python
from dash.dependencies import Input, Output

@app.callback(
    Output('output-div', 'children'),
    [Input('dropdown', 'value')]
)
def update_output(selected_value):
    return f'You selected: {selected_value}'
```

## Deployment

### Local Deployment

```bash
# Install gunicorn
pip install gunicorn

# Run app
gunicorn -w 4 -b :8050 app:server
```

### Heroku Deployment

1. Create `Procfile`:

```
web: gunicorn app:server
```

2. Create `requirements.txt`
3. Deploy:

```bash
git init
git add .
git commit -m "Initial commit"
heroku create
git push heroku main
```

## Example Dashboard

```python
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# Sample data
df = px.data.iris()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Iris Dashboard"), width=12, className="text-center my-4")
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    html.Label("X-Axis"),
                    dcc.Dropdown(
                        id='x-axis',
                        options=[{'label': col, 'value': col} 
                                for col in df.columns[:4]],
                        value='sepal_width'
                    ),
                    
                    html.Label("Y-Axis", className="mt-3"),
                    dcc.Dropdown(
                        id='y-axis',
                        options=[{'label': col, 'value': col} 
                                for col in df.columns[:4]],
                        value='sepal_length'
                    ),
                    
                    html.Label("Species", className="mt-3"),
                    dcc.Checklist(
                        id='species-filter',
                        options=[
                            {'label': species, 'value': species}
                            for species in df['species'].unique()
                        ],
                        value=df['species'].unique().tolist(),
                        labelStyle={'display': 'block'}
                    )
                ])
            ])
        ], md=3),
        
        dbc.Col([
            dcc.Graph(id='scatter-plot')
        ], md=9)
    ])
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis', 'value'),
     Input('y-axis', 'value'),
     Input('species-filter', 'value')]
)
update_scatter(x_col, y_col, selected_species):
    filtered_df = df[df['species'].isin(selected_species)]
    
    fig = px.scatter(
        filtered_df,
        x=x_col,
        y=y_col,
        color='species',
        title=f"{y_col} vs {x_col}",
        labels={
            x_col: x_col.replace('_', ' ').title(),
            y_col: y_col.replace('_', ' ').title()
        }
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

## Best Practices

1. **Code Organization**
   - Split code into multiple files
   - Use functions for data processing
   - Keep callbacks organized

2. **Performance**
   - Use caching
   - Load data efficiently
   - Update only necessary components

3. **UI/UX**
   - Make it responsive
   - Add loading states
   - Provide clear feedback

4. **Security**
   - Validate inputs
   - Use environment variables for secrets
   - Implement authentication if needed

## Further Reading

- [Dash Documentation](https://dash.plotly.com/)
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
- [Plotly Express](https://plotly.com/python/plotly-express/)
