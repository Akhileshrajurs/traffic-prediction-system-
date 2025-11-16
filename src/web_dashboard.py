"""
Web Dashboard using Flask and Plotly Dash for Traffic Congestion Prediction.
Provides interactive visualizations, real-time updates, and route planning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask, jsonify, request
import dash_bootstrap_components as dbc

from gps_input import load_model, simulate_balanced_gps_points, predict_congestion
from route_optimization import RouteOptimizer, print_route_summary
from real_time_data import RealTimeDataFetcher
from data_preprocessing import prepare_dataset


# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app
app = Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Inject custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
            
            * {
                font-family: 'Poppins', sans-serif;
            }
            
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            .stat-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            }
            
            .control-card {
                background: white;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border: none;
            }
            
            .chart-card {
                background: white;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border: none;
                margin-bottom: 1.5rem;
            }
            
            .btn-custom {
                border-radius: 25px;
                padding: 0.75rem 2rem;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .btn-custom:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            
            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                color: #667eea;
            }
            
            .metric-label {
                font-size: 0.9rem;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .badge-custom {
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: 600;
            }
            
            .header-icon {
                font-size: 3rem;
                margin-right: 1rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Global variables for model and data
MODEL_PATH = Path("models") / "traffic_congestion_model.joblib"
DATA_PATH = Path("data") / "traffic_data.csv"
model_bundle = None
current_predictions = []
processed_df = None


def load_model_data():
    """Load model and training data."""
    global model_bundle, processed_df
    if MODEL_PATH.exists():
        model_bundle = load_model(MODEL_PATH)
        _, _, _, processed_df = prepare_dataset(str(DATA_PATH))
    else:
        model_bundle = None
        processed_df = None


# Load model on startup
load_model_data()


# Flask API endpoints
@server.route('/api/predictions', methods=['GET'])
def get_predictions():
    """API endpoint to get current predictions."""
    return jsonify(current_predictions)


@server.route('/api/statistics', methods=['GET'])
def get_statistics():
    """API endpoint to get statistics."""
    if not current_predictions:
        return jsonify({"error": "No predictions available"}), 404
    
    pred_df = pd.DataFrame(current_predictions)
    stats = {
        "total_locations": len(current_predictions),
        "congestion_distribution": pred_df['congestion_level'].value_counts().to_dict(),
        "average_speed": float(pred_df['speed(kmph)'].mean()),
        "average_vehicles": float(pred_df['vehicle_count'].mean()),
    }
    return jsonify(stats)


@server.route('/api/route', methods=['POST'])
def calculate_route():
    """API endpoint to calculate optimal route."""
    data = request.json
    start = tuple(data['start'])
    end = tuple(data['end'])
    num_alternatives = data.get('num_alternatives', 3)
    
    if not current_predictions:
        return jsonify({"error": "No predictions available"}), 404
    
    optimizer = RouteOptimizer(current_predictions)
    routes = optimizer.find_optimal_route(start, end, num_alternatives)
    
    # Convert routes to JSON-serializable format
    routes_data = []
    for route in routes:
        routes_data.append({
            "waypoints": route.waypoints,
            "total_distance": route.total_distance,
            "estimated_time": route.estimated_time,
            "congestion_score": route.congestion_score,
            "congestion_levels": route.congestion_levels
        })
    
    return jsonify({"routes": routes_data})


@server.route('/api/update', methods=['POST'])
def update_predictions():
    """API endpoint to update predictions."""
    global current_predictions
    
    if not model_bundle:
        return jsonify({"error": "Model not loaded"}), 500
    
    num_per_level = request.json.get('num_per_level', 10)
    
    simulated_points = simulate_balanced_gps_points(
        num_per_level=num_per_level,
        bounds=model_bundle.training_data_bounds,
    )
    
    current_predictions = predict_congestion(
        model_bundle,
        simulated_points,
        hour_of_day=datetime.now().hour,
        include_confidence=True
    )
    
    return jsonify({"success": True, "count": len(current_predictions)})


# Dash Layout
app.layout = html.Div([
    dbc.Container([
        # Header Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("🚦", className="header-icon"),
                        html.Div([
                            html.H1("Traffic Congestion Prediction", 
                                   style={"margin": "0", "fontWeight": "700", "fontSize": "2.5rem"}),
                            html.P("Real-time Traffic Analysis & Route Optimization Dashboard", 
                                  style={"margin": "0.5rem 0 0 0", "opacity": "0.9", "fontSize": "1.1rem"}),
                        ])
                    ], style={"display": "flex", "alignItems": "center"}),
                ], className="main-header")
            ], width=12)
        ], className="mb-4"),
        
        # Statistics Cards Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Span("📍", style={"fontSize": "2rem", "marginRight": "1rem"}),
                                html.Div([
                                    html.Div(id="stat-total-locations", className="metric-value", children="0"),
                                    html.Div("Total Locations", className="metric-label"),
                                ])
                            ], style={"display": "flex", "alignItems": "center"})
                        ])
                    ])
                ], className="stat-card")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Span("⚡", style={"fontSize": "2rem", "marginRight": "1rem"}),
                                html.Div([
                                    html.Div(id="stat-avg-speed", className="metric-value", children="0"),
                                    html.Div("Avg Speed (km/h)", className="metric-label"),
                                ])
                            ], style={"display": "flex", "alignItems": "center"})
                        ])
                    ])
                ], className="stat-card")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Span("🚗", style={"fontSize": "2rem", "marginRight": "1rem"}),
                                html.Div([
                                    html.Div(id="stat-avg-vehicles", className="metric-value", children="0"),
                                    html.Div("Avg Vehicles", className="metric-label"),
                                ])
                            ], style={"display": "flex", "alignItems": "center"})
                        ])
                    ])
                ], className="stat-card")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Span("🎯", style={"fontSize": "2rem", "marginRight": "1rem"}),
                                html.Div([
                                    html.Div(id="stat-accuracy", className="metric-value", children="0%"),
                                    html.Div("Model Accuracy", className="metric-label"),
                                ])
                            ], style={"display": "flex", "alignItems": "center"})
                        ])
                    ])
                ], className="stat-card")
            ], width=3),
        ], className="mb-4"),
        
        # Controls Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🎛️ Dashboard Controls", style={"margin": "0", "fontWeight": "600", "color": "#2c3e50"})
                    ], style={"background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "color": "white", "borderRadius": "15px 15px 0 0"}),
                    dbc.CardBody([
                        html.Div([
                            dbc.Label("Number of points per congestion level:", 
                                     style={"fontWeight": "600", "marginBottom": "1rem", "color": "#495057"}),
                            dcc.Slider(
                                id="num-points-slider",
                                min=5,
                                max=20,
                                step=5,
                                value=10,
                                marks={i: {"label": str(i), "style": {"fontWeight": "600"}} for i in range(5, 21, 5)},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Br(),
                            html.Div([
                                dbc.Button([
                                    html.Span("🔄", style={"marginRight": "0.5rem"}),
                                    "Update Predictions"
                                ], id="update-btn", 
                                color="primary", className="btn-custom me-3", size="lg"),
                                dbc.Button([
                                    html.Span("🗺️", style={"marginRight": "0.5rem"}),
                                    "Calculate Route"
                                ], id="route-btn", 
                                color="success", className="btn-custom", size="lg"),
                            ], style={"display": "flex", "justifyContent": "center", "marginTop": "1.5rem"}),
                            html.Div(id="update-status", className="mt-3")
                        ])
                    ])
                ], className="control-card mb-4")
            ], width=12)
        ]),
    
        # Charts Row 1
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📈 Congestion Distribution", style={"margin": "0", "fontWeight": "600", "color": "#2c3e50"})
                    ], style={"background": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", "color": "white", "borderRadius": "15px 15px 0 0"}),
                    dbc.CardBody([
                        dcc.Graph(id="congestion-pie-chart", config={"displayModeBar": False})
                    ], style={"padding": "1.5rem"})
                ], className="chart-card")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🚗 Speed Distribution", style={"margin": "0", "fontWeight": "600", "color": "#2c3e50"})
                    ], style={"background": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)", "color": "white", "borderRadius": "15px 15px 0 0"}),
                    dbc.CardBody([
                        dcc.Graph(id="speed-histogram", config={"displayModeBar": False})
                    ], style={"padding": "1.5rem"})
                ], className="chart-card")
            ], width=6)
        ], className="mb-4"),
    
        # Charts Row 2
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📍 Speed vs Vehicle Count Analysis", style={"margin": "0", "fontWeight": "600", "color": "#2c3e50"})
                    ], style={"background": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)", "color": "white", "borderRadius": "15px 15px 0 0"}),
                    dbc.CardBody([
                        dcc.Graph(id="speed-vehicle-scatter", config={"displayModeBar": False})
                    ], style={"padding": "1.5rem"})
                ], className="chart-card")
            ], width=12)
        ], className="mb-4"),
    
        # Interactive Map
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🗺️ Interactive Traffic Map", style={"margin": "0", "fontWeight": "600", "color": "#2c3e50"})
                    ], style={"background": "linear-gradient(135deg, #30cfd0 0%, #330867 100%)", "color": "white", "borderRadius": "15px 15px 0 0"}),
                    dbc.CardBody([
                        html.Iframe(
                            id="map-iframe",
                            srcDoc=open("traffic_map.html").read() if Path("traffic_map.html").exists() else "<p>Map not available. Run main.py first.</p>",
                            style={"width": "100%", "height": "600px", "border": "none", "borderRadius": "10px"}
                        )
                    ], style={"padding": "1.5rem"})
                ], className="chart-card mb-4")
            ], width=12)
        ]),
    
        # Route Planning Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📍 Route Planning & Optimization", style={"margin": "0", "fontWeight": "600", "color": "#2c3e50"})
                    ], style={"background": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)", "color": "#2c3e50", "borderRadius": "15px 15px 0 0"}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    dbc.Label("📍 Start Location", style={"fontWeight": "600", "marginBottom": "0.5rem", "color": "#495057"}),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Lat"),
                                        dbc.Input(id="start-lat", type="number", value=12.9715, step=0.0001, 
                                                 style={"borderRadius": "0 10px 10px 0"})
                                    ], className="mb-3"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Lon"),
                                        dbc.Input(id="start-lon", type="number", value=77.5946, step=0.0001,
                                                 style={"borderRadius": "0 10px 10px 0"})
                                    ]),
                                ])
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    dbc.Label("🎯 End Location", style={"fontWeight": "600", "marginBottom": "0.5rem", "color": "#495057"}),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Lat"),
                                        dbc.Input(id="end-lat", type="number", value=12.9700, step=0.0001,
                                                 style={"borderRadius": "0 10px 10px 0"})
                                    ], className="mb-3"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Lon"),
                                        dbc.Input(id="end-lon", type="number", value=77.5957, step=0.0001,
                                                 style={"borderRadius": "0 10px 10px 0"})
                                    ]),
                                ])
                            ], width=6)
                        ]),
                        html.Div([
                            dbc.Button([
                                html.Span("🔍", style={"marginRight": "0.5rem"}),
                                "Find Optimal Route"
                            ], id="find-route-btn", color="info", className="btn-custom", size="lg"),
                        ], style={"display": "flex", "justifyContent": "center", "marginTop": "2rem"}),
                        html.Div(id="route-results", className="mt-4")
                    ], style={"padding": "2rem"})
                ], className="control-card mb-4")
            ], width=12)
        ]),
    
        # Predictions Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📋 Live Predictions Table", style={"margin": "0", "fontWeight": "600", "color": "#2c3e50"})
                    ], style={"background": "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)", "color": "#2c3e50", "borderRadius": "15px 15px 0 0"}),
                    dbc.CardBody([
                        html.Div(id="predictions-table")
                    ], style={"padding": "1.5rem"})
                ], className="chart-card mb-4")
            ], width=12)
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Hr(style={"borderTop": "2px solid #dee2e6", "margin": "2rem 0"}),
                    html.P([
                        "🚦 Traffic Congestion Prediction System | ",
                        html.Span("Powered by Machine Learning & Real-time Data", 
                                 style={"opacity": "0.7", "fontSize": "0.9rem"})
                    ], style={"textAlign": "center", "color": "#6c757d", "marginBottom": "1rem"}),
                    html.P([
                        "Auto-refreshes every 30 seconds | ",
                        html.A("API Documentation", href="/api/", style={"color": "#667eea", "textDecoration": "none"})
                    ], style={"textAlign": "center", "color": "#adb5bd", "fontSize": "0.85rem"})
                ])
            ], width=12)
        ], className="mb-4"),
        
        dcc.Interval(
            id='interval-component',
            interval=30*1000,  # Update every 30 seconds
            n_intervals=0
        ),
        
    ], fluid=True, style={"maxWidth": "1400px", "padding": "2rem", "background": "#f8f9fa", "minHeight": "100vh"})
])


# Dash Callbacks
@app.callback(
    [Output('congestion-pie-chart', 'figure'),
     Output('speed-histogram', 'figure'),
     Output('speed-vehicle-scatter', 'figure'),
     Output('stat-total-locations', 'children'),
     Output('stat-avg-speed', 'children'),
     Output('stat-avg-vehicles', 'children'),
     Output('stat-accuracy', 'children'),
     Output('predictions-table', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('update-btn', 'n_clicks')],
    [State('num-points-slider', 'value')]
)
def update_dashboard(n_intervals, n_clicks, num_points):
    """Update dashboard charts and statistics."""
    global current_predictions
    
    # Update predictions if button clicked (n_clicks will be > 0 if clicked)
    if n_clicks and n_clicks > 0:
        if model_bundle:
            simulated_points = simulate_balanced_gps_points(
                num_per_level=num_points,
                bounds=model_bundle.training_data_bounds,
            )
            current_predictions = predict_congestion(
                model_bundle,
                simulated_points,
                hour_of_day=datetime.now().hour,
                include_confidence=True
            )
    
    if not current_predictions:
        # Generate initial predictions
        if model_bundle:
            simulated_points = simulate_balanced_gps_points(
                num_per_level=10,
                bounds=model_bundle.training_data_bounds,
            )
            current_predictions = predict_congestion(
                model_bundle,
                simulated_points,
                hour_of_day=datetime.now().hour,
                include_confidence=True
            )
        else:
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="Model not loaded. Run model_training.py first.",
                                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return empty_fig, empty_fig, empty_fig, "0", "0", "0", "N/A", "Model not loaded. Run model_training.py first."
    
    df = pd.DataFrame(current_predictions)
    
    # Congestion Pie Chart with enhanced styling
    congestion_counts = df['congestion_level'].value_counts()
    pie_fig = px.pie(
        values=congestion_counts.values,
        names=congestion_counts.index,
        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
        hole=0.4
    )
    pie_fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2)),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    pie_fig.update_layout(
        font=dict(family="Poppins", size=12),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Speed Histogram with enhanced styling
    hist_fig = px.histogram(
        df,
        x='speed(kmph)',
        nbins=20,
        labels={'speed(kmph)': 'Speed (km/h)', 'count': 'Frequency'},
        color_discrete_sequence=['#3498db']
    )
    hist_fig.add_vline(x=40, line_dash="dash", line_color="#2ecc71", line_width=2,
                       annotation_text="Low threshold (40 km/h)", annotation_position="top")
    hist_fig.add_vline(x=20, line_dash="dash", line_color="#e74c3c", line_width=2,
                       annotation_text="High threshold (20 km/h)", annotation_position="top")
    hist_fig.update_layout(
        font=dict(family="Poppins", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Speed vs Vehicle Count Scatter with enhanced styling
    scatter_fig = px.scatter(
        df,
        x='speed(kmph)',
        y='vehicle_count',
        color='congestion_level',
        labels={'speed(kmph)': 'Speed (km/h)', 'vehicle_count': 'Vehicle Count'},
        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
        hover_data=['confidence'] if 'confidence' in df.columns else [],
        size_max=15
    )
    scatter_fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white')))
    scatter_fig.update_layout(
        font=dict(family="Poppins", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        legend=dict(title="Congestion Level", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Statistics values for metric cards
    total_locations = len(df)
    avg_speed = f"{df['speed(kmph)'].mean():.1f}"
    avg_vehicles = f"{df['vehicle_count'].mean():.0f}"
    accuracy = f"{model_bundle.accuracy:.1%}" if model_bundle else "N/A"
    
    # Predictions Table with enhanced styling
    table_df = df[['latitude', 'longitude', 'speed(kmph)', 'vehicle_count', 
                   'congestion_level']].head(20).copy()
    table_df['speed(kmph)'] = table_df['speed(kmph)'].round(2)
    
    # Add color coding to congestion levels
    def color_congestion(val):
        colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
        return f"background-color: {colors.get(val, '#95a5a6')}; color: white; font-weight: 600;"
    
    table = dbc.Table.from_dataframe(
        table_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm",
        style={"fontSize": "0.9rem"}
    )
    
    return pie_fig, hist_fig, scatter_fig, total_locations, avg_speed, avg_vehicles, accuracy, table


@app.callback(
    Output('update-status', 'children'),
    Input('update-btn', 'n_clicks'),
    State('num-points-slider', 'value')
)
def update_status(n_clicks, num_points):
    """Show update status."""
    if n_clicks:
        return dbc.Alert(f"✅ Updated with {num_points} points per level!", 
                        color="success", duration=3000)
    return ""


@app.callback(
    Output('route-results', 'children'),
    Input('find-route-btn', 'n_clicks'),
    [State('start-lat', 'value'),
     State('start-lon', 'value'),
     State('end-lat', 'value'),
     State('end-lon', 'value')]
)
def find_route(n_clicks, start_lat, start_lon, end_lat, end_lon):
    """Calculate and display route."""
    if not n_clicks or not current_predictions:
        return ""
    
    try:
        optimizer = RouteOptimizer(current_predictions)
        routes = optimizer.find_optimal_route(
            start=(start_lat, start_lon),
            end=(end_lat, end_lon),
            num_alternatives=3
        )
        
        best_route = routes[0]
        time_est = optimizer.estimate_travel_time(best_route)
        
        results = [
            html.H5("Route Results", className="mt-2"),
            dbc.Alert([
                html.Strong(f"Best Route Found!"),
                html.Br(),
                f"Distance: {best_route.total_distance} km",
                html.Br(),
                f"Estimated Time: {time_est['adjusted_time_minutes']} minutes",
                html.Br(),
                f"Congestion Score: {best_route.congestion_score:.2f}",
            ], color="success"),
            html.H6("All Alternatives:"),
            html.Ul([
                html.Li([
                    f"Route {i+1}: {r.total_distance} km, "
                    f"{r.estimated_time} min, "
                    f"Congestion: {r.congestion_score:.2f}"
                ]) for i, r in enumerate(routes)
            ])
        ]
        
        return results
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🚀 Starting Traffic Congestion Prediction Web Dashboard")
    print("="*70)
    print(f"  📍 Access the dashboard at: http://127.0.0.1:8050")
    print(f"  📊 API endpoints available at: http://127.0.0.1:8050/api/")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)

