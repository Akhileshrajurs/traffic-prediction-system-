"""
Folium map utilities for visualising traffic congestion predictions.
Enhanced with legend, variable marker sizes, and heatmap overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict
from collections import Counter

import folium
from folium.plugins import HeatMap, MarkerCluster


def create_map(
    predictions: Iterable[Dict[str, float]],
    html_output_path: Path,
    default_location: tuple[float, float] | None = None,
    zoom_start: int = 14,
    add_heatmap: bool = True,
    add_legend: bool = True,
) -> Path:
    """
    Plot congestion predictions on an enhanced Folium map with legend, variable markers, and heatmap.
    
    Parameters
    ----------
    predictions : Iterable[Dict[str, float]]
        List of prediction dictionaries with congestion data
    html_output_path : Path
        Where to save the HTML map file
    default_location : tuple[float, float] | None
        Center point (lat, lon) for the map
    zoom_start : int
        Initial zoom level
    add_heatmap : bool
        Whether to add a heatmap overlay
    add_legend : bool
        Whether to add a legend
    """
    predictions = list(predictions)
    if not predictions:
        raise ValueError("No predictions provided for map display.")

    if default_location is None:
        avg_lat = sum(p["latitude"] for p in predictions) / len(predictions)
        avg_lon = sum(p["longitude"] for p in predictions) / len(predictions)
        default_location = (avg_lat, avg_lon)

    # Create map with multiple tile layers
    traffic_map = folium.Map(
        location=default_location,
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )
    
    # Add alternative tile layers
    folium.TileLayer('CartoDB positron').add_to(traffic_map)
    folium.TileLayer('CartoDB dark_matter').add_to(traffic_map)

    # Prepare heatmap data (weighted by congestion severity)
    congestion_weights = {"Low": 1, "Medium": 2, "High": 3}
    heat_data = []
    
    # Create marker cluster for better performance with many markers
    marker_cluster = MarkerCluster().add_to(traffic_map)
    
    for point in predictions:
        congestion = point['congestion_level']
        lat = point["latitude"]
        lon = point["longitude"]
        speed = point['speed(kmph)']
        vehicles = point['vehicle_count']
        color = point.get("marker_color", "blue")
        
        # Variable marker size based on vehicle count (5-15 radius)
        marker_size = 5 + (vehicles / 50) * 10
        marker_size = max(5, min(15, marker_size))
        
        # Enhanced popup with more details
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 5px 0; color: {color};">{congestion} Congestion</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;"><b>Speed:</b> {speed:.2f} km/h</p>
            <p style="margin: 3px 0;"><b>Vehicles:</b> {vehicles}</p>
            <p style="margin: 3px 0;"><b>Location:</b></p>
            <p style="margin: 3px 0; font-size: 0.9em;">Lat: {lat:.6f}</p>
            <p style="margin: 3px 0; font-size: 0.9em;">Lon: {lon:.6f}</p>
        </div>
        """
        
        # Create marker with custom icon
        folium.CircleMarker(
            location=(lat, lon),
            radius=marker_size,
            popup=folium.Popup(popup_html, max_width=250),
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"{congestion}: {speed:.1f} km/h, {vehicles} vehicles"
        ).add_to(marker_cluster)
        
        # Add to heatmap data (weighted by congestion level)
        weight = congestion_weights.get(congestion, 1)
        heat_data.append([lat, lon, weight])
    
    # Add heatmap overlay
    if add_heatmap and heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
        ).add_to(traffic_map)
    
    # Add legend
    if add_legend:
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4 style="margin-top: 0;">Congestion Levels</h4>
        <p><i class="fa fa-circle" style="color:green"></i> Low: Speed > 40 km/h</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Medium: 20-40 km/h</p>
        <p><i class="fa fa-circle" style="color:red"></i> High: Speed < 20 km/h</p>
        <p style="font-size: 11px; margin-top: 5px;">Marker size = Vehicle count</p>
        </div>
        """
        traffic_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(traffic_map)
    
    # Add statistics summary
    congestion_counts = Counter(p['congestion_level'] for p in predictions)
    total = len(predictions)
    stats_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px;">
    <h4 style="margin-top: 0;">Statistics</h4>
    <p><b>Total Locations:</b> {total}</p>
    <p>🟢 Low: {congestion_counts.get('Low', 0)}</p>
    <p>🟡 Medium: {congestion_counts.get('Medium', 0)}</p>
    <p>🔴 High: {congestion_counts.get('High', 0)}</p>
    </div>
    """
    traffic_map.get_root().html.add_child(folium.Element(stats_html))

    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    traffic_map.save(str(html_output_path))
    return html_output_path


