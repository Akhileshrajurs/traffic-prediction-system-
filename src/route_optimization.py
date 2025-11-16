"""
Route optimization module for finding optimal paths avoiding high congestion.
Includes multi-point planning, travel time estimation, and alternative routes.
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Route:
    """Represents a route with waypoints and metadata."""
    waypoints: List[Tuple[float, float]]  # List of (lat, lon) tuples
    total_distance: float  # in kilometers
    estimated_time: float  # in minutes
    congestion_score: float  # 0-1, higher = more congestion
    congestion_levels: List[str]  # Congestion level at each segment
    alternative_rank: int = 1  # Rank among alternatives


class RouteOptimizer:
    """
    Optimizes routes to avoid high congestion areas.
    """
    
    def __init__(self, congestion_predictions: List[Dict]):
        """
        Initialize route optimizer with congestion data.
        
        Parameters
        ----------
        congestion_predictions : List[Dict]
            List of predictions with latitude, longitude, and congestion_level
        """
        self.predictions = pd.DataFrame(congestion_predictions)
        self.congestion_map = self._build_congestion_map()
    
    def _build_congestion_map(self) -> Dict[Tuple[float, float], str]:
        """
        Build a map of coordinates to congestion levels.
        """
        congestion_map = {}
        for _, row in self.predictions.iterrows():
            # Round to 4 decimal places for grid matching (~11m precision)
            key = (
                round(row['latitude'], 4),
                round(row['longitude'], 4)
            )
            congestion_map[key] = row['congestion_level']
        return congestion_map
    
    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two GPS points using Haversine formula.
        Returns distance in kilometers.
        """
        R = 6371  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    def _get_congestion_at_point(self, lat: float, lon: float) -> str:
        """
        Get congestion level at a specific point.
        """
        # Find nearest prediction point
        key = (round(lat, 4), round(lon, 4))
        
        if key in self.congestion_map:
            return self.congestion_map[key]
        
        # If exact match not found, find nearest neighbor
        min_dist = float('inf')
        nearest_congestion = "Medium"
        
        for pred_lat, pred_lon in self.congestion_map.keys():
            dist = self._haversine_distance(lat, lon, pred_lat, pred_lon)
            if dist < min_dist:
                min_dist = dist
                nearest_congestion = self.congestion_map[(pred_lat, pred_lon)]
        
        return nearest_congestion
    
    def _calculate_congestion_score(self, congestion_level: str) -> float:
        """
        Convert congestion level to numeric score (0-1).
        """
        scores = {"Low": 0.2, "Medium": 0.5, "High": 0.9}
        return scores.get(congestion_level, 0.5)
    
    def find_optimal_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_alternatives: int = 3,
        avoid_high_congestion: bool = True
    ) -> List[Route]:
        """
        Find optimal route from start to end, avoiding high congestion.
        
        Parameters
        ----------
        start : Tuple[float, float]
            Starting point (lat, lon)
        end : Tuple[float, float]
            Destination (lat, lon)
        num_alternatives : int
            Number of alternative routes to generate
        avoid_high_congestion : bool
            Whether to prioritize avoiding high congestion
        
        Returns
        -------
        List[Route]
            List of route alternatives, sorted by best first
        """
        routes = []
        
        # Generate multiple route alternatives
        for i in range(num_alternatives):
            waypoints = self._generate_route_waypoints(start, end, i, num_alternatives)
            route = self._evaluate_route(waypoints, avoid_high_congestion)
            route.alternative_rank = i + 1
            routes.append(route)
        
        # Sort by congestion score (lower is better) and time
        routes.sort(key=lambda r: (r.congestion_score, r.estimated_time))
        
        return routes
    
    def _generate_route_waypoints(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        variant: int,
        total_variants: int
    ) -> List[Tuple[float, float]]:
        """
        Generate waypoints for a route variant.
        """
        start_lat, start_lon = start
        end_lat, end_lon = end
        
        # Simple waypoint generation (in real implementation, would use routing API)
        # Create intermediate waypoints with slight variations
        num_waypoints = 3 + variant
        
        waypoints = [start]
        
        for i in range(1, num_waypoints):
            # Interpolate with some variation
            t = i / num_waypoints
            
            # Add variation based on variant number
            variation = 0.01 * (variant - total_variants / 2) / total_variants
            
            lat = start_lat + (end_lat - start_lat) * t + variation
            lon = start_lon + (end_lon - start_lon) * t + variation * 0.5
            
            waypoints.append((lat, lon))
        
        waypoints.append(end)
        return waypoints
    
    def _evaluate_route(
        self,
        waypoints: List[Tuple[float, float]],
        avoid_high_congestion: bool
    ) -> Route:
        """
        Evaluate a route and calculate metrics.
        """
        total_distance = 0
        total_time = 0
        congestion_levels = []
        congestion_scores = []
        
        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]
            lat2, lon2 = waypoints[i + 1]
            
            # Calculate segment distance
            segment_dist = self._haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += segment_dist
            
            # Get congestion at midpoint
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            congestion = self._get_congestion_at_point(mid_lat, mid_lon)
            congestion_levels.append(congestion)
            
            # Calculate speed based on congestion
            speed_map = {"Low": 50, "Medium": 30, "High": 15}  # km/h
            speed = speed_map.get(congestion, 30)
            
            # Calculate time for this segment
            segment_time = (segment_dist / speed) * 60  # minutes
            total_time += segment_time
            
            # Accumulate congestion score
            congestion_scores.append(self._calculate_congestion_score(congestion))
        
        # Average congestion score
        avg_congestion_score = np.mean(congestion_scores) if congestion_scores else 0.5
        
        return Route(
            waypoints=waypoints,
            total_distance=round(total_distance, 2),
            estimated_time=round(total_time, 1),
            congestion_score=round(avg_congestion_score, 3),
            congestion_levels=congestion_levels
        )
    
    def plan_multi_point_route(
        self,
        points: List[Tuple[float, float]],
        optimize_order: bool = True
    ) -> Route:
        """
        Plan a route visiting multiple points.
        
        Parameters
        ----------
        points : List[Tuple[float, float]]
            List of points to visit (lat, lon)
        optimize_order : bool
            Whether to optimize the order of visits
        
        Returns
        -------
        Route
            Optimized multi-point route
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points for route planning")
        
        if optimize_order:
            # Simple optimization: visit points in order that minimizes total distance
            # (In production, would use TSP solver)
            optimized_points = self._optimize_point_order(points)
        else:
            optimized_points = points
        
        # Build route through all points
        waypoints = []
        for i in range(len(optimized_points) - 1):
            segment_waypoints = self._generate_route_waypoints(
                optimized_points[i],
                optimized_points[i + 1],
                0, 1
            )
            if i == 0:
                waypoints.extend(segment_waypoints)
            else:
                waypoints.extend(segment_waypoints[1:])  # Avoid duplicate waypoints
        
        return self._evaluate_route(waypoints, avoid_high_congestion=True)
    
    def _optimize_point_order(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Optimize the order of points to minimize total distance.
        Simple nearest-neighbor heuristic.
        """
        if len(points) <= 2:
            return points
        
        start = points[0]
        remaining = points[1:]
        optimized = [start]
        current = start
        
        while remaining:
            # Find nearest unvisited point
            nearest = min(
                remaining,
                key=lambda p: self._haversine_distance(
                    current[0], current[1], p[0], p[1]
                )
            )
            optimized.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return optimized
    
    def get_alternative_routes(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_routes: int = 3
    ) -> List[Route]:
        """
        Get multiple alternative routes between two points.
        
        Parameters
        ----------
        start : Tuple[float, float]
            Starting point
        end : Tuple[float, float]
            Destination
        num_routes : int
            Number of alternative routes to generate
        
        Returns
        -------
        List[Route]
            List of alternative routes
        """
        return self.find_optimal_route(start, end, num_alternatives=num_routes)
    
    def estimate_travel_time(
        self,
        route: Route,
        current_time: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Estimate travel time for a route, considering time of day.
        
        Parameters
        ----------
        route : Route
            Route to estimate
        current_time : Optional[str]
            Current time (ISO format) for time-based adjustments
        
        Returns
        -------
        Dict[str, float]
            Time estimates in minutes
        """
        base_time = route.estimated_time
        
        # Adjust for time of day if provided
        if current_time:
            from datetime import datetime
            dt = datetime.fromisoformat(current_time)
            hour = dt.hour
            
            # Rush hour multipliers
            if (8 <= hour <= 10) or (17 <= hour <= 19):
                multiplier = 1.3  # 30% slower during rush hour
            elif 22 <= hour or hour <= 6:
                multiplier = 0.9  # 10% faster during off-peak
            else:
                multiplier = 1.0
            
            adjusted_time = base_time * multiplier
        else:
            adjusted_time = base_time
        
        return {
            "base_time_minutes": round(base_time, 1),
            "adjusted_time_minutes": round(adjusted_time, 1),
            "congestion_penalty_minutes": round(adjusted_time - base_time, 1)
        }


def print_route_summary(routes: List[Route], title: str = "Route Options") -> None:
    """
    Print a formatted summary of route options.
    
    Parameters
    ----------
    routes : List[Route]
        List of routes to display
    title : str
        Title for the summary
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    
    for i, route in enumerate(routes, 1):
        congestion_emoji = {
            "Low": "🟢",
            "Medium": "🟡",
            "High": "🔴"
        }
        
        # Get dominant congestion level
        congestion_counts = {}
        for level in route.congestion_levels:
            congestion_counts[level] = congestion_counts.get(level, 0) + 1
        dominant = max(congestion_counts, key=congestion_counts.get)
        
        print(f"\n📍 Route {i} (Rank {route.alternative_rank}):")
        print(f"   Distance: {route.total_distance} km")
        print(f"   Estimated Time: {route.estimated_time} minutes")
        print(f"   Congestion Score: {route.congestion_score:.2f} {congestion_emoji.get(dominant, '⚪')}")
        print(f"   Dominant Congestion: {dominant}")
        print(f"   Waypoints: {len(route.waypoints)} points")
        
        if i < len(routes):
            print()

