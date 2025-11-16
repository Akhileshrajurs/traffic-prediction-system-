"""
Entry point for the Traffic Congestion Prediction mini-project.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from gps_input import load_model, simulate_balanced_gps_points, predict_congestion
from map_display import create_map
from analysis import generate_statistics, print_statistics_report
from visualizations import create_statistical_dashboard
from export_utils import export_predictions_to_csv, export_predictions_to_json, export_statistics_to_json
from data_preprocessing import prepare_dataset
from real_time_data import RealTimeDataFetcher, merge_data_sources
from route_optimization import RouteOptimizer, print_route_summary

MODEL_PATH = Path("models") / "traffic_congestion_model.joblib"
MAP_OUTPUT = Path("traffic_map.html")
DATA_PATH = Path("data") / "traffic_data.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run 'python src/model_training.py' first."
        )

    model_bundle = load_model(MODEL_PATH)

    current_hour = datetime.now().hour

    # Generate 10 points for each congestion level (Low, Medium, High) = 30 total
    simulated_points = simulate_balanced_gps_points(
        num_per_level=10,
        bounds=model_bundle.training_data_bounds,
    )

    predictions = predict_congestion(
        model_bundle, 
        simulated_points, 
        hour_of_day=current_hour,
        include_confidence=True
    )

    print("Live Congestion Predictions")
    print("---------------------------")
    for idx, result in enumerate(predictions, start=1):
        congestion = result['congestion_level']
        lat = result['latitude']
        lon = result['longitude']
        speed = result['speed(kmph)']
        vehicles = result['vehicle_count']
        confidence = result.get('confidence', 0)
        conf_str = f" (Confidence: {confidence:.2%})" if confidence > 0 else ""
        print(f"{idx}. Location ({lat:.6f}, {lon:.6f}) - Speed: {speed:.2f} km/h - Vehicles: {vehicles} - Congestion: {congestion}{conf_str}")

    # Count predictions by congestion level
    from collections import Counter
    congestion_counts = Counter(p['congestion_level'] for p in predictions)
    
    print("\n" + "="*50)
    print("Congestion Level Summary:")
    print("="*50)
    for level in ["Low", "Medium", "High"]:
        count = congestion_counts.get(level, 0)
        color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(level, "⚪")
        print(f"{color} {level}: {count} locations")
    print("="*50)

    # Generate and display statistics
    _, _, _, processed_df = prepare_dataset(str(DATA_PATH))
    stats = generate_statistics(predictions, processed_df)
    print_statistics_report(stats)
    
    # Create enhanced map
    create_map(predictions, MAP_OUTPUT, add_heatmap=True, add_legend=True)
    print(f"\n✅ Interactive map saved to: {MAP_OUTPUT.resolve()}")
    
    # Create statistical dashboard
    dashboard_dir = Path("reports")
    create_statistical_dashboard(predictions, processed_df, dashboard_dir)
    
    # Export predictions
    export_dir = Path("exports")
    export_predictions_to_csv(predictions, export_dir / "predictions.csv")
    export_predictions_to_json(predictions, export_dir / "predictions.json")
    export_statistics_to_json(stats, export_dir / "statistics.json")
    
    print(f"\n✅ Model accuracy: {model_bundle.accuracy:.2%}")
    print(f"✅ All outputs saved successfully!")
    
    # Real-time data integration demo
    print("\n" + "="*70)
    print("  REAL-TIME DATA INTEGRATION DEMO")
    print("="*70)
    
    fetcher = RealTimeDataFetcher()
    
    # Fetch live GPS data
    print("\n📡 Fetching live GPS data...")
    live_data = fetcher.fetch_live_gps_data(
        bounds=model_bundle.training_data_bounds,
        num_points=5,
        use_api=False
    )
    print(f"✅ Fetched {len(live_data)} live GPS points")
    
    # Simulate web scraping
    print("\n🌐 Simulating web scraping for traffic data...")
    scraped_data = fetcher.scrape_traffic_data(location="Bangalore", use_real_scraping=False)
    print(f"✅ Scraped {len(scraped_data)} data points")
    
    # Fetch historical data
    print("\n📚 Fetching historical data (last 24 hours)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    historical_df = fetcher.fetch_historical_data(
        start_date=start_date,
        end_date=end_date,
        bounds=model_bundle.training_data_bounds,
        use_api=False
    )
    print(f"✅ Fetched {len(historical_df)} historical data points")
    
    # Merge all data sources
    merged_data = merge_data_sources([live_data, scraped_data, historical_df.to_dict('records')])
    print(f"\n✅ Merged {len(merged_data)} total data points from all sources")
    
    # Route optimization demo
    print("\n" + "="*70)
    print("  ROUTE OPTIMIZATION DEMO")
    print("="*70)
    
    # Use predictions for route optimization
    optimizer = RouteOptimizer(predictions)
    
    # Example: Route from first to last prediction point
    if len(predictions) >= 2:
        start_point = (predictions[0]['latitude'], predictions[0]['longitude'])
        end_point = (predictions[-1]['latitude'], predictions[-1]['longitude'])
        
        print(f"\n🗺️  Finding optimal route from:")
        print(f"   Start: ({start_point[0]:.6f}, {start_point[1]:.6f})")
        print(f"   End: ({end_point[0]:.6f}, {end_point[1]:.6f})")
        
        # Find optimal routes
        routes = optimizer.find_optimal_route(
            start=start_point,
            end=end_point,
            num_alternatives=3,
            avoid_high_congestion=True
        )
        
        print_route_summary(routes, "Optimal Route Alternatives")
        
        # Get travel time estimate
        best_route = routes[0]
        time_estimate = optimizer.estimate_travel_time(
            best_route,
            current_time=datetime.now().isoformat()
        )
        
        print(f"\n⏱️  Travel Time Estimate for Best Route:")
        print(f"   Base Time: {time_estimate['base_time_minutes']} minutes")
        print(f"   Adjusted Time: {time_estimate['adjusted_time_minutes']} minutes")
        if time_estimate['congestion_penalty_minutes'] > 0:
            print(f"   Congestion Penalty: +{time_estimate['congestion_penalty_minutes']} minutes")
        
        # Multi-point route example
        if len(predictions) >= 4:
            print(f"\n🗺️  Multi-Point Route Planning...")
            multi_points = [
                (predictions[i]['latitude'], predictions[i]['longitude'])
                for i in [0, len(predictions)//3, 2*len(predictions)//3, -1]
            ]
            
            multi_route = optimizer.plan_multi_point_route(
                points=multi_points,
                optimize_order=True
            )
            
            print(f"\n📍 Multi-Point Route:")
            print(f"   Total Distance: {multi_route.total_distance} km")
            print(f"   Estimated Time: {multi_route.estimated_time} minutes")
            print(f"   Congestion Score: {multi_route.congestion_score:.2f}")
            print(f"   Points Visited: {len(multi_points)}")
    
    print("\n" + "="*70)
    print("✅ All features completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()


