import requests
import numpy as np
import pandas as pd
import math
import torch
import sys
import os
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler

# Add the parent directory to path to import from main2.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model class from main2.py
from main2 import PlumeNetDeep

# Configuration
API_KEY = "d8f119041413d028c00ae60ded02231a"
GEO_URL = "http://api.openweathermap.org/geo/1.0/zip"
AIR_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"
ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

# Load the neighbor coordinates
def load_neighbor_coordinates():
    """Load the predefined neighbor coordinates from the CSV file"""
    df = pd.read_csv("multi_zone_ring_layout_relative_coords.csv")
    neighbors = []
    for _, row in df.iterrows():
        neighbors.append({
            'radius_km': row['Radius (km)'],
            'angle_rad': row['Angle (radians)'],
            'delta_lat': row['Delta Latitude'],
            'delta_lon': row['Delta Longitude']
        })
    return neighbors

def get_elevation(lat: float, lon: float) -> float:
    """Get elevation for given coordinates using Open-Elevation API"""
    url = f"{ELEVATION_URL}?locations={lat},{lon}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['results'][0]['elevation']
    else:
        print(f"Error getting elevation: {response.status_code}")
        return 0.0

def get_pm25(lat: float, lon: float) -> float:
    """Get current PM2.5 for given coordinates using OpenWeatherMap API"""
    air_params = {'lat': lat, 'lon': lon, 'appid': API_KEY}
    air_resp = requests.get(AIR_URL, params=air_params)
    if air_resp.status_code == 200:
        air_data = air_resp.json()
        return air_data['list'][0]['components']['pm2_5']
    else:
        print(f"Error getting PM2.5: {air_resp.status_code}")
        return 0.0

def get_wind_data(lat: float, lon: float) -> Tuple[float, float, float, float]:
    """Get wind data and convert to U/V components"""
    weather_params = {'lat': lat, 'lon': lon, 'appid': API_KEY}
    weather_resp = requests.get(WEATHER_URL, params=weather_params)
    
    if weather_resp.status_code == 200:
        weather_data = weather_resp.json()
        wind_speed = weather_data['wind']['speed']  # m/s
        wind_direction = weather_data['wind']['deg']  # degrees
        
        # Convert to U/V components
        wind_dir_rad = math.radians(wind_direction)
        u = -wind_speed * math.sin(wind_dir_rad)
        v = -wind_speed * math.cos(wind_dir_rad)
        
        # Assume same values for 10m and 50m heights
        return u, v, u, v
    else:
        print(f"Error getting wind data: {weather_resp.status_code}")
        return 0.0, 0.0, 0.0, 0.0

def gaussian_plume_estimate(source_coords: Tuple[float, float], 
                           point_coords: Tuple[float, float], 
                           u: float, v: float, 
                           Q=1000, H=10, sigma_y=50, sigma_z=20) -> float:
    """Calculate Gaussian plume estimate"""
    dx = point_coords[0] - source_coords[0]
    dy = point_coords[1] - source_coords[1]
    
    wind_mag = math.sqrt(u**2 + v**2) + 1e-6
    wind_dir_x = u / wind_mag
    wind_dir_y = v / wind_mag
    
    x = dx * wind_dir_x + dy * wind_dir_y  # downwind distance
    y = -dx * wind_dir_y + dy * wind_dir_x  # crosswind
    
    if x <= 0:
        return 0.0  # upwind or no impact
    
    exponent = -0.5 * (y / sigma_y) ** 2
    vertical = math.exp(-0.5 * (H / sigma_z) ** 2)
    denom = 2 * math.pi * wind_mag * sigma_y * sigma_z
    C = (Q / denom) * math.exp(exponent) * vertical
    return C

def get_center_features(center_lat: float, center_lon: float) -> List[float]:
    """Get the 8 center features for the main location"""
    # 1. Current PM2.5
    current_pm25 = get_pm25(center_lat, center_lon)
    
    # 2. Plume prediction - calculate impact of all neighbors on center point
    wind_u10m, wind_v10m, wind_u50m, wind_v50m = get_wind_data(center_lat, center_lon)
    
    # Load neighbor coordinates to calculate plume impact
    neighbors = load_neighbor_coordinates()
    total_plume_impact = 0.0
    
    for neighbor in neighbors:
        neighbor_lat = center_lat + neighbor['delta_lat']
        neighbor_lon = center_lon + neighbor['delta_lon']
        
        # Get PM2.5 at neighbor location
        neighbor_pm25 = get_pm25(neighbor_lat, neighbor_lon)
        
        # Calculate plume impact from this neighbor to center point
        plume_impact = gaussian_plume_estimate(
            (neighbor_lon, neighbor_lat),  # source (neighbor)
            (center_lon, center_lat),      # target (center)
            (wind_u10m + wind_u50m) / 2, (wind_v10m + wind_v50m) / 2
        )
        
        # Weight by neighbor's PM2.5 concentration
        total_plume_impact += neighbor_pm25 * plume_impact
    
    plume_pred = total_plume_impact
    
    # 7. Elevation
    elevation = get_elevation(center_lat, center_lon)
    
    # 8. NLCD (static value as requested)
    nlcd = 250
    
    return [current_pm25, plume_pred, wind_u10m, wind_v10m, wind_u50m, wind_v50m, elevation, nlcd]

def get_neighbor_features(center_lat: float, center_lon: float, 
                         neighbor_delta_lat: float, neighbor_delta_lon: float) -> List[float]:
    """Get the 10 context features for a neighbor location"""
    # Calculate neighbor coordinates
    neighbor_lat = center_lat + neighbor_delta_lat
    neighbor_lon = center_lon + neighbor_delta_lon
    
    # 1-2. Delta lat/lon (relative to center)
    delta_lat = neighbor_delta_lat
    delta_lon = neighbor_delta_lon
    
    # 3-6. Wind U/V components
    wind_u10m, wind_v10m, wind_u50m, wind_v50m = get_wind_data(neighbor_lat, neighbor_lon)
    
    # 7. Elevation
    elevation = get_elevation(neighbor_lat, neighbor_lon)
    
    # 8. NLCD (static value)
    nlcd = 250
    
    # 9. PM2.5 at neighbor location
    pm25 = get_pm25(neighbor_lat, neighbor_lon)
    
    # 10. Plume prediction for neighbor
    plume = gaussian_plume_estimate(
        (center_lon, center_lat), (neighbor_lon, neighbor_lat),
        (wind_u10m + wind_u50m) / 2, (wind_v10m + wind_v50m) / 2
    )
    
    return [delta_lat, delta_lon, wind_u10m, wind_v10m, wind_u50m, wind_v50m, elevation, nlcd, pm25, plume]

def load_models():
    """Load the trained models for 1, 3, and 7 day predictions"""
    device = torch.device("cpu")
    models = {}
    
    for horizon in [1, 3, 7]:
        model = PlumeNetDeep().to(device)
        model_path = f"../finals/plumeNet{horizon}_improved.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[horizon] = model
        print(f"✅ Loaded {horizon}-day model from {model_path}")
    
    return models

def predict_pm25(center_lat: float, center_lon: float, models: Dict):
    """Make PM2.5 predictions for 1, 3, and 7 days"""
    print(f"\n🌬️ Making PM2.5 predictions for location ({center_lat}, {center_lon})")
    print("=" * 60)
    
    # Load neighbor coordinates
    neighbors = load_neighbor_coordinates()
    
    # Get center features
    center_features = get_center_features(center_lat, center_lon)
    
    # Debug: Print raw features
    feature_names = ["current_pm25", "plume_pred", "wind_u10m", "wind_v10m", "wind_u50m", "wind_v50m", "elevation", "nlcd"]
    print("Raw center features:")
    for i, (name, value) in enumerate(zip(feature_names, center_features)):
        print(f"  {name}: {value:.3f}")
    
    # Scale center features using reasonable scaling values
    # These are approximate values based on typical ranges in the training data
    scaling_params = {
        "current_pm25": {"mean": 15.0, "std": 20.0},
        "plume_pred": {"mean": 0.5, "std": 2.0},
        "wind_u10m": {"mean": 0.0, "std": 5.0},
        "wind_v10m": {"mean": 0.0, "std": 5.0},
        "wind_u50m": {"mean": 0.0, "std": 5.0},
        "wind_v50m": {"mean": 0.0, "std": 5.0},
        "elevation": {"mean": 500.0, "std": 1000.0},
        "nlcd": {"mean": 50.0, "std": 30.0}
    }
    
    center_features_scaled = []
    for i, (name, value) in enumerate(zip(feature_names, center_features)):
        params = scaling_params[name]
        scaled_value = (value - params["mean"]) / params["std"]
        center_features_scaled.append(scaled_value)
    
    print("\nScaled center features:")
    for i, (name, value) in enumerate(zip(feature_names, center_features_scaled)):
        print(f"  {name}: {value:.3f}")
    
    center_tensor = torch.tensor([center_features_scaled], dtype=torch.float32)
    
    # Get context features for all neighbors
    context_features = []
    for neighbor in neighbors:
        features = get_neighbor_features(
            center_lat, center_lon, 
            neighbor['delta_lat'], neighbor['delta_lon']
        )
        context_features.append(features)
    
    # Pad to max_neighbors=20 if needed
    max_neighbors = 20
    while len(context_features) < max_neighbors:
        context_features.append([0.0] * 10)  # Zero padding
    
    # Normalize context features (like in training)
    context_features_array = np.array(context_features, dtype=np.float32)
    context_features_normalized = (context_features_array - context_features_array.mean(axis=0)) / (context_features_array.std(axis=0) + 1e-6)
    
    context_tensor = torch.tensor([context_features_normalized], dtype=torch.float32)
    
    # Make predictions for each horizon
    predictions = {}
    with torch.no_grad():
        for horizon, model in models.items():
            prediction = model(center_tensor, context_tensor)
            predictions[horizon] = prediction.item()
            print(f"📊 {horizon}-day PM2.5 prediction: {prediction.item():.3f} μg/m³")
    
    return predictions

def main():
    """Main function to demonstrate PM2.5 predictions"""
    # Test coordinates (New York City)
    center_lat = 40.7239
    center_lon = -74.3072
    
    print("🌬️ PM2.5 PREDICTION SYSTEM")
    print("=" * 60)
    
    try:
        # Load the trained models
        print("🤖 Loading trained models...")
        models = load_models()
        
        # Make predictions
        predictions = predict_pm25(center_lat, center_lon, models)
        
        print(f"\n🎯 PREDICTION RESULTS")
        print("=" * 60)
        print(f"📍 Location: ({center_lat}, {center_lon})")
        print(f"📅 Current PM2.5: {get_pm25(center_lat, center_lon):.3f} μg/m³")
        print()
        print("🔮 Future PM2.5 Predictions:")
        for horizon, pred in predictions.items():
            print(f"   {horizon}-day: {pred:.3f} μg/m³")
        
        print(f"\n✅ All predictions completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()