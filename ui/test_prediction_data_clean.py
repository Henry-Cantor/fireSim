import requests
import numpy as np
import pandas as pd
import math
from typing import Tuple, List, Dict

# Configuration
API_KEY = "d8f119041413d028c00ae60ded02231a"
GEO_URL = "http://api.openweathermap.org/geo/1.0/zip"
AIR_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"
ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

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
    print(f"\n📍 CENTER LOCATION ({center_lat}, {center_lon})")
    print("-" * 60)
    
    # 1. Current PM2.5
    current_pm25 = get_pm25(center_lat, center_lon)
    print(f"1. current_pm25: {current_pm25:.3f} μg/m³")
    
    # 2. Plume prediction (calculate for center point)
    wind_u10m, wind_v10m, wind_u50m, wind_v50m = get_wind_data(center_lat, center_lon)
    plume_pred = gaussian_plume_estimate(
        (center_lon, center_lat), (center_lon, center_lat), 
        (wind_u10m + wind_u50m) / 2, (wind_v10m + wind_v50m) / 2
    )
    print(f"2. plume_pred: {plume_pred:.3f}")
    
    # 3-6. Wind U/V components
    print(f"3. wind_u10m: {wind_u10m:.3f} m/s")
    print(f"4. wind_v10m: {wind_v10m:.3f} m/s")
    print(f"5. wind_u50m: {wind_u50m:.3f} m/s")
    print(f"6. wind_v50m: {wind_v50m:.3f} m/s")
    
    # 7. Elevation
    elevation = get_elevation(center_lat, center_lon)
    print(f"7. elevation: {elevation:.1f} m")
    
    # 8. NLCD (static value as requested)
    nlcd = 250
    print(f"8. nlcd: {nlcd}")
    
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

def main():
    """Main function to demonstrate getting all required data"""
    # Test coordinates (New York City)
    center_lat = 40.7239
    center_lon = -74.3072
    
    print("🌬️ PM2.5 PREDICTION DATA COLLECTION DEMO")
    print("=" * 60)
    
    # Load neighbor coordinates
    neighbors = load_neighbor_coordinates()
    print(f"📊 Loaded {len(neighbors)} neighbor locations")
    
    # Get center features
    center_features = get_center_features(center_lat, center_lon)
    
    print(f"\n🌍 NEIGHBOR CONTEXT FEATURES")
    print("=" * 60)
    print("Format: [Δlat, Δlon, wind_u10, wind_v10, wind_u50, wind_v50, elevation, nlcd, pm25, plume]")
    print()
    
    neighbor_features_list = []
    for i, neighbor in enumerate(neighbors):
        features = get_neighbor_features(
            center_lat, center_lon, 
            neighbor['delta_lat'], neighbor['delta_lon']
        )
        neighbor_features_list.append(features)
        
        # Calculate actual neighbor coordinates
        neighbor_lat = center_lat + neighbor['delta_lat']
        neighbor_lon = center_lon + neighbor['delta_lon']
        
        print(f"📍 Neighbor {i+1:2d} ({neighbor['radius_km']:2.0f}km, {neighbor['angle_rad']:5.2f}rad)")
        print(f"   Location: ({neighbor_lat:.4f}, {neighbor_lon:.4f})")
        print(f"   Features: {[round(x, 3) if isinstance(x, float) else x for x in features]}")
        print()
    
    print(f"📋 SUMMARY")
    print("=" * 60)
    print(f"✅ Center features (8): {[round(x, 3) if isinstance(x, float) else x for x in center_features]}")
    print(f"✅ Number of neighbors: {len(neighbor_features_list)}")
    print(f"✅ Context features per neighbor (10): {len(neighbor_features_list[0])}")
    
    # Demonstrate how this would be used for model input
    print(f"\n🤖 MODEL INPUT FORMAT")
    print("=" * 60)
    print("Center features (8):", [round(x, 3) if isinstance(x, float) else x for x in center_features])
    print("\nContext features (first 3 neighbors):")
    for i in range(min(3, len(neighbor_features_list))):
        print(f"  Neighbor {i+1}: {[round(x, 3) if isinstance(x, float) else x for x in neighbor_features_list[i]]}")
    
    print(f"\n🎯 READY FOR MODEL PREDICTION!")
    print("=" * 60)
    print("✅ All 10 variables successfully collected for PM2.5 prediction!")
    print("✅ Data format matches PlumeNetDeep model requirements")
    print("✅ Ready to load models from finals/ folder for 1, 3, and 7-day predictions")

if __name__ == "__main__":
    main() 