from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import numpy as np
import pandas as pd
import math
import torch
import sys
import os
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main2 import PlumeNetDeep

app = Flask(__name__)
CORS(app)

API_KEY = "d8f119041413d028c00ae60ded02231a"
GEO_URL = "http://api.openweathermap.org/geo/1.0/zip"
AIR_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"
ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

models = None

def load_models():
    global models
    if models is not None:
        return models
    
    device = torch.device("cpu")
    models = {}
    
    for horizon in [1, 3, 7]:
        model = PlumeNetDeep().to(device)
        model_path = f"../finals/plumeNet{horizon}_improved.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[horizon] = model
    
    return models

def load_neighbor_coordinates():
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
    url = f"{ELEVATION_URL}?locations={lat},{lon}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['results'][0]['elevation']
    else:
        return 0.0

def fetch_pm25_data(lat: float, lon: float) -> float:
    air_params = {'lat': lat, 'lon': lon, 'appid': API_KEY}
    air_resp = requests.get(AIR_URL, params=air_params)
    if air_resp.status_code == 200:
        air_data = air_resp.json()
        return air_data['list'][0]['components']['pm2_5']
    else:
        return 0.0

def get_wind_data(lat: float, lon: float) -> Tuple[float, float, float, float]:
    """Get wind data and convert to U/V components"""
    weather_params = {'lat': lat, 'lon': lon, 'appid': API_KEY}
    weather_resp = requests.get(WEATHER_URL, params=weather_params)
    
    if weather_resp.status_code == 200:
        weather_data = weather_resp.json()
        wind_speed = weather_data['wind']['speed'] 
        wind_direction = weather_data['wind']['deg'] 
        
        wind_dir_rad = math.radians(wind_direction)
        u = -wind_speed * math.sin(wind_dir_rad)
        v = -wind_speed * math.cos(wind_dir_rad)
        
        return u, v, u, v
    else:
        return 0.0, 0.0, 0.0, 0.0

def gaussian_plume_estimate(source_coords: Tuple[float, float], 
                           point_coords: Tuple[float, float], 
                           u: float, v: float, 
                           Q=1000, H=10, sigma_y=50, sigma_z=20) -> float:
    dx = point_coords[0] - source_coords[0]
    dy = point_coords[1] - source_coords[1]
    
    wind_mag = math.sqrt(u**2 + v**2) + 1e-6
    wind_dir_x = u / wind_mag
    wind_dir_y = v / wind_mag
    
    x = dx * wind_dir_x + dy * wind_dir_y  
    y = -dx * wind_dir_y + dy * wind_dir_x 
    
    if x <= 0:
        return 0.0 
    
    exponent = -0.5 * (y / sigma_y) ** 2
    vertical = math.exp(-0.5 * (H / sigma_z) ** 2)
    denom = 2 * math.pi * wind_mag * sigma_y * sigma_z
    C = (Q / denom) * np.exp(exponent) * vertical
    return C

def get_center_features(center_lat: float, center_lon: float) -> List[float]:
    """Get the 8 center features for the main location"""
   
    current_pm25 = fetch_pm25_data(center_lat, center_lon)
    
  
    wind_u10m, wind_v10m, wind_u50m, wind_v50m = get_wind_data(center_lat, center_lon)
    
    neighbors = load_neighbor_coordinates()
    total_plume_impact = 0.0
    
    for neighbor in neighbors:
        neighbor_lat = center_lat + neighbor['delta_lat']
        neighbor_lon = center_lon + neighbor['delta_lon']
        
        neighbor_pm25 = fetch_pm25_data(neighbor_lat, neighbor_lon)
        
       
        plume_impact = gaussian_plume_estimate(
            (neighbor_lon, neighbor_lat),  # source (neighbor)
            (center_lon, center_lat),      # target (center)
            (wind_u10m + wind_u50m) / 2, (wind_v10m + wind_v50m) / 2
        )
        
       
        total_plume_impact += neighbor_pm25 * plume_impact
    
    plume_pred = total_plume_impact
    
    elevation = get_elevation(center_lat, center_lon)
    
   
    nlcd = 250
    
    return [current_pm25, plume_pred, wind_u10m, wind_v10m, wind_u50m, wind_v50m, elevation, nlcd]

def get_neighbor_features(center_lat: float, center_lon: float, 
                         neighbor_delta_lat: float, neighbor_delta_lon: float) -> List[float]:

    neighbor_lat = center_lat + neighbor_delta_lat
    neighbor_lon = center_lon + neighbor_delta_lon
    
    delta_lat = neighbor_delta_lat
    delta_lon = neighbor_delta_lon
    
    wind_u10m, wind_v10m, wind_u50m, wind_v50m = get_wind_data(neighbor_lat, neighbor_lon)
    
    elevation = get_elevation(neighbor_lat, neighbor_lon)
    
    nlcd = 250
    
    pm25 = fetch_pm25_data(neighbor_lat, neighbor_lon)
    
    plume = gaussian_plume_estimate(
        (center_lon, center_lat), (neighbor_lon, neighbor_lat),
        (wind_u10m + wind_u50m) / 2, (wind_v10m + wind_v50m) / 2
    )
    
    return [delta_lat, delta_lon, wind_u10m, wind_v10m, wind_u50m, wind_v50m, elevation, nlcd, pm25, plume]

def get_pm25_danger_level(pm25_value: float) -> Dict:
    if pm25_value <= 12.0:
        return {"level": "Good", "color": "green", "risk": "Low"}
    elif pm25_value <= 35.4:
        return {"level": "Moderate", "color": "yellow", "risk": "Moderate"}
    elif pm25_value <= 55.4:
        return {"level": "Unhealthy for Sensitive Groups", "color": "orange", "risk": "High"}
    elif pm25_value <= 150.4:
        return {"level": "Unhealthy", "color": "red", "risk": "Very High"}
    elif pm25_value <= 250.4:
        return {"level": "Very Unhealthy", "color": "purple", "risk": "Hazardous"}
    else:
        return {"level": "Hazardous", "color": "maroon", "risk": "Extreme"}

def predict_pm25(center_lat: float, center_lon: float) -> Dict:
    models = load_models()
    
    neighbors = load_neighbor_coordinates()
    
    center_features = get_center_features(center_lat, center_lon)
    
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
    
    feature_names = ["current_pm25", "plume_pred", "wind_u10m", "wind_v10m", "wind_u50m", "wind_v50m", "elevation", "nlcd"]
    center_features_scaled = []
    for i, (name, value) in enumerate(zip(feature_names, center_features)):
        params = scaling_params[name]
        scaled_value = (value - params["mean"]) / params["std"]
        center_features_scaled.append(scaled_value)
    
    center_tensor = torch.tensor([center_features_scaled], dtype=torch.float32)
    
    context_features = []
    for neighbor in neighbors:
        features = get_neighbor_features(
            center_lat, center_lon, 
            neighbor['delta_lat'], neighbor['delta_lon']
        )
        context_features.append(features)
    
    max_neighbors = 20
    while len(context_features) < max_neighbors:
        context_features.append([0.0] * 10) 
    
    context_features_array = np.array(context_features, dtype=np.float32)
    context_features_normalized = (context_features_array - context_features_array.mean(axis=0)) / (context_features_array.std(axis=0) + 1e-6)
    
    context_tensor = torch.tensor([context_features_normalized], dtype=torch.float32)
    
    predictions = {}
    with torch.no_grad():
        for horizon, model in models.items():
            prediction = model(center_tensor, context_tensor)
            predictions[horizon] = prediction.item()
    
    return predictions

@app.route('/pm25', methods=['GET'])
def get_pm25():
    zipcode = request.args.get('zipcode')
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is not None and lon is not None:
     
        use_lat, use_lon = lat, lon
    elif zipcode:
        geo_params = {'zip': f'{zipcode},US', 'appid': API_KEY}
        geo_resp = requests.get(GEO_URL, params=geo_params)
        if geo_resp.status_code != 200:
            return jsonify({'error': 'Invalid zipcode or geo API error'}), 400
        geo_data = geo_resp.json()
        use_lat = geo_data.get('lat')
        use_lon = geo_data.get('lon')
        if use_lat is None or use_lon is None:
            return jsonify({'error': 'Could not find coordinates for zipcode'}), 400
    else:
        return jsonify({'error': 'Missing zipcode or lat/lon'}), 400
 
    air_params = {'lat': use_lat, 'lon': use_lon, 'appid': API_KEY}
    air_resp = requests.get(AIR_URL, params=air_params)
    if air_resp.status_code != 200:
        return jsonify({'error': 'Air pollution API error'}), 500
    air_data = air_resp.json()
    try:
        pm25 = air_data['list'][0]['components']['pm2_5']
    except (KeyError, IndexError):
        return jsonify({'error': 'Could not retrieve PM2.5 data'}), 500
    return jsonify({'pm2_5': pm25})

@app.route('/predict', methods=['GET'])
def predict():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if lat is None or lon is None:
        return jsonify({'error': 'Missing lat/lon parameters'}), 400
    
    try:
        current_pm25 = fetch_pm25_data(lat, lon)
        
        predictions = predict_pm25(lat, lon)
        
        result = {
            'current': {
                'value': round(current_pm25, 2),
                'danger': get_pm25_danger_level(current_pm25)
            },
            'predictions': {}
        }
        
        for horizon, pred_value in predictions.items():
            result['predictions'][f'{horizon}_day'] = {
                'value': round(pred_value, 2),
                'danger': get_pm25_danger_level(pred_value)
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500) 