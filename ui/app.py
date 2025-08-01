from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = "d8f119041413d028c00ae60ded02231a"
GEO_URL = "http://api.openweathermap.org/geo/1.0/zip"
AIR_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

@app.route('/pm25', methods=['GET'])
def get_pm25():
    zipcode = request.args.get('zipcode')
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is not None and lon is not None:
        # Use provided coordinates
        use_lat, use_lon = lat, lon
    elif zipcode:
        # Step 1: Get coordinates from zipcode
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
    # Step 2: Get PM2.5
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500) 