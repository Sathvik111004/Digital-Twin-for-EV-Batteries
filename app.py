from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import time
import random
from datetime import datetime
import json

app = Flask(__name__)
 
# Load your trained models
try:
    rf_temp_model = joblib.load('battery_temperature_model.pkl')
    rf_soc_model = joblib.load('battery_soc_model.pkl')
    xgb_temp_model = joblib.load('temp_model.pkl')
    xgb_soc_model = joblib.load('soc_model.pkl')
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    rf_temp_model = rf_soc_model = xgb_temp_model = xgb_soc_model = None

class DigitalTwin:
    def __init__(self):
        self.simulation_data = []
        self.is_running = False
        
    def predict_temperature(self, ambient_temp, current, velocity, season):
        """Combined prediction using both RF and XGBoost"""
        try:
            # Random Forest prediction
            rf_features = [[ambient_temp, current, velocity, season, 
                          (current * 370) / 1000, 0 if velocity < 30 else (1 if velocity < 60 else 2)]]
            rf_pred = rf_temp_model.predict(rf_features)[0] if rf_temp_model else ambient_temp + 5
            
            # XGBoost prediction (with cleaned column names)
            xgb_features = pd.DataFrame([[ambient_temp, current, velocity, season, (current * 370) / 1000]], 
                                      columns=['Ambient_Temperature_degC', 'Battery_Current_A', 'Velocity_km/h', 'Season', 'Power_kW'])
            xgb_pred = xgb_temp_model.predict(xgb_features)[0] if xgb_temp_model else ambient_temp + 3
            
            # Ensemble: weighted average (RF performed better, so higher weight)
            ensemble_pred = (0.7 * rf_pred) + (0.3 * xgb_pred)
            return round(ensemble_pred, 1)
        except:
            return round(ambient_temp + 5, 1)  # Fallback
    
    def predict_soc(self, current, velocity, ambient_temp, time_hours, season):
        """Combined SoC prediction"""
        try:
            # Random Forest prediction
            rf_features = [[current, velocity, ambient_temp, time_hours, season, 
                          (current * 370) / 1000, 1 if velocity > 50 else 0]]
            rf_pred = rf_soc_model.predict(rf_features)[0] if rf_soc_model else 70
            
            # XGBoost prediction
            xgb_features = pd.DataFrame([[current, velocity, ambient_temp, season]], 
                                      columns=['Battery_Current_A', 'Velocity_km/h', 'Ambient_Temperature_degC', 'Season'])
            xgb_pred = xgb_soc_model.predict(xgb_features)[0] if xgb_soc_model else 65
            
            # Ensemble prediction
            ensemble_pred = (0.7 * rf_pred) + (0.3 * xgb_pred)
            return round(max(0, min(100, ensemble_pred)), 1)
        except:
            return 70.0  # Fallback
    
    def generate_simulation_data(self, scenario="normal"):
        """Generate realistic simulation data"""
        scenarios = {
            "normal": {"temp_range": (15, 25), "current_range": (-50, -20), "velocity_range": (30, 80)},
            "highway": {"temp_range": (20, 30), "current_range": (-80, -40), "velocity_range": (80, 120)},
            "city": {"temp_range": (10, 20), "current_range": (-30, -10), "velocity_range": (10, 50)},
            "charging": {"temp_range": (15, 25), "current_range": (10, 50), "velocity_range": (0, 5)},
            "winter": {"temp_range": (-5, 10), "current_range": (-60, -30), "velocity_range": (20, 70)}
        }
        
        params = scenarios.get(scenario, scenarios["normal"])
        
        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "ambient_temp": round(random.uniform(*params["temp_range"]), 1),
            "current": round(random.uniform(*params["current_range"]), 1),
            "velocity": round(random.uniform(*params["velocity_range"]), 1),
            "season": 1 if scenario != "winter" else 0
        }

# Initialize Digital Twin
twin = DigitalTwin()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    """Handle manual input predictions"""
    data = request.json
    
    ambient_temp = float(data.get('ambient_temp', 20))
    current = float(data.get('current', -40))
    velocity = float(data.get('velocity', 50))
    season = int(data.get('season', 1))
    
    # Make predictions
    battery_temp = twin.predict_temperature(ambient_temp, current, velocity, season)
    soc = twin.predict_soc(current, velocity, ambient_temp, 0.5, season)
    
    # Calculate additional metrics
    power_kw = round((current * 370) / 1000, 2)
    efficiency = round(max(70, 100 - abs(current) * 0.1), 1)
    
    return jsonify({
        "battery_temp": battery_temp,
        "soc": soc,
        "power_kw": power_kw,
        "efficiency": efficiency,
        "status": "success"
    })

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """Start real-time simulation"""
    data = request.json
    scenario = data.get('scenario', 'normal')
    
    twin.is_running = True
    twin.simulation_data = []
    
    return jsonify({"status": "simulation_started", "scenario": scenario})

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop simulation"""
    twin.is_running = False
    return jsonify({"status": "simulation_stopped"})

@app.route('/get_simulation_data')
def get_simulation_data():
    """Get latest simulation data point"""
    if not twin.is_running:
        return jsonify({"status": "stopped"})
    
    # Generate new data point
    scenario = request.args.get('scenario', 'normal')
    sim_data = twin.generate_simulation_data(scenario)
    
    # Make predictions
    battery_temp = twin.predict_temperature(
        sim_data['ambient_temp'], sim_data['current'], 
        sim_data['velocity'], sim_data['season']
    )
    soc = twin.predict_soc(
        sim_data['current'], sim_data['velocity'], 
        sim_data['ambient_temp'], 0.1, sim_data['season']
    )
    
    # Calculate metrics
    power_kw = round((sim_data['current'] * 370) / 1000, 2)
    efficiency = round(max(70, 100 - abs(sim_data['current']) * 0.1), 1)
    
    result = {
        **sim_data,
        "battery_temp": battery_temp,
        "soc": soc,
        "power_kw": power_kw,
        "efficiency": efficiency,
        "status": "running"
    }
    
    # Store data point
    twin.simulation_data.append(result)
    
    # Keep only last 50 points
    if len(twin.simulation_data) > 50:
        twin.simulation_data = twin.simulation_data[-50:]
    
    return jsonify(result)

@app.route('/get_history')
def get_history():
    """Get simulation history for charts"""
    return jsonify(twin.simulation_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)