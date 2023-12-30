import requests
import Secrets
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import joblib

from datetime import datetime
from dateutil import parser

# Home Assistant details
home_assistant_url = 'http://homeassistant.local:8123'
access_token = Secrets.access_token

# Neural network model path
model_path = 'model.keras'
scaler_path = 'scaler.pkl'

entity_id_lookup = {
    'sun.sun': 4,
    'weather.forecast_home': 16,
    'switch.tradfri_outlet': 62,
    'switch.tradfri_outlet_2': 64,
    'fan.tradfri_outlet_2': 65,
    'sensor.karolis_iphone_xr_distance': 66,
    'sensor.karolis_iphone_xr_floors_ascended': 67,
    'sensor.karolis_iphone_xr_floors_descended': 68,
    'sensor.karolis_iphone_xr_steps': 69,
    'sensor.karolis_iphone_xr_average_active_pace': 70,
    'sensor.karolis_iphone_xr_battery_level': 71,
    'sensor.karolis_iphone_xr_battery_state': 72,
    'sensor.karolis_iphone_xr_storage': 73,
    'light.tradfri_bulb_3': 76,
    'light.tradfri_bulb_2': 77,
    'light.tradfri_bulb_4': 78,
    'sensor.tradfri_on_off_switch_2_battery': 79,
    'sensor.karolis_iphone_xr_activity': 80,
    'light.tradfri_gunnarp': 81,
    'binary_sensor.karolis_iphone_xr_focus': 82,
    'sensor.tradfri_on_off_switch_4_battery': 83,
    'switch.kitchen_light_bar': 84,
    'sensor.tradfri_remote_control_battery': 85,
    'update.home_assistant_supervisor_update': 86,
    'light.pantry': 87,
    'sensor.karolis_iphone_xr_connection_type': 88
}

state_lookup = {
    'on': 1,
    'off': 0,
    'unavailable':2,
    'Unknown':0,

    'above_horizon':1,
    'below_horizon':0,

    'Not Charging':0,
    'Charging':1,
    'Full':2,

    'rainy':0,
    'cloudy':1,
    'fog':2,
    'snowy-rainy':3,
    'snowy':4,
    'partlycloudy':5,

    'Wi-Fi':0,
    'Cellular':1,

    'Stationary':0   
}

device_id_lookup = [62,63,64,65,76,77,78,81,84,87]

def map_or_parse_data(value):
    try:
        return float(value)
    except ValueError:
        return state_lookup.get(value, value)

# Function to fetch sensor data from Home Assistant
def get_sensor_data(url, token):
    headers = {'Authorization': 'Bearer ' + token, 'content-type': 'application/json'}
    response = requests.get(f"{url}/api/states", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return []

# Function to preprocess sensor data
def preprocess_data(sensor_data):
    # Convert the raw sensor data into a DataFrame
    data = pd.DataFrame(sensor_data)
    #Cleanup
    data.drop(['attributes','last_changed','context'], axis=1, inplace=True)

    #Timestamp conversion
    data['last_updated'] = data['last_updated'].apply(lambda x: parser.isoparse(x))
    
    hours = data['last_updated'].dt.hour.astype(str)
    minutes = data['last_updated'].dt.minute.astype(str)
    #Cleanup
    data['timestamp'] =hours+'.'+minutes
    data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
    data.drop('last_updated', axis=1, inplace=True)

    #Map various states to values
    data['entity_id'] = data['entity_id'].map(entity_id_lookup)
    data = data.dropna(subset=['entity_id'])
    data['state'] = data['state'].apply(map_or_parse_data)

    pivoted_data = data.pivot(index='timestamp', columns='entity_id', values='state')

    pivoted_data.fillna(method='ffill', inplace=True)
    pivoted_data.fillna(0, inplace=True)

    return pivoted_data

# Modify the main function
def main():
    sensor_data = get_sensor_data(home_assistant_url, access_token)
    if sensor_data:
        expected_time_steps = 10  # The number of time steps the model was trained on
        expected_features = list(range(4))  # List of expected feature indices

        print(sensor_data)

        input_data = preprocess_data(sensor_data)
        print(input_data)
        model = load_model(model_path)
        
        predictions = model.predict(input_data)
        print(predictions)

if __name__ == "__main__":
    main()