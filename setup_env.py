from pydrake.all import Simulator, StartMeshcat
from manipulation.station import LoadScenario, MakeHardwareStation
import numpy as np
import os

# start the mesh 
meshcat = StartMeshcat()

# Set up paths for the scenario file
current_dir = os.path.dirname(os.path.abspath(__file__))
kitchen_model_path = os.path.join(current_dir, "kitchen_model")
assets_path = os.path.join(current_dir, "assets")

# Load scenario file
scenario_file = os.path.join(kitchen_model_path, "real_kitchen_scenario.yaml")

# Read and substitute paths in the YAML file
with open(scenario_file, 'r') as f:
    scenario_data = f.read()
    scenario_data = scenario_data.replace("{KITCHEN_MODEL_PATH}", kitchen_model_path)
    scenario_data = scenario_data.replace("{ASSETS_PATH}", assets_path)

# Load scenario and create hardware station
scenario = LoadScenario(data=scenario_data)
station = MakeHardwareStation(scenario, meshcat)

# Create simulator using the station
simulator = Simulator(station)
simulator.Initialize()

# get context and publish initial state
context = simulator.get_context()
station.ForcedPublish(context)

print(f"Meshcat is running at: {meshcat.web_url()}")

try:
    input()
except KeyboardInterrupt:
    print("\nProgram terminated")
