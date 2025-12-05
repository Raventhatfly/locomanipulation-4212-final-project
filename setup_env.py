from pydrake.all import Simulator, StartMeshcat
from manipulation.station import LoadScenario, MakeHardwareStation
import numpy as np
import os
import time

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
simulator.set_target_realtime_rate(1.0)  # Run at real-time speed

# get context and publish initial state
context = simulator.get_context()
station.ForcedPublish(context)

print(f"Meshcat is running at: {meshcat.web_url()}")

# Start recording animation in Meshcat
meshcat.StartRecording()

# Run simulation with animation - publish frames continuously
print("\nRunning simulation with animation...")
simulation_time = 5.0
dt = 0.01  # 10ms per frame (100 fps for smooth recording)
current_time = 0.0

while current_time < simulation_time:
    simulator.AdvanceTo(current_time + dt)
    current_time += dt
    
    if int(current_time * 10) % 10 == 0:  # Print every second
        print(f"  Time: {current_time:.1f}s")

# Stop recording and publish the animation
meshcat.StopRecording()
meshcat.PublishRecording()

print("✓ Simulation complete - Animation is now available in Meshcat!")
print("   Click the 'Animations' menu in Meshcat to play/pause the animation.")

try:
    input("\nPress Enter to exit...")
except KeyboardInterrupt:
    print("\nProgram terminated")
