# Cell 1
from pydrake.all import (Meshcat, Simulator, InverseKinematics, Solve)
from manipulation.station import LoadScenario, MakeHardwareStation

from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import SolverOptions, SnoptSolver
import numpy as np
import time

meshcat = Meshcat()

try:
    input("Press any key to continue.")
except KeyboardInterrupt:
    print("\nProgram Exit.")

scenario_data = """
directives:
- add_model:
    name: mobile_iiwa
    file: package://manipulation/mobile_iiwa14_primitive_collision.urdf
model_drivers:
    mobile_iiwa: !InverseDynamicsDriver {}
"""

scenario = LoadScenario(data=scenario_data)
station = MakeHardwareStation(scenario, meshcat)
sim = Simulator(station)
context = sim.get_mutable_context()

x0 = station.GetOutputPort("mobile_iiwa.state_estimated").Eval(context)
station.GetInputPort("mobile_iiwa.desired_state").FixValue(context, x0)
sim.AdvanceTo(0.1)

# Cell 3


# Cell 4
# Create a room environment in meshcat
from pydrake.geometry import Box, Cylinder, Rgba
from pydrake.all import RigidTransform
import numpy as np

# Floor (large flat box)
floor_geometry = Box(16, 16, 0.4)  # width, depth, height (4x larger)
floor_pose = RigidTransform([0, 0, -0.2])  # slightly below origin
meshcat.SetObject("floor", floor_geometry, rgba=Rgba(0.5, 0.5, 0.5, 0.8))
meshcat.SetTransform("floor", floor_pose)

# Walls (4 walls)
wall_height = 10.0
wall_thickness = 0.4

# Front wall (x-direction)
front_wall = Box(16, wall_thickness, wall_height)
front_pose = RigidTransform([0, 8, wall_height/2])
meshcat.SetObject("front_wall", front_wall, rgba=Rgba(0.8, 0.8, 0.8, 0.6))
meshcat.SetTransform("front_wall", front_pose)

# Back wall (x-direction)
back_wall = Box(16, wall_thickness, wall_height)
back_pose = RigidTransform([0, -8, wall_height/2])
meshcat.SetObject("back_wall", back_wall, rgba=Rgba(0.8, 0.8, 0.8, 0.6))
meshcat.SetTransform("back_wall", back_pose)

# Left wall (y-direction)
left_wall = Box(wall_thickness, 16, wall_height)
left_pose = RigidTransform([-8, 0, wall_height/2])
meshcat.SetObject("left_wall", left_wall, rgba=Rgba(0.8, 0.8, 0.8, 0.6))
meshcat.SetTransform("left_wall", left_pose)

# Right wall
right_wall = Box(wall_thickness, 16, wall_height)
right_pose = RigidTransform([8, 0, wall_height/2])
meshcat.SetObject("right_wall", right_wall, rgba=Rgba(0.8, 0.8, 0.8, 0.6))
meshcat.SetTransform("right_wall", right_pose)

# Cell 5
# Add tables
table_positions = [(-4, -4, 0.75), (3, -4, 0.75), (1, 3, 0.75)]
table_size = (2, 1.5, 0.05)  # width, depth, height

for idx, (tx, ty, tz) in enumerate(table_positions):
    # Table top
    top_geometry = Box(table_size[0], table_size[1], table_size[2])
    top_pose = RigidTransform([tx, ty, tz])
    meshcat.SetObject(f"table_{idx}_top", top_geometry, rgba=Rgba(0.6, 0.4, 0.2, 0.8))
    meshcat.SetTransform(f"table_{idx}_top", top_pose)
    
    # Table legs (4 cylinders)
    leg_radius = 0.03
    leg_height = tz - 0.025
    leg_offset_x = table_size[0] / 2 - 0.1
    leg_offset_y = table_size[1] / 2 - 0.1
    
    for leg_idx, (dx, dy) in enumerate([(-leg_offset_x, -leg_offset_y), 
                                         (leg_offset_x, -leg_offset_y),
                                         (-leg_offset_x, leg_offset_y), 
                                         (leg_offset_x, leg_offset_y)]):
        leg_geometry = Cylinder(leg_radius, leg_height)
        leg_pose = RigidTransform([tx + dx, ty + dy, leg_height / 2])
        meshcat.SetObject(f"table_{idx}_leg_{leg_idx}", leg_geometry, rgba=Rgba(0.4, 0.3, 0.2, 0.8))
        meshcat.SetTransform(f"table_{idx}_leg_{leg_idx}", leg_pose)

print("Tables created at positions:", table_positions)

# Waypoints around the room (circular path)
x_center, y_center, z_center = 0.0, 0.0, 1.5  # Around the center
radius = 5.0
theta_values = np.linspace(0, 2 * np.pi, num=8, endpoint=False)
waypoints = [(x_center + radius * np.cos(theta), y_center + radius * np.sin(theta), z_center)
             for theta in theta_values]

waypoints.insert(0, (0.0, 0.0, 1.5)) # Add initial position as the first waypoint

print("\nGenerated waypoints around the room (circular path):")
for i, (x, y, z) in enumerate(waypoints):
    print(f"Waypoint {i+1}: x={x}, y={y}, z={z}")

    
# Cell 6
from pydrake.all import InverseKinematics, PiecewisePolynomial

def plan_motion_to_waypoint(initial_state, target_position, station):
    """
    Plans a motion trajectory from initial_state to target_position for the mobile_iiwa.
    """
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.CreateDefaultContext()

    # Set initial position
    plant.SetPositions(plant_context, initial_state[:plant.num_positions()])

    # Create IK problem to find a target configuration
    ik = InverseKinematics(plant, plant_context)
    q_variables = ik.q()

    # Constraint: end-effector close to target_position
    ee_frame = plant.GetFrameByName("iiwa_link_7")
    ik.AddPositionConstraint(
        frameA=plant.world_frame(),
        frameB=ee_frame,
        p_BQ=np.array([0.0, 0.0, 0.0]),  # end-effector position in its frame
        p_AQ_lower=np.array(target_position) - 0.01,
        p_AQ_upper=np.array(target_position) + 0.01
    )

    # Solve IK
    ik_result = Solve(ik.prog())
    if not ik_result.is_success():
        print("IK failed for waypoint:", target_position)
        return None

    q_final = ik_result.GetSolution(q_variables)

    # Create a simple linear trajectory in joint space
    T = 5.0  # Trajectory duration (s)
    times = np.array([0.0, T])
    positions = np.vstack([initial_state[:plant.num_positions()], q_final])
    traj = PiecewisePolynomial.FirstOrderHold(times, positions.T)

    return traj, T

# Cell 7
from pydrake.all import KinematicTrajectoryOptimization, SnoptSolver, SolverOptions
import time  # Make sure time is imported at the top

def plan_motion_kinematically(station, start_state, target_position):
    """
    Plans a feasible joint-space trajectory to move the robot's end effector to the target_position,
    respecting joint limits and smoothness. Uses IK + linear interpolation.
    """

    plant = station.GetSubsystemByName("plant")
    plant_context = plant.CreateDefaultContext()

    # Use IK to find a target configuration that achieves the target position
    plant.SetPositions(plant_context, start_state[:plant.num_positions()])
    
    ee_frame = plant.GetFrameByName("iiwa_link_7")
    
    # Solve IK for target configuration
    ik = InverseKinematics(plant, plant_context)
    q_var = ik.q()
    ik.AddPositionConstraint(
        frameA=plant.world_frame(),
        frameB=ee_frame,
        p_BQ=np.array([0.0, 0.0, 0.0]),
        p_AQ_lower=np.array(target_position) - 0.05,
        p_AQ_upper=np.array(target_position) + 0.05
    )

    ik_result = Solve(ik.prog())
    if not ik_result.is_success():
        print(f"  IK failed for target position {target_position}")
        return None, None

    q_goal = ik_result.GetSolution(q_var)
    q_start = start_state[:plant.num_positions()]

    # Create a simple linear interpolation trajectory in joint space
    # This is guaranteed to be feasible since both endpoints are valid
    from pydrake.trajectories import PiecewisePolynomial
    
    times = np.array([0.0, 3.0])  # 3-second trajectory
    knots = np.column_stack([q_start, q_goal])
    
    traj = PiecewisePolynomial.FirstOrderHold(times, knots)
    T = traj.end_time()

    print(f"  IK successful. Linear trajectory duration: {T:.2f} seconds")
    return traj, T

# Cell 8
all_trajectories = []

for wp_idx, waypoint in enumerate(waypoints):
    x0 = station.GetOutputPort("mobile_iiwa.state_estimated").Eval(context)
    initial_state = x0.copy()

    print(f"\nPlanning kinematic motion to waypoint {wp_idx}: {waypoint}")
    traj, T = plan_motion_kinematically(
        station=station,
        start_state=initial_state,
        target_position=np.array(waypoint)
    )

    if traj is None:
        print(f"Skipping waypoint {wp_idx}: Kinematic plan failed.")
        continue

    all_trajectories.append((traj, T))

    # Animate the trajectory in MeshCat
    plant = station.GetSubsystemByName("plant")
    
    time_samples = np.linspace(0, T, num=50)
    
    for t in time_samples:
        q = traj.value(t).ravel()
        # Get the plant context from the simulator context
        sim_context = sim.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(sim_context)
        plant.SetPositions(plant_context, q)
        station.ForcedPublish(sim_context)
        time.sleep(0.03)
    
    # Finalize position at waypoint
    sim_context = sim.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(sim_context)
    plant.SetPositions(plant_context, traj.value(T).ravel())
    station.ForcedPublish(sim_context)
    print(f"Reached waypoint {wp_idx}: ({waypoints[wp_idx][0]}, {waypoints[wp_idx][1]})")

print(f"Exploration complete! Visited {len(all_trajectories)} waypoints.")


try:
    input("Press any key to end.")
except KeyboardInterrupt:
    print("\nProgram Exit.")