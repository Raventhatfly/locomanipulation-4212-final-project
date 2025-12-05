"""
Grid-based coverage path planning for kitchen exploration.
Uses occupancy grid to generate paths that cover all free space.
"""
from pydrake.all import Simulator, StartMeshcat
from manipulation.station import LoadScenario, MakeHardwareStation
import numpy as np
import os
import time
from scipy.interpolate import interp1d
from collections import deque
import matplotlib.pyplot as plt

# Start meshcat
meshcat = StartMeshcat()

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
kitchen_model_path = os.path.join(current_dir, "kitchen_model")
assets_path = os.path.join(current_dir, "assets")

# Load scenario
scenario_file = os.path.join(kitchen_model_path, "real_kitchen_scenario.yaml")
with open(scenario_file, 'r') as f:
    scenario_data = f.read()
    scenario_data = scenario_data.replace("{KITCHEN_MODEL_PATH}", kitchen_model_path)
    scenario_data = scenario_data.replace("{ASSETS_PATH}", assets_path)

scenario = LoadScenario(data=scenario_data)
station = MakeHardwareStation(scenario, meshcat)
simulator = Simulator(station)
simulator.Initialize()
context = simulator.get_context()
station.ForcedPublish(context)

print(f"Meshcat is running at: {meshcat.web_url()}")

# ============================================================================
# Load Occupancy Grid
# ============================================================================

print("\n" + "="*60)
print("Loading Occupancy Grid")
print("="*60)

data_dir = os.path.join(current_dir, "data")
grid_path = os.path.join(data_dir, "kitchen_occupancy_grid.npy")
metadata_path = os.path.join(data_dir, "kitchen_occupancy_grid_metadata.npy")

if not os.path.exists(grid_path):
    print("ERROR: Occupancy grid not found!")
    print("Please run generate_occupancy_grid_from_mesh.py first")
    exit(1)

grid = np.load(grid_path)
metadata = np.load(metadata_path, allow_pickle=True).item()

print(f"✓ Loaded occupancy grid:")
print(f"  Resolution: {metadata['resolution']}m")
print(f"  Size: {metadata['width']} x {metadata['height']} cells")
print(f"  Bounds: X=[{metadata['x_min']:.2f}, {metadata['x_max']:.2f}], "
      f"Y=[{metadata['y_min']:.2f}, {metadata['y_max']:.2f}]")

# ============================================================================
# Grid-Based Coverage Path Planning
# ============================================================================

class GridCoveragePlanner:
    """Generate coverage path that visits all free cells in occupancy grid"""
    
    def __init__(self, grid, metadata, coverage_radius=0.8):
        self.grid = grid  # 0=free, 1=occupied
        self.metadata = metadata
        self.resolution = metadata['resolution']
        self.coverage_radius = coverage_radius
        
        # Coverage radius in grid cells
        self.coverage_cells = int(np.ceil(coverage_radius / self.resolution))
        
        # Create coverage map (tracks which cells are covered)
        self.coverage_map = np.zeros_like(grid, dtype=bool)
        
        # Mark occupied cells as already "covered" (don't need to visit)
        self.coverage_map[grid == 1] = True
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = self.metadata['x_min'] + grid_x * self.resolution
        y = self.metadata['y_min'] + grid_y * self.resolution
        return x, y
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.metadata['x_min']) / self.resolution)
        grid_y = int((y - self.metadata['y_min']) / self.resolution)
        return grid_x, grid_y
    
    def mark_covered(self, grid_x, grid_y):
        """Mark all cells within coverage radius as covered"""
        h, w = self.grid.shape
        
        for dy in range(-self.coverage_cells, self.coverage_cells + 1):
            for dx in range(-self.coverage_cells, self.coverage_cells + 1):
                # Check if within circular coverage radius
                if dx*dx + dy*dy <= self.coverage_cells * self.coverage_cells:
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        self.coverage_map[ny, nx] = True
    
    def get_uncovered_cells(self):
        """Get list of uncovered free cells"""
        uncovered = []
        h, w = self.grid.shape
        
        for y in range(h):
            for x in range(w):
                if not self.coverage_map[y, x]:
                    uncovered.append((x, y))
        
        return uncovered
    
    def find_nearest_uncovered(self, current_pos):
        """Find nearest uncovered cell to current position"""
        uncovered = self.get_uncovered_cells()
        
        if not uncovered:
            return None
        
        # Find closest uncovered cell
        distances = [np.sqrt((x - current_pos[0])**2 + (y - current_pos[1])**2) 
                    for x, y in uncovered]
        nearest_idx = np.argmin(distances)
        
        return uncovered[nearest_idx]
    
    def boustrophedon_coverage(self, start_grid_pos):
        """
        Generate boustrophedon (back-and-forth) coverage path.
        Sweeps through the space in a lawnmower pattern.
        """
        path = []
        h, w = self.grid.shape
        
        # Start position
        current_x, current_y = start_grid_pos
        path.append((current_x, current_y))
        self.mark_covered(current_x, current_y)
        
        # Sweep parameters
        sweep_spacing = self.coverage_cells * 2  # Spacing between sweep lines
        
        # Generate sweep lines (vertical sweeps)
        sweep_x_positions = range(0, w, sweep_spacing)
        
        for i, sweep_x in enumerate(sweep_x_positions):
            # Alternate direction: up or down
            if i % 2 == 0:
                y_range = range(0, h)
            else:
                y_range = range(h-1, -1, -1)
            
            for y in y_range:
                # Check if this cell is free
                if self.grid[y, sweep_x] == 0:
                    path.append((sweep_x, y))
                    self.mark_covered(sweep_x, y)
        
        return path
    
    def greedy_coverage(self, start_grid_pos):
        """
        Greedy coverage: always go to nearest uncovered cell.
        Good for irregular spaces.
        """
        path = []
        current_pos = start_grid_pos
        
        path.append(current_pos)
        self.mark_covered(current_pos[0], current_pos[1])
        
        max_iterations = 10000
        iteration = 0
        
        while iteration < max_iterations:
            # Find nearest uncovered cell
            next_cell = self.find_nearest_uncovered(current_pos)
            
            if next_cell is None:
                print(f"  ✓ Full coverage achieved!")
                break
            
            # Plan path to next cell using A*
            segment = self.astar_path(current_pos, next_cell)
            
            if segment:
                path.extend(segment[1:])  # Skip first point (already in path)
                current_pos = next_cell
                self.mark_covered(next_cell[0], next_cell[1])
            else:
                # Can't reach this cell, mark as covered anyway
                self.mark_covered(next_cell[0], next_cell[1])
            
            iteration += 1
            
            if iteration % 100 == 0:
                uncovered_count = len(self.get_uncovered_cells())
                coverage_pct = 100 * (1 - uncovered_count / np.sum(self.grid == 0))
                print(f"  Progress: {coverage_pct:.1f}% covered ({uncovered_count} cells remaining)")
        
        return path
    
    def astar_path(self, start, goal):
        """A* pathfinding on grid"""
        h, w = self.grid.shape
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        open_set = [(start, 0)]
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            # Get node with lowest f_score
            open_set.sort(key=lambda x: x[1] + heuristic(x[0], goal))
            current, _ = open_set.pop(0)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            # Check neighbors (4-connected)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor
                
                # Check bounds and obstacles
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if self.grid[ny, nx] == 1:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    if neighbor not in [x[0] for x in open_set]:
                        open_set.append((neighbor, tentative_g))
        
        return None  # No path found

def smooth_path(path, num_points=100):
    """Interpolate path for smooth motion"""
    if len(path) < 2:
        return path
    
    path_array = np.array(path)
    distances = np.cumsum([0] + [np.linalg.norm(path_array[i+1] - path_array[i]) 
                                   for i in range(len(path)-1)])
    t = distances / distances[-1]
    
    fx = interp1d(t, path_array[:, 0], kind='linear')
    fy = interp1d(t, path_array[:, 1], kind='linear')
    
    t_smooth = np.linspace(0, 1, num_points)
    smooth_path = np.column_stack([fx(t_smooth), fy(t_smooth)])
    
    return smooth_path

# ============================================================================
# Generate Coverage Path
# ============================================================================

print("\n" + "="*60)
print("Generating Coverage Path")
print("="*60)

# Create planner
coverage_radius = 1.2  # meters - increased for more margin
planner = GridCoveragePlanner(grid, metadata, coverage_radius=coverage_radius)

# Start position (robot starts at x=-4.0, y=0.0)
start_world = (-4.0, 0.0)
start_grid = planner.world_to_grid(start_world[0], start_world[1])

print(f"Start position: world=({start_world[0]:.2f}, {start_world[1]:.2f}), "
      f"grid=({start_grid[0]}, {start_grid[1]})")
print(f"Coverage radius: {coverage_radius}m ({planner.coverage_cells} cells)")

# Generate path using greedy coverage
print("\nGenerating greedy coverage path...")
grid_path = planner.greedy_coverage(start_grid)

print(f"\n✓ Generated path with {len(grid_path)} waypoints")

# Convert to world coordinates
world_path = [planner.grid_to_world(x, y) for x, y in grid_path]

# Calculate coverage statistics
total_free_cells = np.sum(grid == 0)
covered_cells = np.sum(planner.coverage_map & (grid == 0))
coverage_pct = 100 * covered_cells / total_free_cells

print(f"Coverage: {covered_cells}/{total_free_cells} cells ({coverage_pct:.1f}%)")

# Calculate path length
path_length = sum(np.linalg.norm(np.array(world_path[i+1]) - np.array(world_path[i])) 
                 for i in range(len(world_path)-1))
print(f"Path length: {path_length:.2f} m")

# ============================================================================
# Visualize Path on Grid
# ============================================================================

print("\nGenerating path visualization...")
plt.figure(figsize=(14, 10))

# Show occupancy grid
plt.imshow(grid, cmap='gray_r', origin='lower',
           extent=[metadata['x_min'], metadata['x_max'], 
                  metadata['y_min'], metadata['y_max']], alpha=0.5)

# Show coverage map
coverage_display = planner.coverage_map.astype(float)
coverage_display[grid == 1] = np.nan  # Don't show coverage on obstacles
plt.imshow(coverage_display, cmap='Greens', origin='lower',
           extent=[metadata['x_min'], metadata['x_max'],
                  metadata['y_min'], metadata['y_max']], alpha=0.3)

# Plot path
path_array = np.array(world_path)
plt.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2, label='Coverage Path')
plt.plot(path_array[0, 0], path_array[0, 1], 'go', markersize=10, label='Start')
plt.plot(path_array[-1, 0], path_array[-1, 1], 'ro', markersize=10, label='End')

plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title(f'Coverage Path Planning ({coverage_pct:.1f}% coverage)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(data_dir, 'coverage_path.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to data/coverage_path.png")

# ============================================================================
# Animate Robot
# ============================================================================

print("\n" + "="*60)
print("Animating Robot")
print("="*60)

plant = station.GetSubsystemByName("plant")
plant_context = plant.GetMyContextFromRoot(context)
mobile_iiwa = plant.GetModelInstanceByName("mobile_iiwa")
base_x_joint = plant.GetJointByName("iiwa_base_x", mobile_iiwa)
base_y_joint = plant.GetJointByName("iiwa_base_y", mobile_iiwa)

# Smooth the path
print("Smoothing path...")
smooth_waypoints = smooth_path(world_path, num_points=min(500, len(world_path) * 3))
print(f"Smoothed to {len(smooth_waypoints)} points")

print("\nAnimating...")
for i, waypoint in enumerate(smooth_waypoints):
    if i % 50 == 0:
        progress = (i / len(smooth_waypoints)) * 100
        print(f"Progress: {progress:.1f}% - Position: ({waypoint[0]:.2f}, {waypoint[1]:.2f})")
    
    base_x_joint.set_translation(plant_context, waypoint[0])
    base_y_joint.set_translation(plant_context, waypoint[1])
    station.ForcedPublish(context)
    time.sleep(0.02)

print("\n✓ Animation complete!")
print(f"Final position: ({smooth_waypoints[-1][0]:.2f}, {smooth_waypoints[-1][1]:.2f})")

print("\n" + "="*60)
print("✓ Coverage Path Planning Complete!")
print("="*60)
print(f"  Coverage: {coverage_pct:.1f}%")
print(f"  Path length: {path_length:.2f} m")
print(f"  Waypoints: {len(world_path)}")
print("="*60)

print("\nPress Enter to exit...")
try:
    input()
except KeyboardInterrupt:
    print("\nProgram terminated")
