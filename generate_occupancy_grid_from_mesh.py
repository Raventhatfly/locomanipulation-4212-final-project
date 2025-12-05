"""
Generate 2D occupancy grid from kitchen mesh for path planning.
Reads the OBJ file and creates a top-down occupancy map.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def parse_obj_file(obj_path):
    """Parse OBJ file to extract vertices and faces"""
    vertices = []
    faces = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Vertex: v x y z
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
            elif line.startswith('f '):
                # Face: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                parts = line.strip().split()[1:]
                face_indices = []
                for part in parts:
                    # Extract vertex index (first number before /)
                    idx = int(part.split('/')[0]) - 1  # OBJ is 1-indexed
                    face_indices.append(idx)
                faces.append(face_indices)
    
    return np.array(vertices), faces

def create_occupancy_grid(vertices, faces, resolution=0.05, robot_height_min=0.1, robot_height_max=1.5):
    """
    Create 2D occupancy grid from 3D mesh.
    
    Args:
        vertices: Nx3 array of vertices
        faces: List of face indices
        resolution: Grid cell size in meters
        robot_height_min: Minimum height for robot collision (floor level)
        robot_height_max: Maximum height for robot collision
    
    Returns:
        grid: 2D occupancy grid (0=free, 1=occupied)
        metadata: Dict with grid parameters
    """
    # The mesh is rotated: floor is X-Z plane, Y is height
    # Apply coordinate transformation: Drake X-Y plane = mesh X-(-Z) plane
    # Negate Z to flip the Y axis
    vertices_transformed = vertices.copy()
    vertices_transformed[:, 2] = -vertices_transformed[:, 2]
    
    # Get bounds in X-Z plane (the actual floor in the mesh)
    x_min, x_max = vertices_transformed[:, 0].min(), vertices_transformed[:, 0].max()
    z_min, z_max = vertices_transformed[:, 2].min(), vertices_transformed[:, 2].max()
    
    print(f"Mesh bounds: X=[{x_min:.2f}, {x_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")
    
    # Create grid
    grid_width = int(np.ceil((x_max - x_min) / resolution))  # X dimension
    grid_height = int(np.ceil((z_max - z_min) / resolution))  # Z dimension (maps to Drake Y)
    
    print(f"Grid size: {grid_width} x {grid_height} cells ({resolution}m resolution)")
    
    # Initialize grid (0 = free, 1 = occupied)
    # grid[z_idx, x_idx] for proper numpy indexing
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # Project all triangular faces onto 2D grid
    print(f"Projecting mesh faces onto 2D grid...")
    print(f"  Detecting obstacles: Y (height) between {robot_height_min}m and {robot_height_max}m")
    
    wall_count = 0
    obstacle_count = 0
    
    for i, face in enumerate(faces):
        if i % 10000 == 0:
            print(f"  Processing face {i}/{len(faces)}")
        
        # Get vertices of this face
        face_verts = vertices_transformed[face]
        y_coords = vertices[face][:, 1]  # Y is the height in the original mesh
        face_y_min = y_coords.min()
        face_y_max = y_coords.max()
        face_y_span = face_y_max - face_y_min
        
        # Mark as obstacle if:
        # 1. Obstacle at robot height: has vertices between 0.1m and 1.5m (counters, tables, NOT floor)
        # 2. Wall: vertical structure spanning >1.5m in height
        is_obstacle_height = (face_y_max > robot_height_min) and (face_y_min < robot_height_max)
        is_wall = face_y_span > 1.5
        
        if is_wall:
            wall_count += 1
        if is_obstacle_height:
            obstacle_count += 1
        
        if is_obstacle_height or is_wall:
            # Project to 2D using X-Z plane
            for vert in face_verts:
                x, z = vert[0], vert[2]
                
                # Convert to grid coordinates
                grid_x = int((x - x_min) / resolution)
                grid_z = int((z - z_min) / resolution)
                
                # Mark cell as occupied (with bounds check)
                if 0 <= grid_x < grid_width and 0 <= grid_z < grid_height:
                    grid[grid_z, grid_x] = 1
            
            # Also fill the triangle
            pts_2d = face_verts[:, [0, 2]]  # Take X and Z
            # Convert world coords to grid coords
            pts_grid_x = (pts_2d[:, 0] - x_min) / resolution
            pts_grid_z = (pts_2d[:, 1] - z_min) / resolution
            # Stack as [[x1, z1], [x2, z2], ...] for cv2.fillPoly
            pts_grid = np.column_stack((pts_grid_x, pts_grid_z)).astype(np.int32)
            
            # Use OpenCV to fill triangle
            cv2.fillPoly(grid, [pts_grid], 1)
    
    print(f"\nDetection summary:")
    print(f"  Wall faces (y_span > 1.5m): {wall_count}")
    print(f"  Obstacle faces ({robot_height_min}m < y < {robot_height_max}m): {obstacle_count}")
    
    # Dilate obstacles for robot safety margin
    kernel_size = int(0.3 / resolution)  # 30cm safety margin
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    grid = cv2.dilate(grid, kernel, iterations=1)

    metadata = {
        'resolution': resolution,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': z_min,  # Map mesh Z to Drake Y
        'y_max': z_max,
        'width': grid_width,
        'height': grid_height
    }
    
    return grid, metadata

def visualize_grid(grid, metadata, save_path=None):
    """Visualize the occupancy grid"""
    plt.figure(figsize=(12, 10))
    # Grid is [z_idx, x_idx] where mesh X-(-Z) becomes Drake X-Y
    # Coordinates already transformed, no flipping needed
    
    plt.imshow(grid, cmap='gray_r', origin='lower', 
               extent=[metadata['x_min'], metadata['x_max'],
                      metadata['y_min'], metadata['y_max']],
               aspect='equal')
    plt.colorbar(label='Occupancy (0=free, 1=occupied)')
    plt.xlabel('Drake X (mesh X) [meters]')
    plt.ylabel('Drake Y (mesh Z) [meters]')
    plt.title('Kitchen Occupancy Grid (Top-Down View)')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def main():
    # Paths
    current_dir = Path(__file__).parent
    obj_path = current_dir / "kitchen_model" / "meshes_new" / "kitchen_env.obj"
    
    print("="*60)
    print("Generating Occupancy Grid from Kitchen Mesh")
    print("="*60)
    
    # Parse mesh
    print("\nParsing OBJ file...")
    vertices, faces = parse_obj_file(obj_path)
    print(f"Loaded {len(vertices)} vertices and {len(faces)} faces")
    
    # Create occupancy grid
    print("\nCreating occupancy grid...")
    # Consider obstacles at robot height (0.1m to 1.5m) plus walls
    # This captures all obstacles that block the mobile robot's path
    grid, metadata = create_occupancy_grid(vertices, faces, resolution=0.05, robot_height_min=0.1, robot_height_max=1.5)
    
    # Calculate statistics
    total_cells = grid.size
    occupied_cells = np.sum(grid)
    free_cells = total_cells - occupied_cells
    
    print(f"\nGrid statistics:")
    print(f"  Total cells: {total_cells}")
    print(f"  Occupied cells: {occupied_cells} ({100*occupied_cells/total_cells:.1f}%)")
    print(f"  Free cells: {free_cells} ({100*free_cells/total_cells:.1f}%)")
    
    # Save grid and metadata
    output_dir = current_dir / "data"
    output_dir.mkdir(exist_ok=True)
    
    grid_path = output_dir / "kitchen_occupancy_grid.npy"
    metadata_path = output_dir / "kitchen_occupancy_grid_metadata.npy"
    
    np.save(grid_path, grid)
    np.save(metadata_path, metadata)
    
    print(f"\nSaved occupancy grid to {grid_path}")
    print(f"Saved metadata to {metadata_path}")
    
    # Visualize
    print("\nGenerating visualization...")
    viz_path = output_dir / "kitchen_occupancy_grid.png"
    visualize_grid(grid, metadata, save_path=viz_path)
    
    print("\n" + "="*60)
    print("✓ Occupancy grid generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
