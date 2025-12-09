from __future__ import annotations
import numpy as np
import random
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
import json
from pydrake.all import PiecewisePolynomial

dt = 0.05
pause_before_action = 1.0
pause_after_action = 1.0
opened, closed = 0.107, 0.0


class RRTPlanner:
    """RRT path planner using occupancy grid for collision checking."""
    def __init__(self, grid_info, step_size=0.3, max_iterations=5000, goal_sample_rate=0.1):
        self.occupancy_grid = grid_info['occupancy_grid']
        self.x_positions = grid_info['x_positions']
        self.y_positions = grid_info['y_positions']
        self.resolution = grid_info['resolution']
        self.bounds = grid_info['bounds']
        self.x_min, self.x_max, self.y_min, self.y_max = self.bounds
        
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        
    def is_collision_free(self, x, y):
        i = int((x - self.x_min) / self.resolution)
        j = int((y - self.y_min) / self.resolution)
        
        if i < 0 or i >= self.occupancy_grid.shape[1] or j < 0 or j >= self.occupancy_grid.shape[0]:
            return False
        
        return self.occupancy_grid[j, i] == 0
    
    def is_path_collision_free(self, p1, p2, num_checks=20):
        for i in range(num_checks + 1):
            t = i / num_checks
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if not self.is_collision_free(x, y):
                return False
        return True
    
    def sample_random_point(self):
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        return np.array([x, y])
    
    def get_nearest_node(self, tree, point):
        distances = [np.linalg.norm(node - point) for node in tree]
        return np.argmin(distances)
    
    def steer(self, from_point, to_point):
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_point
        else:
            return from_point + (direction / distance) * self.step_size
    
    def plan(self, start, goal):
        start = np.array(start)
        goal = np.array(goal)

        if not self.is_collision_free(*start):
            print("Start is in collision!")
            return None
        if not self.is_collision_free(*goal):
            print("Goal is in collision!")
            return None
        
        tree = [start]
        parent = {0: None}
        
        for _ in range(self.max_iterations):
            sample = goal if random.random() < self.goal_sample_rate else self.sample_random_point()
            
            nearest_idx = self.get_nearest_node(tree, sample)
            nearest = tree[nearest_idx]
            
            new_point = self.steer(nearest, sample)
            
            if self.is_path_collision_free(nearest, new_point):
                new_idx = len(tree)
                tree.append(new_point)
                parent[new_idx] = nearest_idx
                
                if np.linalg.norm(new_point - goal) < self.step_size:
                    if self.is_path_collision_free(new_point, goal):
                        goal_idx = len(tree)
                        tree.append(goal)
                        parent[goal_idx] = new_idx
                        
                        path = []
                        cur = goal_idx
                        while cur is not None:
                            path.append(tuple(tree[cur]))
                            cur = parent[cur]
                        return path[::-1]
        
        print("Failed to find a path.")
        return None

def load_kitchen_grid(data_path, waypoint_resolution=0.2):
    data_dir = Path(data_path)

    occupancy_grid_full = np.load(data_dir / 'kitchen_occupancy_grid.npy')
    metadata = np.load(data_dir / 'kitchen_occupancy_grid_metadata.npy', allow_pickle=True).item()

    grid_resolution = metadata.get('resolution', 0.1)

    X_MIN, X_MAX = -7.7, -2.7
    Y_MIN, Y_MAX = -3.0, 1.0

    orig_x_min = metadata.get('x_min', -11.0)
    orig_x_max = metadata.get('x_max', 1.0)
    orig_y_min = metadata.get('y_min', -6.0)
    orig_y_max = metadata.get('y_max', 6.0)

    orig_x_positions = np.arange(orig_x_min, orig_x_max, grid_resolution)
    orig_y_positions = np.arange(orig_y_min, orig_y_max, grid_resolution)

    x_start_idx = np.searchsorted(orig_x_positions, X_MIN)
    x_end_idx = np.searchsorted(orig_x_positions, X_MAX)
    y_start_idx = np.searchsorted(orig_y_positions, Y_MIN)
    y_end_idx = np.searchsorted(orig_y_positions, Y_MAX)

    occupancy_grid = occupancy_grid_full[y_start_idx:y_end_idx, x_start_idx:x_end_idx]

    if abs(grid_resolution - waypoint_resolution) > 1e-2:
        occupancy_grid = zoom(
            occupancy_grid,
            grid_resolution / waypoint_resolution,
            order=0
        ).astype(np.uint8)

    x_positions = np.arange(X_MIN, X_MAX, waypoint_resolution)
    y_positions = np.arange(Y_MIN, Y_MAX, waypoint_resolution)

    return {
        "occupancy_grid": occupancy_grid,
        "x_positions": x_positions,
        "y_positions": y_positions,
        "resolution": waypoint_resolution,
        "bounds": (X_MIN, X_MAX, Y_MIN, Y_MAX)
    }


class KitchenMap:
    def __init__(self, plant, plant_context, object_names=None):
        self.plant = plant
        self.context = plant_context
        self.object_names = object_names or ['apple', 'banana', 'lemon', 'orange', 'peach']
        self.map = {}
        self.update()

    def update(self):
        """Refresh all object positions by querying the plant."""
        updated = 0
        new_map = {}

        for name in self.object_names:
            try:
                model_instance = self.plant.GetModelInstanceByName(name)
                body = self.plant.GetBodyByName(f"{name}_link", model_instance)
                pose = self.plant.EvalBodyPoseInWorld(self.context, body)
                pos = pose.translation()

                new_map[name] = {
                    "position": pos.tolist(),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2])
                }
                updated += 1

            except Exception as e:
                print(f"⚠️ Could not find '{name}': {e}")

        self.map = new_map
        print(f"✓ Updated kitchen map ({updated} objects)")

    def get_position(self, name):
        """Return (x, y, z) of a named object, or None if missing."""
        entry = self.map.get(name)
        return (entry["x"], entry["y"], entry["z"]) if entry else None

    def as_json(self):
        """Return entire map as JSON string."""
        return json.dumps(self.map, indent=2)

    def print(self):
        """Pretty-print the map."""
        print(self.as_json())

def _append_path(
    times: list[float],
    Q: list[np.ndarray],
    t: float,
    path: list[np.ndarray],
) -> float:
    for q in path:
        if times:
            t += dt
        times.append(t)
        Q.append(np.asarray(q, dtype=float))
    return t


def _hold(
    times: list[float],
    Q: list[np.ndarray],
    t: float,
    q_hold: np.ndarray,
    duration: float,
) -> float:
    if duration <= 0:
        return t
    t += duration
    times.append(t)
    Q.append(np.asarray(q_hold, dtype=float))
    return t


def build_trajs_place(
    path_place: list[np.ndarray],
    # path_upright: list[np.ndarray],
    q_grasp: np.ndarray,
    start_time: float = 0.0,
    dt_unused: float = 0.05,
) -> tuple[PiecewisePolynomial, PiecewisePolynomial]:
    """
    Place-phase trajectory builder.

    Sequence:
      - follow path_place (ends at q_grasp, gripper CLOSED)
      - pause_before_action (hold q_grasp)
      - "open" moment (t_open)
      - pause_after_action (still at q_grasp)
      - follow path_upright
    """
    times: list[float] = []
    Q: list[np.ndarray] = []
    t = start_time

    # 1) path_place (ends at q_grasp, stays closed)
    t = _append_path(times, Q, t, path_place)

    # 2) pause BEFORE opening
    t = _hold(times, Q, t, q_grasp, pause_before_action)

    # 3) OPEN moment (no motion)
    t_open = t

    # 4) pause AFTER opening
    t = _hold(times, Q, t, q_grasp, pause_after_action)

    # 5) path_upright (ends at q_upright)
    # t = _append_path(times, Q, t, path_upright)

    q_samples = np.stack(Q, axis=1)
    traj_q = PiecewisePolynomial.FirstOrderHold(times, q_samples)

    wsg_knots = [times[0], t_open, times[-1]]
    wsg_vals = [closed, opened, opened]
    traj_wsg = PiecewisePolynomial.ZeroOrderHold(
        wsg_knots, np.asarray(wsg_vals).reshape(1, -1)
    )

    print(f"[build_trajs_place] q_samples shape: {q_samples.shape}, T={times[-1]:.3f}s")
    return traj_q, traj_wsg


def build_trajs_pick(
    path_pick: list[np.ndarray],
    path_upright: list[np.ndarray],
    q_grasp: np.ndarray,
    start_time: float = 0.0,
) -> tuple[PiecewisePolynomial, PiecewisePolynomial]:
    """
    Pick-phase trajectory builder.

    Sequence:
      1) path_pick → q_grasp (gripper OPEN)
      2) (optional) move to q_grasp if last waypoint != q_grasp
      3) pause_before_action at q_grasp (open)
      4) CLOSE at t_close (no motion)
      5) pause_after_action at q_grasp (closed)
      6) path_upright
      7) pause_before_action at final pose (no motion)
    """
    times: list[float] = []
    Q: list[np.ndarray] = []
    t = 0.0

    # 1) path_pick (ends at q_grasp ideally)
    t = _append_path(times, Q, t, path_pick)

    # 2) move to q_grasp (WSG stays OPEN)
    if not np.allclose(Q[-1], q_grasp):
        t += 10 * dt
        times.append(t)
        Q.append(np.asarray(q_grasp, dtype=float))

    # 3) pause BEFORE CLOSE
    t = _hold(times, Q, t, q_grasp, pause_before_action)

    # 4) CLOSE (no motion)
    t_close = t
    t = _hold(times, Q, t, q_grasp, pause_after_action)

    # 5) (optional) extra move to q_grasp again (usually redundant)
    if not np.allclose(Q[-1], q_grasp):
        t += 10 * dt
        times.append(t)
        Q.append(np.asarray(q_grasp, dtype=float))

    # 6) path_upright
    t = _append_path(times, Q, t, path_upright)

    # 7) pause BEFORE OPEN (no motion)
    t = _hold(times, Q, t, Q[-1], pause_before_action)

    q_samples = np.stack(Q, axis=1)
    traj_q = PiecewisePolynomial.FirstOrderHold(times, q_samples)

    wsg_knots = [times[0], t_close, times[-1]]
    wsg_vals = [opened, closed, closed]
    traj_wsg = PiecewisePolynomial.ZeroOrderHold(
        wsg_knots, np.asarray(wsg_vals).reshape(1, -1)
    )

    print(f"[build_trajs_pick] q_samples shape: {q_samples.shape}, T={times[-1]:.3f}s")
    return traj_q, traj_wsg

