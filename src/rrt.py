from __future__ import annotations
import numpy as np
from pydrake.geometry import Sphere, Rgba

class RRTNode:
	__slots__ = ("q", "parent")

	def __init__(self, q, parent):
		self.q = np.asarray(q, dtype=float)
		self.parent = parent


class RRTTools:
	def __init__(
		self,
		collision_checker,
		q_lo,
		q_hi,
		df_start,
		df_end,
		step_size: float = 0.1,
		goal_threshold: float = 0.2,
		rng=None,
	):
		self.collision_checker = collision_checker
		self.df_start = df_start
		self.df_end = df_end

		self.q_lo_full = np.asarray(q_lo, dtype=float)
		self.q_hi_full = np.asarray(q_hi, dtype=float)
		self.q_lo = self.q_lo_full[df_start:df_end]
		self.q_hi = self.q_hi_full[df_start:df_end]

		self.step_size = float(step_size)
		self.goal_threshold = float(goal_threshold)

		self.rng = np.random.default_rng() if rng is None else rng

		# Sampling bounds for active dims
		samp_lo = self.q_lo.copy()
		samp_hi = self.q_hi.copy()

		mask = ~np.isfinite(samp_lo) | ~np.isfinite(samp_hi)
		if np.any(mask):
			center = 0.0
			span = 1.0
			samp_lo[mask] = center - span
			samp_hi[mask] = center + span

		self.samp_lo = samp_lo
		self.samp_hi = samp_hi

		# Filled in during plan()
		self.fixed_head = None
		self.fixed_tail = None


	def steer(self, q_from, q_to):
		"""Move from q_from toward q_to by at most step_size (active dims)."""
		q_from = np.asarray(q_from, dtype=float)
		q_to = np.asarray(q_to, dtype=float)

		dq_active = q_to[self.df_start:self.df_end] - q_from[self.df_start:self.df_end]
		d = np.linalg.norm(dq_active)

		if d <= self.step_size:
			return q_to.copy()

		q_new = q_from.copy()
		q_new[self.df_start:self.df_end] = (
			q_from[self.df_start:self.df_end]
			+ (self.step_size / max(d, 1e-12)) * dq_active
		)
		return q_new

	def sample_config(self):
		q_active = self.rng.uniform(self.samp_lo, self.samp_hi)
		q = np.concatenate([self.fixed_head, q_active, self.fixed_tail])
		return q

	@staticmethod
	def nearest(tree, q):
		qs = np.stack([n.q for n in tree], axis=0)
		idx = int(np.argmin(np.linalg.norm(qs - q, axis=1)))
		return idx

	@staticmethod
	def add_node(tree, q, parent_idx):
		tree.append(RRTNode(q, parent_idx))
		return len(tree) - 1

	@staticmethod
	def build_path(tree, idx):
		path = []
		while idx is not None:
			node = tree[idx]
			path.append(node.q)
			idx = node.parent
		return list(reversed(path))

	def connect_greedy(self, tree, q_target):
		checker = self.collision_checker
		step_size = self.step_size

		idx_curr = self.nearest(tree, q_target)
		q_curr = tree[idx_curr].q

		while True:
			d = np.linalg.norm(q_target - q_curr)
			if d < step_size:
				if checker.CheckEdgeCollisionFree(q_curr, q_target):
					idx_new = self.add_node(tree, q_target, idx_curr)
					return idx_new, True
				return idx_curr, False

			q_next = self.steer(q_curr, q_target)
			if not checker.CheckEdgeCollisionFree(q_curr, q_next):
				return idx_curr, False

			idx_next = self.add_node(tree, q_next, idx_curr)
			idx_curr = idx_next
			q_curr = tree[idx_curr].q
			
	def plan(self, q_start, q_goal, max_iterations=10000):
		checker = self.collision_checker

		if not checker.CheckConfigCollisionFree(q_start):
			print("[RRT] Start configuration is in collision.")
			return None
		if not checker.CheckConfigCollisionFree(q_goal):
			print("[RRT] Goal configuration is in collision.")
			return None

		T_start = [RRTNode(q_start, None)]
		T_goal  = [RRTNode(q_goal, None)]

		# Fixed dims = everything outside [df_start:df_end]
		self.fixed_head = np.asarray(q_start[:self.df_start], dtype=float)
		self.fixed_tail = np.asarray(q_start[self.df_end:], dtype=float)

		for it in range(max_iterations):
			if it % 1000 == 0:
				print(f"[RRT] iteration {it}")

			q_rand = self.sample_config()

			# Alternate which tree grows
			if it % 2 == 0:
				Ta, Tb = T_start, T_goal
			else:
				Ta, Tb = T_goal, T_start

			idx_near = self.nearest(Ta, q_rand)
			q_near = Ta[idx_near].q
			q_new = self.steer(q_near, q_rand)

			if not checker.CheckEdgeCollisionFree(q_near, q_new):
				continue

			idx_new = self.add_node(Ta, q_new, idx_near)
			q_a = Ta[idx_new].q
			idx_b, complete = self.connect_greedy(Tb, q_a)

			if complete:
				print(f"[RRT] Connected in {it+1} iterations")

				path_a = self.build_path(T_start, idx_new if Ta is T_start else idx_b)
				path_b = self.build_path(T_goal,  idx_b if Tb is T_goal  else idx_new)

				if Ta is T_goal:
					path_a, path_b = path_b, path_a

				# path_b is connection→goal; avoid double-counting connection
				return path_a + path_b[-2::-1]

		print("[RRT] Failed to find a path.")
		return None

	# Shortcutting helpers
	@staticmethod
	def check_equal(a, b, atol: float = 1e-9) -> bool:
		return np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=0.0)

	@staticmethod
	def splice_with_shortcut(path, i, j, edge):
		return path[:i] + edge + path[j+1:]

	@staticmethod
	def interpolate_edge(q_i, q_j, max_step):
		q_i = np.asarray(q_i, dtype=float)
		q_j = np.asarray(q_j, dtype=float)
		d = np.linalg.norm(q_j - q_i)

		if d < 1e-12:
			return [q_i.copy(), q_j.copy()]

		n_steps = max(1, int(np.ceil(d / max_step)))
		ts = np.linspace(0.0, 1.0, n_steps + 1)
		return [(1 - t) * q_i + t * q_j for t in ts]

	def shortcut_path(self, path, passes=200, min_separation=2, max_step=0.1):
		if not path or len(path) < 3:
			return path

		checker = self.collision_checker
		rng = self.rng
		current = [np.asarray(q, dtype=float) for q in path]

		for _ in range(passes):
			n = len(current)
			if n < 3:
				break

			i = int(rng.integers(0, n - min_separation))
			j = int(rng.integers(i + min_separation, n))

			q_i, q_j = current[i], current[j]
			edge = self.interpolate_edge(q_i, q_j, max_step)

			collision_free = True
			for k in range(len(edge) - 1):
				if not checker.CheckEdgeCollisionFree(edge[k], edge[k+1]):
					collision_free = False
					break

			if not collision_free:
				continue

			if not (self.check_equal(edge[0], q_i) and self.check_equal(edge[-1], q_j)):
				continue

			current = self.splice_with_shortcut(current, i, j, edge)

		cleaned = []
		prev = None
		for q in current:
			if prev is None or not self.check_equal(q, prev):
				cleaned.append(q)
			prev = q

		return cleaned
	
	def visualize_rrt_waypoints(rrt_path, station, meshcat, robot_name="mobile_iiwa", ee_body_name="iiwa_link_ee"):
		plant = station.GetSubsystemByName("plant")
		context = station.CreateDefaultContext()
		plant_context = plant.GetMyMutableContextFromRoot(context)

		robot_instance = plant.GetModelInstanceByName(robot_name)
		ee_body = plant.GetBodyByName(ee_body_name, robot_instance)
		meshcat.Delete("rrt_waypoints")
		sphere = Sphere(0.03)

		for i, q in enumerate(rrt_path):
			q = np.asarray(q).flatten()
			plant.SetPositions(plant_context, q)
			X_WG = plant.EvalBodyPoseInWorld(plant_context, ee_body)

			# Color: green start, red goal, blue intermediates
			if i == 0:
				color = Rgba(0.0, 1.0, 0.0, 0.8)
			elif i == len(rrt_path) - 1:
				color = Rgba(1.0, 0.0, 0.0, 0.8)
			else:
				color = Rgba(0.0, 0.0, 1.0, 0.5)

			path = f"rrt_waypoints/wp_{i:03d}"
			meshcat.SetObject(path, sphere, color)
			meshcat.SetTransform(path, X_WG)

		print(f"visualized {len(rrt_path)} RRT waypoints in Meshcat.")