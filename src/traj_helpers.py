from __future__ import annotations
import numpy as np
from pydrake.all import PiecewisePolynomial

dt = 0.05
pause_before_action = 1.0
pause_after_action = 1.0
opened, closed = 0.107, 0.0

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
    path_upright: list[np.ndarray],
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
    t = _append_path(times, Q, t, path_upright)

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
