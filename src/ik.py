import numpy as np
from pydrake.all import (
    MultibodyPlant,
    Context,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    Solve,
)


class IKSolver:
    def __init__(
        self,
        plant: MultibodyPlant,
        plant_context: Context,
        *,
        ix: int,
        iy: int,
        iz: int,
        active_hi: int,
        q0: np.ndarray,
        gripper_body,
        theta_bound: float = 0.01 * np.pi,
        pos_tol: float = 0.02,
        base_tol: float = 0.2,
    ) -> None:
        self.plant = plant
        self.context = plant_context
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.active_hi = active_hi
        self.q0 = np.asarray(q0, dtype=float).copy()
        self.gripper_body = gripper_body

        self.theta_bound = theta_bound
        self.pos_tol = pos_tol
        self.base_tol = base_tol

    def solve(self, X_WG_target: RigidTransform, q0new) -> np.ndarray:
        world_frame = self.plant.world_frame()
        gripper_frame = self.plant.GetFrameByName("body")

        ik = InverseKinematics(self.plant, self.context,)
        ik.AddPositionConstraint(gripper_frame, np.zeros(3), 
                                world_frame, X_WG_target.translation() - self.pos_tol,
                                X_WG_target.translation() + self.pos_tol)
        ik.AddOrientationConstraint(world_frame, X_WG_target.rotation(),
                                    gripper_frame, RotationMatrix(np.eye(3)), self.theta_bound)
        ik.AddMinimumDistanceLowerBoundConstraint(0.01)

        # Solve IK problem
        prog = ik.get_mutable_prog()
        q_dec = ik.q()
        prog.AddBoundingBoxConstraint(q0new[[self.ix,self.iy]]-self.base_tol, q0new[[self.ix,self.iy]]+self.base_tol, q_dec[[self.ix, self.iy]])
        prog.AddBoundingBoxConstraint(q0new[self.iz], q0new[self.iz]+self.pos_tol, q_dec[self.iz])
        prog.SetInitialGuess(q_dec, q0new)
        result = Solve(prog)

        if result.is_success():
            q_ik = result.GetSolution(ik.q())
            q_ik[self.active_hi:] = self.q0[self.active_hi:]
            # self.plant.SetPositions(self.plant_context, q_ik)
            # X_WG_achieved = self.plant.EvalBodyPoseInWorld(self.plant_context, self.gripper_body)
            # print(f"base position: ({q_ik[self.ix]:.2f}, {q_ik[self.iy]:.2f}, {q_ik[self.iz]:.2f})")
            # print(f"Achieved gripper position: {X_WG_achieved.translation()}")
            return q_ik
        else:
            print(f"Infeasible constraints: {result.GetInfeasibleConstraintNames(prog)}")
            return None

