import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        p = np.asarray(p, dtype=float).flatten()
        if p.size == 1:
            p = np.array([p[0], p[0]])

        self.L = np.array([
            [3 * p[0], 0], [0, 3 * p[1]],
            [3 * p[0] ** 2, 0], [0, 3 * p[1] ** 2],
            [p[0] ** 3, 0], [0, p[1] ** 3]
        ])

        W = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x_model = np.concatenate([q, q_dot])

        M = self.model.M(x_model)
        C = self.model.C(x_model)
        M_inv = np.linalg.inv(M)

        Z = np.zeros((2, 2))
        I = np.eye(2)

        self.eso.A = np.block([
            [Z, I, Z],
            [Z, -M_inv @ C, I],
            [Z, Z, Z]
        ])

        self.eso.B = np.vstack([Z, M_inv,Z])

        return M, C

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        z_hat = self.eso.get_state()

        q_hat = z_hat[:2]
        q_dot_hat = z_hat[2:4]
        f_hat = z_hat[4:6]

        M, C = self.update_params(q_hat, q_dot_hat)

        e = q_d - q_hat
        e_dot = q_d_dot - q_dot_hat

        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e

        u = M @ (v - f_hat) + C @ q_dot_hat

        q = np.array(x[:2])
        self.eso.update(q, u)

        return u
