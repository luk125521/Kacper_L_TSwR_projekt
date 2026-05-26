import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp,use_feedback=True, m3=0.0, r3=0.05):
        self.model = ManiuplatorModel(Tp,m3=m3,r3=r3)
        self.use_feedback = use_feedback

        self.Kp = np.diag([50.0, 50.0])
        self.Kd = np.diag([10.0, 10.0])
    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x

        q = np.array([q1, q2])
        q_dot = np.array([q1_dot, q2_dot])

        M = self.model.M(x)
        C = self.model.C(x)
        if self.use_feedback:
            e = q_r - q
            e_dot = q_r_dot - q_dot
            v = q_r_ddot + self.Kd @ e_dot + self.Kp @ e
        else:
            v = q_r_ddot

        u = M @ v + C @ q_dot

        return u
        #return NotImplementedError()
