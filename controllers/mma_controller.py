import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.Tp = Tp
        self.models = []

        params = [(0.1, 0.05), (0.01, 0.01), (1.0, 0.3)]

        for m3, r3 in params:
            model = ManiuplatorModel(Tp)
            model.m3 = m3
            model.r3 = r3
            model.I_3 = 2. / 5 * model.m3 * model.r3 ** 2
            self.models.append(model)
        self.i = 0
        self.x_prev = None
        self.u_prev = np.zeros(2)
        self.selected_model_history = []

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        if self.x_prev is None:
            self.x_prev = np.array(x, dtype=float)
            return
        
        x = np.array(x, dtype=float)
        x_prev = self.x_prev

        q_dot_prev = x_prev[2:]  #prędkości q1,q2
        q_ddot_real = (x[2:] - x_prev[2:]) / self.Tp #przyśpiesznia q1_dot, q2_dot, rzeczywiste przyspieszenie = zmiana prędkości / czas próbkowania

        errors = []

        for model in self.models:
            M = model.M(x_prev)
            C = model.C(x_prev)

            q_ddot_model = np.linalg.inv(M) @ (self.u_prev - C @ q_dot_prev)

            error = np.linalg.norm(q_ddot_real - q_ddot_model)
            errors.append(error)

        self.i = int(np.argmin(errors))

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        self.selected_model_history.append(self.i)

        q = x[:2]
        q_dot = x[2:]
        Kp = np.diag([50, 50])
        Kd = np.diag([10, 10])

        e = q_r - q
        e_dot = q_r_dot - q_dot

        v = q_r_ddot + Kd @ e_dot + Kp @ e
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v + C @ q_dot
        self.x_prev = np.array(x, dtype=float)
        self.u_prev = u.copy()
        return u
