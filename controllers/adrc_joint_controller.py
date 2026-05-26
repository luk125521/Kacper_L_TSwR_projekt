import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.u_prev = 0.0

        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0], [b], [0]])
        W = np.array([1, 0, 0])
        # gain obserwatora
        L = np.array([3 * p, 3 * p ** 2, p ** 3])

        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        B = np.array([[0], [b], [0]])
        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        q, q_dot = x
        if not hasattr(self, 'u_prev'):
            self.u_prev = 0.0

        self.eso.update(np.array([q]), np.array([self.u_prev])) #ESO potrzebuję u(k-1) dlatego u_prev

        z_hat = self.eso.get_state()

        q_hat = z_hat[0] #pozycja
        q_dot_hat = z_hat[1] #prędkość
        f_hat = z_hat[2] #zakłócenia
        # Celem sterowania: q_ddot = v, podstawiając do modelu: b*u + f = v, względem sterowania: u = (v - f) / b, nie znamy f więc f_hat
        e = q_d - q_hat
        e_dot = q_d_dot - q_dot_hat

        v = q_d_ddot + self.kd * e_dot + self.kp * e

        # ADRC
        u = (v - f_hat) / self.b
        self.u_prev = u

        return float(u)
