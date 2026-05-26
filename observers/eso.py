from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        z_hat = self.state             # estymowany stan części obserwowwalnej
        y_hat = self.W @ z_hat         # estymowane wyjście

        e_o = q - y_hat                #y−C*z_hat=C(z−z_hat), q=y=C*z
        #zad3.10
        if self.L.ndim == 1:
            correction = self.L * e_o
        else:
            correction = self.L @ e_o
        z_dot = self.A @ z_hat + self.B @ u + correction

        # Euler
        self.state = z_hat + self.Tp * z_dot #z_(k+1) = z_k +T_p * z_dot

    def get_state(self):
        return self.state
