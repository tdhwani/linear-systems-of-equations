import numpy as np


class LinearSystem:

    def __init__(self, A):
        # Initialize matrix A, and compute values for Q and R for QR factorization
        self.A = A
        self.Q, self.R = np.linalg.qr(A)
        # self.Q, self.R = self.qr_factorization(A)

    def compute_optimal_x(self, b):
        # Compute the least-squares optimal x for the given b, using QR factorization
        QTb = np.dot(self.Q.T, b)   # QTb represents (Q Transpose).b
        x = np.linalg.solve(self.R, QTb)   # Solve the equation: Rx = QTb
        return x

    def save_state(self, filename):
        # Save the state of an initialized linear system to a file
        np.savez(filename, A=self.A, Q=self.Q, R=self.R)
        print("State saved to: ", filename)

    @classmethod
    def load_state(cls, filename):
        # Load saved state from file
        state = np.load(filename)
        A = state['A']
        Q = state['Q']
        R = state['R']
        linear_system = cls(A)
        linear_system.Q = Q
        linear_system.R = R
        return linear_system

    def residual_norm(self, b, x):
        # Compute the norm of the residual.
        residual = np.dot(self.A, x) - b
        norm = np.linalg.norm(residual)
        return norm