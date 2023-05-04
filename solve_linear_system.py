import numpy as np


class LinearSystem:

    def __init__(self, A):
        # Initialize matrix A, and compute values for U, S, and V for SVD factorization
        self.A = A
        self.U, self.S, self.V = np.linalg.svd(A)

    def compute_optimal_x(self, b):
        # Compute the least-squares optimal x for the given b, using SVD factorization
        UTb = np.dot(self.U.T, b)  # UTb represents (U Transpose).b
        s_inv = np.zeros_like(self.A.T)
        s_inv[:self.S.shape[0], :self.S.shape[0]] = np.diag(1/self.S)
        sinv_dot_UTb = np.dot(s_inv, UTb)
        x = np.dot(self.V.T, sinv_dot_UTb)
        return x

    def save_state(self, filename):
        # Save the state of an initialized linear system to a file
        np.savez(filename, A=self.A, U=self.U, S=self.S, V=self.V)

        print("State saved to: ", filename)

    @classmethod
    def load_state(cls, filename):
        # Load saved state from file
        state = np.load(filename)
        A = state['A']
        U = state['U']
        S = state['S']
        V = state['V']
        linear_system = cls(A)
        linear_system.U = U
        linear_system.S = S
        linear_system.V = V
        return linear_system

    def residual_norm(self, b, x):
        # Compute the norm of the residual.
        residual = np.dot(self.A, x) - b
        norm = np.linalg.norm(residual)
        return norm
