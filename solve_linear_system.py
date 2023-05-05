import numpy as np


class LinearSystem:

    def __init__(self, A):
        # Initialize matrix A, and compute values for U, D, and V for SVD factorization
        self.A = A
        self.U, self.D, self.V = np.linalg.svd(A)

    def compute_optimal_x(self, b):
        # Compute the least-squares optimal x for the given b, using SVD factorization
        
        UTb = np.dot(self.U.T, b)  # UTb represents (U Transpose).b
        
        threshold = np.max(self.A.shape) * np.finfo(self.A.dtype).eps * self.D[0]  

        d_inv = np.zeros_like(self.A.T)  # First, initialize as matrix of zeros with same shape as A transpose
        d_inv[:self.D.shape[0], :self.D.shape[0]] = np.diag(np.where(self.D > threshold, 1 / self.D, 0))  # Now, only the diagonal elements (above a certain threshold) are non-zero

        dinv_dot_UTb = np.dot(d_inv, UTb)

        x = np.dot(self.V.T, dinv_dot_UTb)  # x = (V Transpose).(d inverse).(U Transpose).b
        return x

    def save_state(self, filename):
        # Save the state of an initialized linear system to a file
        np.savez(filename, A=self.A, U=self.U, D=self.D, V=self.V)

        print("State saved to: ", filename)

    @classmethod
    def load_state(cls, filename):
        # Load saved state from file
        state = np.load(filename)
        A = state['A']
        U = state['U']
        D = state['D']
        V = state['V']
        linear_system = cls(A)
        linear_system.U = U
        linear_system.D = D
        linear_system.V = V
        return linear_system

    def residual_norm(self, b, x):
        # Compute the norm of the residual.
        residual = np.dot(self.A, x) - b
        norm = np.linalg.norm(residual)
        return norm
