# linear-systems-of-equations
Solving linear systems of equations

linear_system_qr.py file implements an overconstrained, homogeneous linear system. The module supports:    

• Initializing with a fixed constraint matrix A and storage of whatever internal variables needed for repeatedly minimizing.    

• Acceptance of a right-hand side b, and computation of the least-squares optimal x for the given b   

• Saving the state of an initialized linear system to a file   

• Loading of a saved linear system   

• Computation of the norm of the residual ||Ax − b||2.    

