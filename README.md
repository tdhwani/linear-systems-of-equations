# linear-systems-of-equations
Solving linear systems of equations

linear_system_qr.py file implements an overconstrained, homogeneous linear system. The module supports:    

• Initializing with a fixed constraint matrix A and storage of whatever internal variables needed for repeatedly minimizing.    

• Acceptance of a right-hand side b, and computation of the least-squares optimal x for the given b   

• Saving the state of an initialized linear system to a file   

• Loading of a saved linear system   

• Computation of the norm of the residual ||Ax − b||2.    
   
     
**How to call each function (Example):**     
   
# create linear system object with matrix A of size n X m, where m < n      
A = np.random.rand(5, 3)   
linear_system = LinearSystem(A)   
print("A is:", A)   
   
# solve for b
b = np.random.rand(5)     
print("b is:", b)     
   
x = linear_system.compute_optimal_x(b)   
print("Solved x is:", x)   
print(x.shape)   
   
# save state to file
linear_system.save_state('linear_system.npz')   
   
# load state from file   
lin_sys_loaded = LinearSystem.load_state('saved_linear_system.npz')     
print("Loaded system:", lin_sys_loaded)     
   
# compute residual norm with our loaded system
residual_norm1 = lin_sys_loaded.residual_norm(b, x)
print("Residual norm from loaded system:", residual_norm1)   
     
# compute residual norm from scratch   
residual_norm2 = linear_system.residual_norm(b, x)   
print("Residual norm from scratch: ", residual_norm2)   

