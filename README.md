# linear-systems-of-equations
Solving linear systems of equations

__solve_linear_system.py file implements an overconstrained, homogeneous linear system by using SVD factorization method. The module supports:__        

• Initializing with a fixed constraint matrix A and storage of whatever internal variables needed for repeatedly minimizing.    

• Acceptance of a right-hand side b, and computation of the least-squares optimal x for the given b   

• Saving the state of an initialized linear system to a file   

• Loading of a saved linear system   

• Computation of the norm of the residual ||Ax − b||<sub>2</sub>.    
   
   
# Explanation:   
  
First, initialize the matrix A by calling the constructor for the class **LinearSystem**. It also initializes the necessary internal variables (U, D and V) using numpy's linalg.svd() funactionality.   

To solve a linear system for a given b, call the function **compute_optimal_x**.    

To save the state of an already initialized system, simply call the method **save_state**, and provide the file name (in .npz format) to store into.     

To load an already initialized system, call the method **load_state**. This method will return the instance of LinearSystem.   

Finally, to compute the norm of the residual ||Ax − b||<sub>2</sub>, call the method **residual_norm** and provide values for b and x. There is no need to provide value for A, since it has already been initialized.   

     
# How to call each function (Example):      
   
**Initialize the linear system object with matrix A of size n X m, where m < n**      
```
A = np.random.rand(5, 3)   
linear_system = LinearSystem(A)   
print("A is:", A)   
```
   
**Accept b and solve for x**
```
b = np.random.rand(5)     
print("b is:", b)     
  
x = linear_system.compute_optimal_x(b)   
print("Solved x is:", x)      
```
   
**Compute residual norm**   
```
residual_norm = linear_system.residual_norm(b, x)   
print("Residual norm: ", residual_norm)   
```

**Save state to file**
```
linear_system.save_state('linear_system.npz')   
```
   
**Load state from file**   
```
lin_sys_loaded = LinearSystem.load_state('saved_linear_system.npz')     
print("Loaded system:", lin_sys_loaded)     
```
   
**Compute residual norm with our loaded system**
```
residual_norm = lin_sys_loaded.residual_norm(b, x)
print("Residual norm from loaded system:", residual_norm)   
```


