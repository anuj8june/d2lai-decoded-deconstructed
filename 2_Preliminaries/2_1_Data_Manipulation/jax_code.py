import jax
from jax import numpy as jnp

# Creating a one dimensional tensor
x = jnp.arange(12)
print(f"\nOutput of creating a one dimensional tensor x: {x}")

# Check size of tensor
print(f"\nSize of created tensor x: {x.size}")
print(f"\nSize of created tensor x: {x.shape}")

# Reshape the vector to a matrix
X = x.reshape(3,4)
X_1 = x.reshape(3,-1)
X_2 = x.reshape(-1,4)
print(f"\nOutput after reshape X: \n{X}")
print(f"\nSize of created matrix X: {X.size}")
print(f"\nSize of created matrix X: {X.shape}")
print(f"\nIf X & X_1 are same: {jnp.array_equal(X, X_1)} | If X & X_2 are same: {jnp.array_equal(X, X_2)}")

# Creating tensor with all zeros
X = jnp.zeros((2,3,4))
print(f"\nTensor with all zeros X: {X}")

# Creating tensor with all ones
X = jnp.ones((2,3,4))
print(f"\nTensor with all ones X: {X}")

# Tensor with values from normal distribution
X = jax.random.normal(jax.random.PRNGKey(0), (3,4))
print(f"\nTensor initialized with values from normal distribution X: \n{X}")

# Tensor with all constants.
X = jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(f"\nTensor initialized manually x: \n{x}")

# Indexing and slicing
print(f"\nOutput of x[-1]: \n{x[-1]} \nOutput of x[1:3]: \n{x[1:3]}")

# Writing elements to array. 
X_new_1 = X.at[1,2].set(17)
print(f"\nAfter changing values of elements in X_new_1 : \n{X_new_1}")

# Assigning multiple values at the same time. 
X_new_2 = X.at[:2,:].set(12)
print(f"\nAfter changing values of elements in X_new_1 : \n{X_new_2}")


# Exponential operator
print(f"\nOuput of e^x : \n{jnp.exp(x)}")

# All elementwise operation
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2.0, 2, 2, 2])
print(f"\nOuput of operation of x & y \nAddition: {x+y} \nSubtrction: {x-y} \nMultiplication: {x*y} \nDivision: {x/y} \nPower Function: {x**y}")

# Concatenation operation
X = jnp.arange(12, dtype=jnp.float32).reshape((3,4))
Y = jnp.array([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(f"\nOutput of concatenation along 0 axis: \n{jnp.concatenate((X,Y), axis=0)}")
print(f"\nOutput of concatenation along 1 axis: \n{jnp.concatenate((X,Y), axis=1)}")

# Check if the values are equal
print(f"\nEqual elements in X & Y: \n{X==Y}")

# Sum of all elements in matrix
print(f"\nSum of all elements in matrix: {jnp.sum(X)}")

# Broadcasting
a = jnp.arange(3).reshape((3,1))
b = jnp.arange(2).reshape((1,2))
print(f"\nInitalized arrays with different shapes a: \n{a} and b: {b}")
print(f"\nAddition after broadcasting a+b: \n{a+b}")

# Memory Allocations
before = id(Y)
Y = Y + X
print(f"\nIs the memory location same as before : {id(Y) == before}")

# Inplace update of operation is not allowed in jax

# Conversion to numpy object and back
A = jax.device_get(X)
B = jax.device_put(A)
print(f"\nType A : {type(A)} | Type B: {type(B)}")

# Type conversions
a = jnp.array([3.5])
print(f"\nOutput of a: {a} | a.item(): {a.item()} | float(a): {float(a)} | int(a): {int(a)}")


# Exercise
# 1. Replace X==Y with X>Y and X<Y
X = jnp.arange(12, dtype=jnp.float32).reshape((3,4))
Y = jnp.array([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(f"\nOutput ofX>Y: \n{X>Y}")
print(f"\nOutput ofX<Y: \n{X<Y}")

# 2. Broadcasting
X = jnp.arange(24, dtype=jnp.float32).reshape((2,3,4))
Y = jnp.array([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(f"\nOutput ofX>Y: \n{X>Y}")
print(f"\nOutput ofX<Y: \n{X<Y}")