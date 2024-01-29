from jax import numpy as jnp

# Scalars
x = jnp.array(3.0)
y = jnp.array(2.0)
print(f"\nScalar operation on x and y | x+y: {x+y} | x*y: {x*y} | x/y: {x/y} | x**y: {x**y}")

# Vectors
x = jnp.arange(3)
print(f"\nVector x: {x} | Value at index 2: {x[2]} | Length: {len(x)} | Shape: {x.shape}")

# Matrices
X = jnp.arange(6).reshape(3,2)
print(f"\nMatrix X:\n{X} \nTranspose on X: \n{X.T}")

A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(f"\nMatrix A: \n{A} | \nIs A symmetric: \n{A==A.T}")

# Tensors
A = jnp.arange(24).reshape(2,3,4)
print(f"\nTensor: \n{A}")

# Basic properties of tensor arithmetic
A = jnp.arange(6).reshape(2,3)
B = A
print(f"\nA: \n{A} | \nA+B: \n{A+B}")

# Hadamard Product
print(f"\nHadamard product of A & B: \n{A*B}")

# Scalar addition and multiplication
A = jnp.arange(24).reshape(2,3,4)
print(f"\nScalar addition 2+A: \n{2+A} | \nScalar multiplication: \n{2*A}")

# Reduction sum
x = jnp.arange(3, dtype=jnp.float32)
print(f"\nSum of X: {x} is {x.sum()}")

A = jnp.arange(6).reshape(2,3)
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {A.sum()}")
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {A.sum(axis=0).shape}")
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {A.sum(axis=1).shape}")
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {A.sum(axis=[0,1])} | Both sums equal: {A.sum(axis=[0,1])==A.sum()}")
print(f"\nA: {A} | Mean: {A.sum()/A.size} | Mean: {A.mean()}")
print(f"\nA: {A} | Mean axis=0: {A.sum(axis=0)/A.shape[0]} | Mean axis=0: {A.mean(axis=0)}")

# Non reduction sum
sum_A = A.sum(axis=1, keepdims=True)
print(f"\nA: \n{A} | \nsum A: {sum_A} | \nshape A: {sum_A.shape} | \nNormalisation along axis: {A/sum_A} | \nCumsum: {A.cumsum(axis=0)}")

# Dot Product
x = jnp.arange(3, dtype=jnp.float32)
y = jnp.ones(3, dtype=jnp.float32)
print(f"\nDot product of x {X} & y: {y} is {jnp.dot(x,y)} alternate way: {jnp.sum(x*y)}")

# Matrix vector multiplication
print(f"\nMatrix vector multiplication | A: {A} | x: {x} | A shape: {A.shape} | x shape: {x.shape} | mv : {jnp.matmul(A,x)}")

# Matrix matrix multiplication
B = jnp.ones((3,4), jnp.float32)
print(f"\nMatrix multiplication of A: {A} & B: {B} is {jnp.matmul(A,B)}")

# Norms
u = jnp.array([3.0, -4.0])
print(f"\nl2 norm of u: {u} is {jnp.linalg.norm(u)}")
print(f"\nl1 norm of u: {u} is {jnp.abs(u).sum()}")
A = jnp.ones((4,9))
print(f"\nFrobenius norm of matrix A:{A} is {jnp.linalg.norm(A)}")
