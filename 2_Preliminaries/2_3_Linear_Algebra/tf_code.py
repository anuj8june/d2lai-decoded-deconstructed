import tensorflow as tf

# Scalars
x = tf.constant(3.0)
y = tf.constant(2.0)
print(f"\nScalar operation on x and y | x+y: {x+y} | x*y: {x*y} | x/y: {x/y} | x**y: {x**y}")

# Vectors
x = tf.range(3)
print(f"\nVector x: {x} | Value at index 2: {x[2]} | Length: {len(x)} | Shape: {x.shape}")

# Matrices
X = tf.reshape(tf.range(6), (3,2))
print(f"\nMatrix X:\n{X} \nTranspose on X: \n{tf.transpose(X)}")

A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(f"\nMatrix A: \n{A} | \nIs A symmetric?: \n{A==tf.transpose(A)}")

# Tensors
A = tf.reshape(tf.range(24), (2,3,4))
print(f"\nTensor: \n{A}")

# Basic properties of tensor arithmetic
A = tf.reshape(tf.range(6), (2,3))
B = A
print(f"\nA: \n{A} | \nA+B: \n{A+B}")

# Hadamard Product
print(f"\nHadamard product of A & B: \n{A*B}")

# Scalar addition and multiplication
A = tf.reshape(tf.range(24), (2,3,4))
print(f"\nScalar addition 2+A: \n{2+A} | \nScalar multiplication: \n{2*A}")

# Reduction sum
x = tf.range(3, dtype=tf.float32)
print(f"\nSum of X: {x} is {tf.reduce_sum(x)}")

A = tf.reshape(tf.range(6), (2,3))
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {tf.reduce_sum(A)}")
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {tf.reduce_sum(A, axis=0).shape}")
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {tf.reduce_sum(A, axis=1).shape}")
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {tf.reduce_sum(A, axis=[0,1])} | Both sums equal: {tf.reduce_sum(A, axis=[0,1])==tf.reduce_sum(A)}")
print(f"\nA: {A} | Mean: {tf.reduce_sum(A)/tf.size(A).numpy()} | Mean: {tf.reduce_mean(A)}")
print(f"\nA: {A} | Mean axis=0: {tf.reduce_sum(A, axis=0)/A.shape[0]} | Mean axis=0: {tf.reduce_mean(A, axis=0)}")

# Non reduction sum
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
print(f"\nA: \n{A} | \nsum A: {sum_A} | \nshape A: {sum_A.shape} | \nNormalisation along axis: {A/sum_A} | \nCumsum: {tf.cumsum(A, axis=0)}")

# Dot Product
x = tf.range(3, dtype=tf.float32)
y = tf.ones(3, dtype=tf.float32)
print(f"\nDot product of x {X} & y: {y} is {tf.tensordot(x,y, axes=1)} alternate way: {tf.reduce_sum(x*y)}")

# Matrix vector multiplication
A = tf.cast(A, dtype=tf.float32)
print(f"\nMatrix vector multiplication | A: {A} | x: {x} | A shape: {A.shape} | x shape: {x.shape} | mv : {tf.linalg.matvec(A,x)}")

# Matrix matrix multiplication
B = tf.ones((3,4), tf.float32)
print(f"\nMatrix multiplication of A: {A} & B: {B} is {tf.matmul(A,B)}")

# Norms
u = tf.constant([3.0, -4.0])
print(f"\nl2 norm of u: {u} is {tf.norm(u)}")
print(f"\nl1 norm of u: {u} is {tf.reduce_sum(tf.abs(u))}")
A = tf.cast(tf.ones((4,9)), dtype=tf.float32)
print(f"\nFrobenius norm of matrix A:{A} is {tf.norm(A)}")