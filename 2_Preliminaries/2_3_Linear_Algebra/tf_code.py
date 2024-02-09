import time
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


### Exercise
# 1. Prove that the transpose of the transpose of a matrix is the matrix itself: 
A = tf.cast(tf.reshape(tf.range(6), (2,3)), dtype=tf.float32)
print(f"\nA: {A} | If A.T.T==A: {A==tf.transpose(tf.transpose(A))}")

# 2.Given two matrices and , show that sum and transposition commute
A = tf.cast(tf.reshape(tf.range(6), (2,3)), dtype=tf.float32)
B = tf.ones((2,3), dtype=tf.float32)
sum_of_individual_transpose = tf.transpose(A) + tf.transpose(B)
sum_first_then_transpose = tf.transpose(A+B)
print(f"\nA: {A} | B: {B} | Is sum of individual transpose equal to sum first and then transpose: {sum_of_individual_transpose==sum_first_then_transpose}")

# 3. Given any square matrix A. Is A + A.T always symmetric?
A = tf.cast(tf.reshape(tf.range(9), (3,3)), dtype=tf.float32)
print(f"\nA: {A} | Is A + A.T symmetric?: {A + tf.transpose(A)}")

# 4. We defined the tensor X of shape (2, 3, 4) in this section. What is the output of len(X)?
A = tf.ones((2,3,3), dtype=tf.float32)
print(f"\nLength of A:{A} is {len(A)}")

# 5. or a tensor X of arbitrary shape, does len(X) always correspond to the length of a certain axis of X? What is that axis?
# 1st axis

# 6. Run A / A.sum(axis=1) and see what happens. Can you analyze the results?
A = tf.cast(tf.reshape(tf.range(6), (2,3)), dtype=tf.float32)
print(f"\nA/A.sum(axis=1) = {A/tf.reduce_sum(A, axis=1, keepdims=True)}")

# 7. When traveling between two points in downtown Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
# Find coordinates and use distance formula(l2 norm). Or use google maps for more precision.

# 8. Consider a tensor of shape (2, 3, 4). What are the shapes of the reduce_summation outputs along axes 0, 1, and 2?
A = tf.cast(tf.reshape(tf.range(24), (2,3,4)), dtype=tf.float32)
sum_axis0 = tf.reduce_sum(A, axis=0).shape
sum_axis1 = tf.reduce_sum(A, axis=1).shape
sum_axis2 = tf.reduce_sum(A, axis=2).shape
print(f"\nSum of A(2,3,4) along axis 0: {sum_axis0} along axis 1: {sum_axis1} along axis 2: {sum_axis2}")

# 9. Feed a tensor with three or more axes to the linalg.norm function and observe its output. What does this function compute for tensors of arbitrary shape?
A = tf.cast(tf.reshape(tf.range(6), (2,3)), dtype=tf.float32)
print(f"\nlinalg.norm output on A: {A} is :{tf.linalg.norm(A)}")

# 10. You want to compute the product ABC. Is there any difference in memory footprint and speed, depending on whether you compute (AB)C or A(BC). Why?
A = tf.random.normal((10,16))
B = tf.random.normal((16,5))
C = tf.random.normal((5,14))
start = time.process_time()
AB_C = tf.matmul(tf.matmul(A,B),C)
print(f"Time taken to execute (AB)C: {time.process_time()-start}")
start = time.process_time()
A_BC = tf.matmul(A,tf.matmul(B,C))
print(f"Time taken to execute A(BC): {time.process_time()-start}")

# 11. Is there any difference in speed depending on whether you compute AB or AC.T? Why? What changes if you initialize C=B.T without cloning memory? Why?
A = tf.random.normal((10,16))
B = tf.random.normal((16,5))
C = tf.random.normal((5,16))
start = time.process_time()
AB = tf.matmul(A,B)
print(f"Time taken to execute AB: {time.process_time()-start}")
start = time.process_time()
ACT = tf.matmul(A,tf.transpose(C))
print(f"Time taken to execute AC.T: {time.process_time()-start}")

C = tf.transpose(B)
start = time.process_time()
AB = tf.matmul(A,B)
print(f"Time taken to execute AB: {time.process_time()-start}")
start = time.process_time()
ACT = tf.matmul(A,tf.transpose(C))
print(f"Time taken to execute AC.T when C=B.T: {time.process_time()-start}")

# 12. Construct a tensor with three axes by stacking. What is the dimensionality? Slice out the second coordinate of the third axis to recover. Check that your answer is correct.
A = tf.reshape(tf.range(20000), (100,200))
B = tf.reshape(tf.range(20000,40000), (100,200))
C = tf.reshape(tf.range(40000,60000), (100,200))
ABC = tf.stack([A,B,C], axis=2)
print(f"After stacking ABC shape: {ABC.shape}. Extracting B out using slicing: {ABC[:,:,1]}")