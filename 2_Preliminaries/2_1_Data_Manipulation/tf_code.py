import tensorflow as tf

# Creating a one dimensional tensor 
x = tf.range(12, dtype=tf.float32)
print(f"\nOutput of creating a one dimensional tensor x: {x}")

# Check size of tensor
print(f"\nSize of created tensor x: {tf.size(x)}")
print(f"\nSize of created tensor x: {x.shape}")

# Reshape the vector to a matrix
X = tf.reshape(x, (3,4))
X_1 = tf.reshape(x, (3,-1))
X_2 = tf.reshape(x, (-1,4))
print(f"\nOutput after reshape X: \n{X}")
print(f"\nSize of created matrix X: {tf.size(X)}")
print(f"\nSize of created matrix X: {X.shape}")
print(f"\nIf X & X_1 are same: {tf.math.reduce_all(tf.equal(X,X_1))} | If X & X_2 are same: {tf.math.reduce_all(tf.equal(X,X_2))}")

# Creating tensor with all zeros
X = tf.zeros((2,3,4))
print(f"\nTensor with all zeros X: {X}")

# Creating tensor with all ones
X = tf.ones((2,3,4))
print(f"\nTensor with all ones X: {X}")

# Tensor with values from normal distribution
X = tf.random.normal(shape=(3,4))
print(f"\nTensor initialized with values from normal distribution X: \n{X}")

# Tensor with all constants.
X = tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(f"\nTensor initialized manually X: \n{X}")

# Indexing and slicing
print(f"\nOutput of X[1]: \n{X[-1]} \nOutput of X[1:3]: \n{X[1:3]}")

# Writing elements to array. Tensors are immutable.
X_var = tf.Variable(X)
X_var[1,2].assign(9)
print(f"\nAfter changing values of elements in X_var : \n{X_var}")

# Assigning multiple values at the same time. Tensors are immutable.
X_var = tf.Variable(X)
X_var[:2,:].assign(9)
print(f"\nAfter changing values of elements in X_var : \n{X_var}")


# Exponential operator
print(f"\nOuput of e^x : \n{tf.exp(x)}")

# All elementwise operation
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
print(f"\nOuput of operation of x & y \nAddition: {x+y} \nSubtrction: {x-y} \nMultiplication: {x*y} \nDivision: {x/y} \nPower Function: {x**y}")

# Concatenation operation
X = tf.range(12, dtype=tf.float32)
X = tf.reshape(X, (3,4))
Y = tf.constant([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(f"\nOutput of concatenation along 0 axis: \n{tf.concat([X,Y], axis=0)}")
print(f"\nOutput of concatenation along 1 axis: \n{tf.concat([X,Y], axis=1)}")

# Check if the values are equal
print(f"\nEqual elements in X & Y: \n{X==Y}")

# Sum of all elements in matrix
print(f"\nSum of all elements in matrix: {tf.reduce_sum(X)}")

# Broadcasting
a = tf.reshape(tf.range(3), (3,1))
b = tf.reshape(tf.range(2), (1,2))
print(f"\nInitalized arrays with different shapes a: \n{a} and b: {b}")
print(f"\nAddition after broadcasting a+b: \n{a+b}")

# Memory Allocations
before = id(Y)
Y = Y + X
print(f"\nIs the memory location same as before : {id(Y) == before}")

# Inplace update of operation
Z = tf.Variable(tf.zeros_like(Y))
print(f"\nAddress of Z before operation: {id(Z)}")
Z.assign(X + Y)   # This .assign updates the values in the same memory address
print(f"\nAddress of Z after operation: {id(Z)}")

# Conversion to numpy object and back
A = X.numpy()
B = tf.constant(A)
print(f"\nType A : {type(A)} | Type B: {type(B)}")

# Type conversions
a = tf.constant([3.5]).numpy()
print(f"\nOutput of a: {a} | a.item(): {a.item()} | float(a): {float(a)} | int(a): {int(a)}")


# Exercise
# 1. Replace X==Y with X>Y and X<Y
X = tf.reshape(tf.range(12, dtype=tf.float32), (3,4))
Y = tf.constant([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(f"\nOutput ofX>Y: \n{X>Y}")
print(f"\nOutput ofX<Y: \n{X<Y}")

# 2. Broadcasting
X = tf.reshape(tf.range(24, dtype=tf.float32), (2,3,4))
Y = tf.constant([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(f"\nOutput ofX>Y: \n{X>Y}")
print(f"\nOutput ofX<Y: \n{X<Y}")