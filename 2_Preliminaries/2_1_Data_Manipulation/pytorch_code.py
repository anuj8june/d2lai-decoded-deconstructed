import torch

# Creating a one dimensional tensor
x = torch.arange(12, dtype=torch.float32)
print(f"\nOutput of creating a one dimensional tensor x: {x}")

# Check size of tensor
print(f"\nSize of created tensor x: {x.numel()}")
print(f"\nSize of created tensor x: {x.shape}")

# Reshape the vector to a matrix
X = x.reshape(3,4)
print(f"\nOutput after reshape X: \n{X}")
print(f"\nSize of created matrix X: {X.numel()}")
print(f"\nSize of created matrix X: {X.shape}")

# Creating tensor with all zeros
X = torch.zeros((2,3,4))
print(f"\nTensor with all zeros X: {X}")

# Creating tensor with all ones
X = torch.ones((2,3,4))
print(f"\nTensor with all ones X: {X}")

# Tensor with values from normal distribution
X = torch.randn(3,4)
print(f"\nTensor initialized with values from normal distribution X: \n{X}")

# Tensor with all constants.
X = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(f"\nTensor initialized manually X: \n{X}")

# Indexing and slicing
print(f"\nOutput of X[1]: \n{X[-1]} \nOutput of X[1:3]: \n{X[1:3]}")

# Writing elements to array
X[1,2] = 17
print(f"\nAfter changing values of elements in X : \n{X}")

# Assigning multiple values at the same time
X[:2,:] = 12
print(f"\nAfter changing values of elements in X : \n{X}")


# Exponential operator
print(f"\nOuput of e^x : \n{torch.exp(x)}")

# All elementwise operation
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(f"\nOuput of operation of x & y \nAddition: {x+y} \nSubtrction: {x-y} \nMultiplication: {x*y} \nDivision: {x/y} \nPower Function: {x**y}")

# Concatenation operation
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(f"\nOutput of concatenation along 0 axis: \n{torch.cat((X,Y), dim=0)}")
print(f"\nOutput of concatenation along 1 axis: \n{torch.cat((X,Y), dim=1)}")

# Check if the values are equal
print(f"\nEqual elements in X & Y: \n{X==Y}")

# Sum of all elements in matrix
print(f"\nSum of all elements in matrix: {torch.sum(X)}")


# Broadcasting
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
print(f"\nInitalized arrays with different shapes a: \n{a} and b: {b}")
print(f"\nAddition after broadcasting a+b: \n{a+b}")

# Memory Allocations
before = id(Y)
Y = Y + X
print(f"\nIs the memory location same as before : {id(Y) == before}")

# Inplace update of operation
Z = torch.zeros_like(Y)
print(f"\nAddress of Z before operation: {id(Z)}")
Z[:] = X + Y   # This [:] notation updates the values in the same memory address
print(f"\nAddress of Z after operation: {id(Z)}")

# Conversion to numpy object and back
A = X.numpy()
B = torch.from_numpy(A)
print(f"\nType A : {type(A)} | Type B: {type(B)}")

# Type conversions
a = torch.tensor([3.5])
print(f"\nOutput of a: {a} | a.item(): {a.item()} | float(a): {float(a)} | int(a): {int(a)}")


# Exercise
# 1. Replace X==Y with X>Y and X<Y
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(f"\nOutput ofX>Y: \n{X>Y}")
print(f"\nOutput ofX<Y: \n{X<Y}")

# 2. Broadcasting
X = torch.arange(24, dtype=torch.float32).reshape((2,3,4))
Y = torch.tensor([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(f"\nOutput ofX>Y: \n{X>Y}")
print(f"\nOutput ofX<Y: \n{X<Y}")