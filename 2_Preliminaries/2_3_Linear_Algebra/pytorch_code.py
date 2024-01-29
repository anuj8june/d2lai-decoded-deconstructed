import torch

# Scalars
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(f"\nScalar operation on x and y | x+y: {x+y} | x*y: {x*y} | x/y: {x/y} | x**y: {x**y}")

# Vectors
x = torch.arange(3)
print(f"\nVector x: {x} | Value at index 2: {x[2]} | Length: {len(x)} | Shape: {x.shape}")

# Matrices
X = torch.arange(6).reshape(3,2)
print(f"\nMatrix X:\n{X} \nTranspose on X: \n{X.T}")

A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(f"\nMatrix A: \n{A} | \nIs A symmetric?: \n{A==A.T}")

# Tensors
A = torch.arange(24).reshape(2,3,4)
print(f"\nTensor: \n{A}")

# Basic properties of tensor arithmetic
A = torch.arange(6).reshape(2,3)
B = A.clone()
print(f"\nA: \n{A} | \nA+B: \n{A+B}")

# Hadamard Product
print(f"\nHadamard product of A & B: \n{A*B}")

# Reduction sum
x = torch.arange(3, dtype=torch.float32)
print(f"\nSum of X: {x} is {x.sum()}")

A = torch.arange(6).reshape(2,3)
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {A.sum()}")
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {A.sum(axis=0).shape}")
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {A.sum(axis=1).shape}")
print(f"\nA: \n{A} | \nShape: {A.shape} | \nSum: {A.sum(axis=[0,1])} | Both sums equal: {A.sum(axis=[0,1])==A.sum()}")
print(f"\nA: {A} | Mean: {A.sum()/A.numel()} | Mean: {A.float().mean()}")
print(f"\nA: {A} | Mean axis=0: {A.sum(axis=0)/A.shape[0]} | Mean axis=0: {A.float().mean(axis=0)}")

# Non reduction sum
sum_A = A.sum(axis=1, keepdims=True)
print(f"\nA: \n{A} | \nsum A: {sum_A} | \nshape A: {sum_A.shape} | \nNormalisation along axis: {A/sum_A} | \nCumsum: {A.cumsum(axis=0)}")

# Dot Product
x = torch.arange(3, dtype=torch.float32)
y = torch.ones(3, dtype=torch.float32)
print(f"\nDot product of x {X} & y: {y} is {torch.dot(x,y)} alternate way: {torch.sum(x*y)}")

# Matrix vector multiplication
print(f"\nMatrix vector multiplication | A: {A} | x: {x} | A shape: {A.shape} | x shape: {x.shape} | mv : {torch.mv(A.float(),x.float())} | alternate way mv: {A.float()@x.float()}")

# Matrix matrix multiplication
A = A.type(torch.float32)
B = torch.ones((3,4), dtype=torch.float32)
print(f"\nMatrix multiplication of A: {A} & B: {B} is {torch.matmul(A,B)}")

# Norms
u = torch.tensor([3.0, -4.0])
print(f"\nl2 norm of u: {u} is {torch.norm(u)}")
print(f"\nl1 norm of u: {u} is {torch.abs(u).sum()}")
A = torch.ones((4,9)).type(torch.float32)
print(f"\nFrobenius norm of matrix A:{A} is {torch.norm(A)}")