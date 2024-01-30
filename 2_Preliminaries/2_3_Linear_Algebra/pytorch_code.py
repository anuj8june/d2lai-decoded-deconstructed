import time
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

### Exercise
# 1. Prove that the transpose of the transpose of a matrix is the matrix itself: 
A = torch.arange(6).reshape(2,3)
print(f"\nA: {A} | If A.T.T==A: {A==A.T.T}")

# 2.Given two matrices and , show that sum and transposition commute
A = torch.arange(6).reshape(2,3)
B = torch.ones(2,3)
sum_of_individual_tranpose = A.T + B.T
sum_first_then_transpose = (A+B).T
print(f"\nA: {A} | B: {B} | Is sum of individual tranpose equal to sum first and then transpose: {sum_of_individual_tranpose==sum_first_then_transpose}")

# 3. Given any square matrix A. Is A + A.T always symmetric?
A = torch.arange(9).reshape(3,3)
print(f"\nA: {A} | Is A + A.T symmetric?: {A + A.T}")

# 4. We defined the tensor X of shape (2, 3, 4) in this section. What is the output of len(X)?
A = torch.ones(2,3,3)
print(f"\nLength of A:{A} is {len(A)}")

# 5. or a tensor X of arbitrary shape, does len(X) always correspond to the length of a certain axis of X? What is that axis?
# 1st axis

# 6. Run A / A.sum(axis=1) and see what happens. Can you analyze the results?
A = torch.arange(6).reshape(2,3)
print(f"\nA/A.sum(axis=1) = {A/A.sum(axis=1, keepdims=True)}")

# 7. When traveling between two points in downtown Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
# Find coordinates and use distance formula(l2 norm). Or use google maps for more precision.

# 8. Consider a tensor of shape (2, 3, 4). What are the shapes of the summation outputs along axes 0, 1, and 2?
A = torch.arange(24).reshape(2,3,4)
sum_axis0 = A.sum(axis=0).shape
sum_axis1 = A.sum(axis=1).shape
sum_axis2 = A.sum(axis=2).shape
print(f"\nSum of A(2,3,4) along axis 0: {sum_axis0} along axis 1: {sum_axis1} along axis 2: {sum_axis2}")

# 9. Feed a tensor with three or more axes to the linalg.norm function and observe its output. What does this function compute for tensors of arbitrary shape?
A = torch.arange(24).reshape(2,3,4).type(torch.float32)
print(f"\nlinalg.norm output on A: {A} is :{torch.norm(A)}")

# 10. You want to compute the product ABC. Is there any difference in memory footprint and speed, depending on whether you compute (AB)C or A(BC). Why?
A = torch.randn(10,16)
B = torch.randn(16,5)
C = torch.randn(5,14)
start = time.process_time()
AB_C = torch.matmul(torch.matmul(A,B),C)
print(f"Time taken to execute (AB)C: {time.process_time()-start}")
start = time.process_time()
A_BC = torch.matmul(A,torch.matmul(B,C))
print(f"Time taken to execute A(BC): {time.process_time()-start}")

# 11. Is there any difference in speed depending on whether you compute AB or AC.T? Why? What changes if you initialize C=B.T without cloning memory? Why?
A = torch.randn(10,16)
B = torch.randn(16,5)
C = torch.randn(5,16)
start = time.process_time()
AB = torch.matmul(A,B)
print(f"Time taken to execute AB: {time.process_time()-start}")
start = time.process_time()
ACT = torch.matmul(A,C.T)
print(f"Time taken to execute AC.T: {time.process_time()-start}")

C = B.T
start = time.process_time()
AB = torch.matmul(A,B)
print(f"Time taken to execute AB: {time.process_time()-start}")
start = time.process_time()
ACT = torch.matmul(A,C.T)
print(f"Time taken to execute AC.T when C=B.T: {time.process_time()-start}")

# 12. Construct a tensor with three axes by stacking. What is the dimensionality? Slice out the second coordinate of the third axis to recover. Check that your answer is correct.
A = torch.arange(20000).reshape(100,200)
B = torch.arange(20000, 40000).reshape(100,200)
C = torch.arange(40000, 60000).reshape(100,200)
ABC = torch.stack((A,B,C), dim=2)
print(f"After stacking ABC shape: {ABC.shape}. Extracting B out using slicing: {ABC[:,:,1]}")