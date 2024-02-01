import torch


# A simple function
x = torch.arange(4.0)
print(f"\nVector x: {x}")
x.requires_grad_(True)
x.grad
print(f"\nGradients of y wrt x before y backward called: {x.grad}")
y = 2*torch.dot(x,x)
print(f"\nY=2*X.T*X calculations: {y}")
y.sum().backward()
print(f"\nGradients of y wrt x after y backward called: {x.grad}")
print(f"\nGradient correct: {4*x==x.grad}")

x.grad.zero_()
y = x.sum()
y.backward()
print(f"\nGradient after reset of x and y: {x.grad}")

# Backward for non-scalar variables
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))
print(f"\nGradient after reset of x and y=x*x : {x.grad}")

# Detaching computation
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(f"\nGradient after y component detach from z: {x.grad} | Value of u: {u}")

x.grad.zero_()
y.sum().backward()
print(f"\nGradients of y with deatched graph: {x.grad} | Direct calculation: {2*x}")

# Gradients and python control flow
def f(a):
    b = a*2
    while b.norm()<1000:
        b = b*2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(f"\nGradient of d: {a.grad} | Manual calculation: {d/a}")