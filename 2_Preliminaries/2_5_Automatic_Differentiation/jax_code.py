import jax
from jax import grad
from jax import random
from jax import numpy as jnp


# A simple function
x = jnp.arange(4.0)
print(f"\nVector x: {x}")
y = lambda x: 2*jnp.dot(x, x)
print(f"\nY=2*X.T*X calculations: {y}")
x_grad = grad(y)(x)
print(f"\nGradients of y wrt x after y backward called: {x_grad}")
print(f"\nGradient correct: {4*x==x_grad}")

y = lambda x: x.sum()
print(f"\nGradient after reset of x and y: {grad(y)(x)}")

# Backward for non-scalar variables
y = lambda x: x * x
print(f"\nGradient after reset of x and y=x*x : {grad(lambda x: y(x).sum())(x)}")

# Detaching computation
y = lambda x: x * x
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x
print(f"\nGradient after y component detach from z: {grad(lambda x: z(x).sum())(x)} | Value of u: {y(x)}")

print(f"\nGradients of y with deatched graph: {grad(lambda x: y(x).sum())(x)} | Direct calculation: {2*x}")

# Gradients and python control flow
def f(a):
    b = a*2
    while jnp.linalg.norm(b)<1000:
        b = b*2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
print(f"\nGradient of d: {d_grad} | Manual calculation: {d/a}")