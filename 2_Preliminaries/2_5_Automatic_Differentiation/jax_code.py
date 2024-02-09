import jax
from jax import grad
from jax import random
from jax import numpy as jnp
import matplotlib.pyplot as plt


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


### Exercise ###
# 1. Why is the second derivative much more expensive to compute than the first derivative?
# Because you have to calculate first derivative before you can proceed to calculate derivate. Its a 2 step process

# 2. After running the function for backpropagation, immediately run it again and see what happens. Investigate.
a = jnp.arange(3.0)
print(f"\nInput vector x: {x}")
b = lambda a: 3*jnp.dot(a,a)
# First calculation
grad_a_1 = grad(b)(a)
print(f"\nFirst run | Input x: {a} | y(x): {b} | grad_x: {grad_a_1}")

# Second calculation
grad_a_2 = grad(b)(a)
print(f"\nSecond run | Input x: {a} | y(x): {b} | grad_x: {grad_a_2}")

# 3. In the control flow example where we calculate the derivative of d with respect to a, 
# what would happen if we changed the variable a to a random vector or a matrix? 
# At this point, the result of the calculation f(a) is no longer a scalar. What happens to the result? How do we analyze this?
def new_f(a):
    b = a*2
    while jnp.linalg.norm(b)<1000:
        b = b*2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

m = jnp.arange(10.0)
vec_d = new_f(m)
grad_vec_d = grad(lambda m: new_f(m).sum())(m)
print(f"\nInput vector: {m} | Output vector: {vec_d} | Gradient: {grad_vec_d} | Expected output: {vec_d/m}")

# 4. Let f(x)=sinx. Plot the graph of f and of its derivative f'. Do not exploit the fact that f'(x)=cosx 
# but rather use automatic differentiation to get the result.
x = jnp.linspace(-jnp.pi,jnp.pi,100)
y = lambda x: jnp.sin(x)
y_out = jnp.sin(x)
grad_x = grad(lambda x: y(x).sum())(x)
plt.plot(x, y_out, label="sin(x)", color="red")
plt.plot(x, grad_x, label="cos(x)", color="blue")
plt.legend(loc="best")
plt.savefig("4_graph.jpg")

# 5. Let f(x)=(logx^2 * sinx) + x^-1. Write out a dependency graph tracing results from x to f(x).

# 6. Use the chain rule to compute the derivative df/dx of the aforementioned function, placing each term on the dependency graph that you constructed previously.

# 7. Given the graph and the intermediate derivative results, you have a number of options when computing the gradient. 
# Evaluate the result once starting from to and once from tracing back to. The path from to is commonly known as forward differentiation, 
# whereas the path from to is known as backward differentiation.
x = jnp.arange(2.0,4.0,1)
y = lambda x: jnp.log(x*x)*jnp.sin(x) + (1/x)
y_out = jnp.log(x*x)*jnp.sin(x) + (1/x)
grad_x = grad(lambda x: y(x).sum())(x)
print(f"\nInput x: {x} | Forward y: {y_out} | Backward gradient: {grad_x}")