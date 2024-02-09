import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# A simple function
x = tf.range(4.0)
print(f"\nVector x: {x}")
x = tf.Variable(x)
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
print(f"\nY=2*X.T*X calculations: {y}")
x_grad = t.gradient(y, x)
print(f"\nGradients of y wrt x after y backward called: {x_grad}")
print(f"\nGradient correct: {4*x==x_grad}")

with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
print(f"\nGradient after reset of x and y: {t.gradient(y,x)}")

# Backward for non-scalar variables
with tf.GradientTape() as t:
    y = x * x
print(f"\nGradient after reset of x and y=x*x : {t.gradient(y, x)}")

# Detaching computation
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x
print(f"\nGradient after y component detach from z: {t.gradient(z,x)} | Value of u: {u}")
print(f"\nGradients of y with deatched graph: {t.gradient(y,x)} | Direct calculation: {2*x}")

# Gradients and python control flow
def f(a):
    b = a*2
    while tf.norm(b)<1000:
        b = b*2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c

a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)

print(f"\nGradient of d: {t.gradient(d,a)} | Manual calculation: {d/a}")


### Exercise ###
# 1. Why is the second derivative much more expensive to compute than the first derivative?
# Because you have to calculate first derivative before you can proceed to calculate derivate. Its a 2 step process

# 2. After running the function for backpropagation, immediately run it again and see what happens. Investigate.
a = tf.range(3.0)
a = tf.Variable(x)
print(f"\nInput vector x: {x}")
with tf.GradientTape() as t:
    b = 3*tf.tensordot(a,a, axes=1)
try:
    # First calculation
    a_grad = t.gradient(b,a)
    print(f"\nFirst run | Input x: {a} | y(x): {b} | grad_x: {a_grad}")
    # Second calculation
    a_grad = t.gradient(b,a)
    print(f"\nSecond run | Input x: {a} | y(x): {b} | grad_x: {a_grad}")
except Exception as e:
    print(e)

# 3. In the control flow example where we calculate the derivative of d with respect to a, 
# what would happen if we changed the variable a to a random vector or a matrix? 
# At this point, the result of the calculation f(a) is no longer a scalar. What happens to the result? How do we analyze this?
def new_f(a):
    b = a*2
    while tf.norm(b)<1000:
        b = b*2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c

m = tf.range(10.0)
m = tf.Variable(m)
with tf.GradientTape() as t:
    vec_d = new_f(m)
grad_m = t.gradient(vec_d,m)
print(f"\nInput vector: {m} | Output vector: {vec_d} | Gradient: {grad_m} | Expected output: {vec_d/m}")

# 4. Let f(x)=sinx. Plot the graph of f and of its derivative f'. Do not exploit the fact that f'(x)=cosx 
# but rather use automatic differentiation to get the result.
x_in = np.linspace(-np.pi,np.pi,100)
x = tf.Variable(x_in)
with tf.GradientTape() as t:
    y = tf.sin(x)
y_out = np.sin(x)
plt.plot(x, y_out, label="sin(x)", color="red")
plt.plot(x, t.gradient(y,x), label="cos(x)", color="blue")
plt.legend(loc="best")
plt.savefig("4_graph.jpg")

# 5. Let f(x)=(logx^2 * sinx) + x^-1. Write out a dependency graph tracing results from x to f(x).

# 6. Use the chain rule to compute the derivative df/dx of the aforementioned function, placing each term on the dependency graph that you constructed previously.

# 7. Given the graph and the intermediate derivative results, you have a number of options when computing the gradient. 
# Evaluate the result once starting from to and once from tracing back to. The path from to is commonly known as forward differentiation, 
# whereas the path from to is known as backward differentiation.
x = tf.range(2.0,4.0,1)
x = tf.Variable(x)
with tf.GradientTape() as t:
    y = tf.math.log(x*x)*tf.math.sin(x) + (1/x)
y_out = tf.math.log(x*x)*tf.math.sin(x) + (1/x)
print(f"\nInput x: {x} | Forward y: {y_out} | Backward gradient: {t.gradient(y,x)}")