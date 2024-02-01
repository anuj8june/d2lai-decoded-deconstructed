import tensorflow as tf


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