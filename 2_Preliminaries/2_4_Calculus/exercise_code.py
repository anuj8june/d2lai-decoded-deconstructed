import numpy as np

# 1. So far we took the rules for derivatives for granted. Using the definition and limits prove the properties for 
# (i) f(x)=c, (ii)fx=x^n , (iii)f(x)=e^x  and (iv) f(x)=log x.


def f_constant(x):
    return x


def f_sqaure(x):
    return x*x


def f_exponential(x):
    return np.exp(x)


def f_logarithmic(x):
    return np.log(x)


# Constant function
x=2
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_constant(x) - f_constant(x)) / h
    print(f"For constant function h: {h} f'x: {fx}")

# Square function
x=2
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_sqaure(x+h) - f_sqaure(x)) / h
    print(f"For square function h: {h} f'x: {fx}")

# Exponential function
x=2
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_exponential(x+h) - f_exponential(x)) / h
    print(f"For exponential function h: {h} f'x: {fx}")


# Logarithmic function
x=2
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_logarithmic(x+h) - f_logarithmic(x)) / h
    print(f"For logarithmic function h: {h} f'x: {fx}")


# 2. In the same vein, prove the product, sum, and quotient rule from first principles.

# fxgx = (x^2)(log x)
# Product rule

def f_product(x):
    return x*x*np.log(x)

x=2
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_product(x+h) - f_product(x)) / h
    print(f"For Product rule h: {h} f'x: {fx}")

product_derivative = 2*2*np.log(2) + 2
print(f"Using direct calculation for prodcut rule on fxgx we get : {product_derivative}")


# fxgx = (x^2)+(log x)
# Sum Rule

def f_sum(x):
    return x*x + np.log(x)

x=2
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_sum(x+h) - f_sum(x)) / h
    print(f"For Sum rule h: {h} f'x: {fx}")

sum_derivative = 2*2 + 1/2
print(f"Using direct calculation for sum rule on fxgx we get : {sum_derivative}")


# fxgx = (x^2)/(log x)
# Quotient Rule

def f_quotient(x):
    return (x*x)/np.log(x)

x=2
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_quotient(x+h) - f_quotient(x)) / h
    print(f"For Quotient rule h: {h} f'x: {fx}")

quotient_derivative = (np.log(2)*2*2 - 2) / (np.log(2)*np.log(2))
print(f"Using direct calculation for quotient rule on fxgx we get : {quotient_derivative}")


# 3. Prove that the constant multiple rule follows as a special case of the product rule.

C=5
x=2
# fxgx = C*log(x)
# constant Rule

def f_constant(x):
    return 5*np.log(x)

x=2
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_constant(x+h) - f_constant(x)) / h
    print(f"For constant rule h: {h} f'x: {fx}")

constant_derivative = 5/2
print(f"Using direct calculation for constant rule on fxgx we get : {constant_derivative}")


# 4. Calculate the derivative of f(x)=x^x.

x=2

def f_constant(x):
    return x**x

x=2
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_constant(x+h) - f_constant(x)) / h
    print(f"x^x h: {h} f'x: {fx}")

constant_derivative = (2**2)*(np.log(2)+1)
print(f"Using direct calculation for x^x we get : {constant_derivative}")

# 5. What does it mean that x=0 for some f(x)? Give an example of a function  and a location  for which this might hold. 
# Holds for x**2 at x=0
x=0
for i in range(-1,-11,-1):
    h=10**i
    fx = (f_sqaure(x+h) - f_sqaure(x)) / h
    print(f"For square function h: {h} f'x: {fx}")

# 6. Plotting Later
