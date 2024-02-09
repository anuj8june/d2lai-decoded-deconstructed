# Differentiation and derivates
def f(x):
    return 3*x*x - 4*x

# Set x at 1
x = 1

for h in range(-1,-11,-1):
    h = 10**h
    dfx = (f(x+h)-f(x))/h
    print(f"h: {h} | f'(x): {dfx}")

