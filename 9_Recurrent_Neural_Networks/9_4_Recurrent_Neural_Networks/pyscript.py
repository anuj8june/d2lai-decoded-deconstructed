import numpy as np

X, W_xh = np.random.randn(3,1), np.random.randn(1,4)
H, W_hh = np.random.randn(3,4), np.random.randn(4,4)

# Two way of computing output
method1 = np.matmul(X, W_xh) + np.matmul(H, W_hh)
print(f"\nOutput of method1: {method1}")
method2 = np.matmul(np.concatenate([X, H], axis=1), np.concatenate([W_xh, W_hh], axis=0))
print(f"\nOutput of method2: {method2}")


### Exercise ###

# 1. If we use an RNN to predict the next character in a text sequence, what 
# is the required dimension for any output?
# It will equal to the size of character set. For smallcase english it will be 26.
# On top of this we will use softmax.

# 2. Why can RNNs express the conditional probability of a token at some time 
# step based on all the previous tokens in the text sequence?
# It is because if we unroll the rnn all the tokens contribute to the calculations
# of the next token in the sequence

# 3. What happens to the gradient if you backpropagate through a long sequence?
# It either vanishes or explodes if not kept in check. 

# 4. What are some of the problems associated with the language model described 
# in this section?
# a. It does not perfom well for long sequences.
# b. Vanishing and exploding gradients
# c. can't find correlation between words