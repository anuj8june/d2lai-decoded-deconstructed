import jax
import random
import numpy as np
from jax import numpy as jnp


num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print(f"\nHeads: {heads} | Tails: {tails} | Num Tosses: {num_tosses}")

fair_probs = [0.5, 0.5]
print(f"\nMultinomial outcomes 100 tries:{np.random.multinomial(100, fair_probs)}")
print(f"\nMultinomial outcomes 100 tries:{np.random.multinomial(100, fair_probs)/100}")
print(f"\nMultinomial outcomes 10000 tries:{np.random.multinomial(10000, fair_probs)/10000}")