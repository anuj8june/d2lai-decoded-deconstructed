import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd


num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print(f"\nHeads: {heads} | Tails: {tails} | Num Tosses: {num_tosses}")

fair_probs = tf.ones(2)/2
print(f"\nMultinomial outcomes 100 tries:{tfd.Multinomial(100, fair_probs).sample()}")
print(f"\nMultinomial outcomes 100 tries:{tfd.Multinomial(100, fair_probs).sample()/100}")
print(f"\nMultinomial outcomes 10000 tries:{tfd.Multinomial(10000, fair_probs).sample()/10000}")