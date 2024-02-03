import torch
import random
from torch.distributions .multinomial import Multinomial


num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print(f"\nHeads: {heads} | Tails: {tails} | Num Tosses: {num_tosses}")

fair_probs = torch.tensor([0.5, 0.5])
print(f"\nMultinomial outcomes 100 tries:{Multinomial(100, fair_probs).sample()}")
print(f"\nMultinomial outcomes 100 tries:{Multinomial(100, fair_probs).sample()/100}")
print(f"\nMultinomial outcomes 10000 tries:{Multinomial(10000, fair_probs).sample()/10000}")