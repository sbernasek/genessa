from crandom import generate_floats
import numpy as np

N = 109470
seed = 1006225625
samples = np.array(generate_floats(seed, N))
print('First Three', samples[0:3])
print('Last Three', samples[-3:])
print('Max', samples.max())
print('Min', samples.max())
