import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Define the mappings
def logistic_map(x):
    return 4 * x * (1 - x)

def hybrid_map(x, alpha):
    logistic_term = 4 * x * (1 - x)
    tent_term = 2 * min(x, 1 - x)
    return (1 - alpha) * logistic_term + alpha * tent_term

# Function to generate a sequence from a mapping
def generate_sequence(map_func, x0, n, alpha=None):
    sequence = []
    x = x0
    for _ in range(n):
        if alpha is None:
            x = map_func(x)
        else:
            x = map_func(x, alpha)
        sequence.append(x)
    return sequence

# Parameters
n_iterations = 10000  # For histogram and autocorrelation
x0 = 0.3  # Initial condition
alpha = 0.5  # Mixing parameter for hybrid map

# Generate sequences
logistic_sequence = generate_sequence(logistic_map, x0, n_iterations)
hybrid_sequence = generate_sequence(hybrid_map, x0, n_iterations, alpha=alpha)

# --- Test 1: Histogram of Generated Values ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(logistic_sequence, bins=50, color='blue', alpha=0.7)
plt.title("Logistic Map Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(hybrid_sequence, bins=50, color='red', alpha=0.7)
plt.title("Hybrid Map Histogram (alpha=0.5)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- Test 2: Autocorrelation ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(logistic_sequence, lags=50, ax=plt.gca())
plt.title("Logistic Map Autocorrelation")

plt.subplot(1, 2, 2)
plot_acf(hybrid_sequence, lags=50, ax=plt.gca())
plt.title("Hybrid Map Autocorrelation (alpha=0.5)")
plt.tight_layout()
plt.show()

# --- Test 3: Sensitivity to Initial Conditions ---
x0_1 = 0.3
x0_2 = 0.3000001  # Slightly different initial condition
n_short = 50  # Fewer iterations for clarity

logistic_seq1 = generate_sequence(logistic_map, x0_1, n_short)
logistic_seq2 = generate_sequence(logistic_map, x0_2, n_short)
hybrid_seq1 = generate_sequence(hybrid_map, x0_1, n_short, alpha=alpha)
hybrid_seq2 = generate_sequence(hybrid_map, x0_2, n_short, alpha=alpha)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(logistic_seq1, label="x0 = 0.3", color='blue')
plt.plot(logistic_seq2, label="x0 = 0.3000001", color='orange')
plt.title("Logistic Map Sensitivity")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hybrid_seq1, label="x0 = 0.3", color='red')
plt.plot(hybrid_seq2, label="x0 = 0.3000001", color='green')
plt.title("Hybrid Map Sensitivity (alpha=0.5)")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

