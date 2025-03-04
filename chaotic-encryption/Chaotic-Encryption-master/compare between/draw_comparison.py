import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Define the mappings
def logistic_map(x):
    return 4 * x * (1 - x)

def hybrid_map(x, alpha):
    logistic_term = 4 * x * (1 - x)
    tent_term = 2 * min(x, 1 - x)
    return (1 - alpha) * logistic_term + alpha * tent_term

# Generate a sequence
def generate_sequence(map_func, x0, n, alpha=None):
    sequence = [x0]
    x = x0
    for _ in range(n - 1):
        if alpha is None:
            x = map_func(x)
        else:
            x = map_func(x, alpha)
        sequence.append(x)
    return np.array(sequence)

# Function to calculate difference percentages
def calculate_difference_percentage(x0=0.3, n_histogram=10000, n_sensitivity=50, alpha=0.5, bins=50, lags=50, threshold=0.5):
    # Generate sequences
    logistic_seq = generate_sequence(logistic_map, x0, n_histogram)
    hybrid_seq = generate_sequence(hybrid_map, x0, n_histogram, alpha=alpha)

    # 1. Histogram Difference
    logistic_hist, _ = np.histogram(logistic_seq, bins=bins, range=(0, 1))
    hybrid_hist, _ = np.histogram(hybrid_seq, bins=bins, range=(0, 1))
    logistic_hist_std = np.std(logistic_hist)  # Variation in bin heights
    hybrid_hist_std = np.std(hybrid_hist)
    hist_diff_percent = abs(hybrid_hist_std - logistic_hist_std) / logistic_hist_std * 100

    # 2. Autocorrelation Difference
    logistic_acf = acf(logistic_seq, nlags=lags, fft=True)[1:]  # Exclude lag 0
    hybrid_acf = acf(hybrid_seq, nlags=lags, fft=True)[1:]
    logistic_acf_sum = np.sum(np.abs(logistic_acf))  # Sum of absolute coefficients
    hybrid_acf_sum = np.sum(np.abs(hybrid_acf))
    acf_diff_percent = abs(hybrid_acf_sum - logistic_acf_sum) / logistic_acf_sum * 100

    # 3. Sensitivity Difference
    x0_2 = x0 + 0.0000001  # Slightly different initial condition
    logistic_seq1 = generate_sequence(logistic_map, x0, n_sensitivity)
    logistic_seq2 = generate_sequence(logistic_map, x0_2, n_sensitivity)
    hybrid_seq1 = generate_sequence(hybrid_map, x0, n_sensitivity, alpha=alpha)
    hybrid_seq2 = generate_sequence(hybrid_map, x0_2, n_sensitivity, alpha=alpha)

    # Find iteration where difference exceeds threshold
    logistic_diff = np.abs(logistic_seq1 - logistic_seq2)
    hybrid_diff = np.abs(hybrid_seq1 - hybrid_seq2)
    logistic_div_iter = np.argmax(logistic_diff > threshold) if np.any(logistic_diff > threshold) else n_sensitivity
    hybrid_div_iter = np.argmax(hybrid_diff > threshold) if np.any(hybrid_diff > threshold) else n_sensitivity
    
    # Handle case where divergence doesn't occur within n_sensitivity
    if logistic_div_iter == 0: logistic_div_iter = n_sensitivity
    if hybrid_div_iter == 0: hybrid_div_iter = n_sensitivity
    sens_diff_percent = abs(hybrid_div_iter - logistic_div_iter) / logistic_div_iter * 100

    return {
        "Histogram Difference (%)": hist_diff_percent,
        "Autocorrelation Difference (%)": acf_diff_percent,
        "Sensitivity Difference (%)": sens_diff_percent,
        "Logistic Histogram STD": logistic_hist_std,
        "Hybrid Histogram STD": hybrid_hist_std,
        "Logistic ACF Sum": logistic_acf_sum,
        "Hybrid ACF Sum": hybrid_acf_sum,
        "Logistic Divergence Iteration": logistic_div_iter,
        "Hybrid Divergence Iteration": hybrid_div_iter
    }

# Run the function and print results
results = calculate_difference_percentage()
for key, value in results.items():
    print(f"{key}: {value:.2f}")

# Optional: Plot bar chart of differences
metrics = ["Histogram", "Autocorrelation", "Sensitivity"]
diff_values = [results["Histogram Difference (%)"], results["Autocorrelation Difference (%)"], results["Sensitivity Difference (%)"]]

plt.figure(figsize=(8, 5))
plt.bar(metrics, diff_values, color=['blue', 'green', 'red'])
plt.ylabel("Difference Percentage (%)")
plt.title("Difference Between Hybrid and Logistic Maps")
for i, v in enumerate(diff_values):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
plt.show()