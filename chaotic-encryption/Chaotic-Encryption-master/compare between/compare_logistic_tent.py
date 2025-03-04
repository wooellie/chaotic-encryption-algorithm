import numpy as np
import matplotlib.pyplot as plt

# Define the maps
def logistic_map(x):
    return 4 * x * (1 - x)

def hybrid_map(x, alpha):
    logistic_term = 4 * x * (1 - x)
    tent_term = 2 * min(x, 1 - x)
    return (1 - alpha) * logistic_term + alpha * tent_term

# Parameters
n_iterations = 100
x0 = 0.3  # Initial condition
alphas = [0.0, 0.5, 1.0]  # Hybrid with alpha=0 (logistic), 0.5 (mixed), 1 (tent)

# Generate time series
def compute_series(map_func, x0, n, alpha=None):
    series = [x0]
    x = x0
    for _ in range(n - 1):
        if alpha is None:
            x = map_func(x)
        else:
            x = map_func(x, alpha)
        series.append(x)
    return series

# Compute series
logistic_series = compute_series(logistic_map, x0, n_iterations)
hybrid_series_05 = compute_series(hybrid_map, x0, n_iterations, alpha=0.5)
hybrid_series_1 = compute_series(hybrid_map, x0, n_iterations, alpha=1.0)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(logistic_series, label="Logistic Map (alpha = 0)", alpha=0.7)
plt.plot(hybrid_series_05, label="Hybrid Map (alpha = 0.5)", alpha=0.7)
plt.plot(hybrid_series_1, label="Hybrid Map (alpha = 1)", alpha=0.7)
plt.xlabel("Iteration (n)")
plt.ylabel("x_n")
plt.title("Time Series: Logistic vs. Hybrid Map")
plt.legend()
plt.grid(True)
plt.show()

#===============================画分岔图(只显示了红色的)=====================================
# Bifurcation diagram
n_iterations = 1000  # Total iterations
n_plot = 200  # Points to plot after transient
alpha_values = np.linspace(0, 1, 400)  # Vary alpha from 0 to 1

# Store points
logistic_points = []
hybrid_points = {alpha: [] for alpha in alpha_values}

# Compute logistic map (fixed)
x = x0
for i in range(n_iterations):
    x = logistic_map(x)
    if i >= (n_iterations - n_plot):
        logistic_points.append(x)

# Compute hybrid map for each alpha
for alpha in alpha_values:
    x = x0
    for i in range(n_iterations):
        x = hybrid_map(x, alpha)
        if i >= (n_iterations - n_plot):
            hybrid_points[alpha].append(x)

# Plot
plt.figure(figsize=(12, 6))

# Logistic map (constant line)
plt.scatter([0] * len(logistic_points), logistic_points, c='blue', s=1, alpha=0.5, label="Logistic Map (alpha = 0)")

# Hybrid map
for alpha in alpha_values:
    plt.scatter([alpha] * len(hybrid_points[alpha]), hybrid_points[alpha], c='red', s=1, alpha=0.5)

plt.xlabel("alpha (Hybrid Parameter)")
plt.ylabel("x_n (Long-term Values)")
plt.title("Bifurcation Diagram: Hybrid Map vs. Logistic")
plt.legend(["Logistic Map", "Hybrid Map"])
plt.grid(True)
plt.show()


#============================蛛网图========================================================
# Cobweb plot function
def cobweb_plot(map_func, x0, n, alpha=None, title="Cobweb Plot"):
    x = np.linspace(0, 1, 400)
    if alpha is None:
        y = [map_func(xi) for xi in x]
    else:
        y = [map_func(xi, alpha) for xi in x]
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, 'b-', label="Map Function")
    plt.plot(x, x, 'k--', label="y = x")
    
    # Iterate and plot
    x_n, y_n = x0, 0
    for _ in range(n):
        if alpha is None:
            y_n = map_func(x_n)
        else:
            y_n = map_func(x_n, alpha)
        plt.plot([x_n, x_n], [x_n, y_n], 'r-', alpha=0.5)
        plt.plot([x_n, y_n], [y_n, y_n], 'r-', alpha=0.5)
        x_n = y_n
    
    plt.xlabel("x_n")
    plt.ylabel("x_{n+1}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate cobweb plots
cobweb_plot(logistic_map, x0, 50, title="Cobweb: Logistic Map")
cobweb_plot(hybrid_map, x0, 50, alpha=0.5, title="Cobweb: Hybrid Map (alpha = 0.5)")








