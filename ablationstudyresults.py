import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Assuming 'results' is your dictionary
param_ranges = {
    'lam': np.array([1 * (10 ** -i) for i in range(10)]),
    'alpha': np.array([0.1 * i for i in range(1, 10)]),
    'beta': np.array([1.1 + 0.5 * i for i in range(10)]),
    'w': [np.ones((16,1)), np.zeros((16,1)), np.random.normal(0, .1, (16, 1)), 
          np.random.normal(0, .05, (16, 1)), np.random.normal(0, .2, (16, 1)),
          np.random.normal(0, .5, (16, 1)), np.random.normal(0, 1, (16, 1)),
          np.random.normal(0, 2, (16, 1)), np.random.normal(0, 3, (16, 1)),
          np.random.normal(5, .1, (16, 1))]  
}

# Finding the key with the minimum value
#min_key = min(resuts, key=resuts.get)
#min_value = resuts[min_key]

# Printing the results
#print(f"The key with the minimum value is '{min_key}' and its value is {min_value}.")

import matplotlib.pyplot as plt
import numpy as np

def plot_loss_vs_w_index(results, param_ranges):
    num_w_configs = len(param_ranges['w'])
    average_loss_per_w = []

    for w_index in range(num_w_configs):
        total_loss = 0
        count = 0

        for lam in param_ranges['lam']:
            for alpha in param_ranges['alpha']:
                for beta in param_ranges['beta']:
                    key = (lam, alpha, beta, w_index)
                    performance = results.get(key, np.nan)

                    # Check if performance is a sequence and take mean if it is
                    if isinstance(performance, (list, np.ndarray)):
                        performance = np.mean(performance)

                    if not np.isnan(performance):
                        total_loss += performance
                        count += 1

        # Calculate average loss for this w_index
        if count > 0:
            average_loss = total_loss / count
        else:
            average_loss = np.nan

        average_loss_per_w.append(average_loss)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_w_configs), average_loss_per_w, marker='o')
    plt.xlabel('Index of w')
    plt.ylabel('Average Loss')
    plt.title('Average Loss vs. w Index')
    plt.grid(True)
    plt.show()

# Call the function with your results and param_ranges
plot_loss_vs_w_index(resuts, param_ranges)


# Function to create a heatmap for a given w_index
def create_heatmap(w_index, results, param_ranges):
    performance_matrix = np.zeros((len(param_ranges['lam']), len(param_ranges['alpha'])))

    for i, lam in enumerate(param_ranges['lam']):
        for j, alpha in enumerate(param_ranges['alpha']):
            beta = param_ranges['beta'][0]  
            key = (lam, alpha, beta, w_index)
            performance = results.get(key, np.nan)

            # Check if performance is a sequence and take mean if it is
            if isinstance(performance, (list, np.ndarray)):
                performance = np.mean(performance)

            performance_matrix[i, j] = performance

    # Creating the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(performance_matrix, xticklabels=param_ranges['alpha'], yticklabels=param_ranges['lam'], annot=True, cmap='viridis')
    plt.xlabel('Alpha')
    plt.ylabel('Lambda')
    plt.title(f'Performance Heatmap for w Index {w_index}')
    plt.show()


# Create heatmaps for each w configuration
for w_index in range(len(param_ranges['w'])):
    create_heatmap(w_index, resuts, param_ranges)
