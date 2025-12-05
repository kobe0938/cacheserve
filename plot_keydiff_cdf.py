import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
PLOT_DIR = '/Users/xiaokun/Desktop/cacheserve/cdf_plots/'
DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/scores_tokens.csv'
df = pd.read_csv(DATA_DIR)

# Configuration
METHOD = 'keydiff'
ANSWER_INDEX = 1
SLO = 0.9
compression_rates = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# datasets = df['dataset'].unique()
datasets = ['samsum']

# Create subplots
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, dataset_name in enumerate(datasets):
    dataset_df = df[df['dataset'] == dataset_name]
    total_contexts = len(dataset_df)
    print("total_contexts: ", total_contexts)
    
    # Process in reverse order (0.1 to 0.9) to accumulate failures correctly
    # Track which contexts have failed at higher compression rates
    failed_contexts = np.zeros(total_contexts, dtype=bool)
    cdf_values = []
    
    # Iterate from low to high compression (0.1 -> 0.9)
    for i, rate in enumerate(reversed(compression_rates)):
        print("rate: ", rate)
        # Get column for this compression rate with specified answer index
        rate_formatted = str(rate).replace('.', 'p')
        col_name = METHOD + '_' + rate_formatted + '_answer' + str(ANSWER_INDEX)
        
        # Check which contexts fail at this compression rate
        current_fails = (dataset_df[col_name].values < SLO)
        
        # Once failed, always failed at higher compression rates (cumulative)
        failed_contexts = failed_contexts | current_fails
        print("failed_contexts: ", failed_contexts)
        
        # Store count of contexts that PASS (not failed yet)
        passing_count = total_contexts - failed_contexts.sum()
        print("passing_count: ", passing_count)
        cdf_values.append(passing_count)
    
    # Reverse to match display order (0.9 to 0.1)
    cdf_values = list(reversed(cdf_values))
    
    # Plot CDF (count of passing contexts)
    x_positions = range(len(compression_rates))
    axes[idx].plot(x_positions, cdf_values, marker='o', linewidth=2)
    axes[idx].set_xticks(x_positions)
    axes[idx].set_xticklabels(compression_rates)
    axes[idx].set_xlabel('Compression Rate')
    axes[idx].set_ylabel('Count of Contexts Passing SLO')
    axes[idx].set_title(dataset_name)
    axes[idx].grid(True, alpha=0.3)

# Hide extra subplots
for idx in range(len(datasets), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
# output_filename = METHOD + '_cdf.png'
output_filename = METHOD + '_answer' + str(ANSWER_INDEX) + '_cdf.png'
plt.savefig(PLOT_DIR + output_filename, dpi=300, bbox_inches='tight')
print("Saved plot to " + output_filename)
