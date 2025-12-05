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
compression_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
datasets = df['dataset'].unique()
# datasets = ['samsum']

# Create subplots
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, dataset_name in enumerate(datasets):
    dataset_df = df[df['dataset'] == dataset_name]
    total_contexts = len(dataset_df)
    
    # Track which contexts have failed (dropped below SLO) at any point
    failed_contexts = np.zeros(total_contexts, dtype=bool)
    print("failed_contexts: ", failed_contexts)
    cdf_values = []
    
    for rate in compression_rates:
        # Get column for this compression rate with specified answer index
        rate_formatted = str(rate).replace('.', 'p')
        col_name = METHOD + '_' + rate_formatted + '_answer' + str(ANSWER_INDEX)
        print("dataset_name: ", dataset_name)
        print("col_name: ", col_name)
        print("value: ", dataset_df[col_name].values)
        
        # Check which contexts fail at this compression rate
        current_fails = (dataset_df[col_name].values < SLO)
        print("current_fails: ", current_fails)
        
        # Once failed, always failed (cumulative)
        failed_contexts = failed_contexts | current_fails
        print("failed_contexts: ", failed_contexts)
        
        cdf_values.append(failed_contexts.sum())
    
    # Plot CDF
    axes[idx].plot(compression_rates, cdf_values, marker='o', linewidth=2)
    axes[idx].set_xlabel('Compression Rate')
    axes[idx].set_ylabel('CDF: Fraction of Contexts Failed')
    axes[idx].set_title(dataset_name)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([0, 1.05])
    axes[idx].set_xlim([0.05, 0.95])

# Hide extra subplots
for idx in range(len(datasets), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
# output_filename = METHOD + '_cdf.png'
output_filename = METHOD + '_answer' + str(ANSWER_INDEX) + '_cdf.png'
plt.savefig(PLOT_DIR + output_filename, dpi=300, bbox_inches='tight')
print("Saved plot to " + output_filename)
