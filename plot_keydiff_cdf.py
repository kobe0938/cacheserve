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
# Original rates for data processing
original_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# Shifted rates for display (each rate shifts down by 0.1)
compression_rates = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

# datasets = df['dataset'].unique()
datasets = ['samsum']

# Create subplots
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, dataset_name in enumerate(datasets):
    dataset_df = df[df['dataset'] == dataset_name]
    total_contexts = len(dataset_df)
    print("total_contexts: ", total_contexts)
    
    # Calculate PDF: incremental failures at each compression rate
    # Process from low to high compression (0.1 -> 0.9)
    failed_contexts = np.zeros(total_contexts, dtype=bool)
    pdf_values = [] 
    
    # Process actual compression rates (skip 1.0 from original_rates)
    actual_rates = [r for r in original_rates if r < 1.0]
    
    for rate in reversed(actual_rates):  # Process 0.1 -> 0.9
        rate_formatted = str(rate).replace('.', 'p')
        col_name = METHOD + '_' + rate_formatted + '_answer' + str(ANSWER_INDEX)
        
        # Check which contexts fail at this compression rate
        current_fails = (dataset_df[col_name].values < SLO)
        
        # Calculate NEW failures at this rate (not failed before)
        new_failures = current_fails & ~failed_contexts
        pdf_values.append(new_failures.sum())
        
        # Update cumulative failures for next iteration
        failed_contexts = failed_contexts | current_fails
    
    # Reverse to match display order (0.9 to 0.1)
    pdf_values = list(reversed(pdf_values))
    
    # Add count for 1.0: contexts that never fail at any compression rate
    # This will be displayed at 0.9 due to the shift
    never_failed = total_contexts - failed_contexts.sum()
    pdf_values = [never_failed] + pdf_values
    
    # CDF
    cdf_values = np.cumsum(pdf_values).tolist()
    
    # Plot CDF with shifted labels
    x_positions = range(len(compression_rates))
    axes[idx].plot(x_positions, cdf_values, marker='o', linewidth=2)
    axes[idx].set_xticks(x_positions)
    axes[idx].set_xticklabels(compression_rates)
    axes[idx].set_xlabel('Compression Rate')
    axes[idx].set_ylabel('Cumulative Count of Contexts Satisfying SLO')
    axes[idx].set_title(dataset_name)
    axes[idx].grid(True, alpha=0.3)

# Hide extra subplots
for idx in range(len(datasets), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
output_filename = METHOD + '_answer' + str(ANSWER_INDEX) + '_cdf.png'
plt.savefig(PLOT_DIR + output_filename, dpi=300, bbox_inches='tight')
print("Saved plot to " + output_filename)
