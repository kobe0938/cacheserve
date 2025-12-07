import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
PLOT_DIR = '/Users/xiaokun/Desktop/cacheserve/cdf_plots/'
DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/scores_tokens_1and2.csv' # llama 8b
df = pd.read_csv(DATA_DIR)

# Configuration
METHOD = 'keydiff'
ANSWER_INDEX = 1
COMPRESSION_RATE = 0.8

# Get all datasets
datasets = df['dataset'].unique()

# Create subplots
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, dataset_name in enumerate(datasets):
    dataset_df = df[df['dataset'] == dataset_name]
    
    # Get the column name for this method/rate/answer combination
    rate_str = str(COMPRESSION_RATE).replace('.', 'p')
    col_name = f"{METHOD}_{rate_str}_answer{ANSWER_INDEX}"
    
    # Get all quality scores for this dataset
    quality_scores = dataset_df[col_name].values
    # print(f"# of quality scores: {len(quality_scores)}")
    print("quality_scores: ", quality_scores)
    # print mean and median of quality scores
    print(f"mean: {np.mean(quality_scores)}")
    print(f"median: {np.median(quality_scores)}")
    
    # Sort quality scores for CDF
    sorted_scores = np.sort(quality_scores)
    
    # CDF
    cdf_values = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    # print("cdf_values: ", cdf_values)
    
    # Plot CDF
    axes[idx].plot(sorted_scores, cdf_values, linewidth=2, color='steelblue')
    axes[idx].set_xlabel('Quality Score')
    axes[idx].set_ylabel('CDF')
    axes[idx].set_title(dataset_name)
    axes[idx].set_xlim([0, 1])
    axes[idx].set_ylim([0, 1.05])
    axes[idx].grid(True, alpha=0.3)

# Hide extra subplots
for idx in range(len(datasets), len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Quality Score CDF - {METHOD} (Rate: {COMPRESSION_RATE}, Answer: {ANSWER_INDEX})', 
             fontsize=16, y=0.995)
plt.tight_layout()

output_filename = f'{METHOD}_rate{rate_str}_answer{ANSWER_INDEX}_quality_cdf.png'
plt.savefig(PLOT_DIR + output_filename, dpi=300, bbox_inches='tight')
print(f"Saved plot to {output_filename}")

