import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Font configuration
font_sz = 8
font = "Arial"
plt.rcParams['font.family'] = font
plt.rcParams['font.size'] = font_sz

# Load data
PLOT_DIR = '/Users/xiaokun/Desktop/cacheserve/cdf_plots/'
DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/scores_tokens_1and2.csv' # llama 8b
df = pd.read_csv(DATA_DIR)

# Configuration
METHOD = 'keydiff'
COMPRESSION_RATE = 0.9

# Specify datasets to process
datasets = ['samsum', 'triviaqa', 'multi_news', '2wikimqa', 'qasper', 'narrativeqa']
# '''
# [
#  '2wikimqa',
#  'gov_report',
#  'hotpotqa',
#  'multi_news',
#  'multifieldqa_en',
#  'musique',
#  'narrativeqa',
#  'qasper',
#  'qmsum',
#  'samsum',
#  'trec',
#  'triviaqa'
# ]
# '''
# datasets = ['samsum', 'triviaqa', 'multi_news', 'musique', 'qasper', 'narrativeqa', '2wikimqa', 'gov_report', 'hotpotqa', 'multifieldqa_en', 'qmsum', 'trec']

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(6.5, 2.5))
axes = axes.flatten()

for idx, dataset_name in enumerate(datasets):
    dataset_df = df[df['dataset'] == dataset_name]
    
    rate_str = str(COMPRESSION_RATE).replace('.', 'p')
    
    # Calculate average quality score across answer indices 1-100 for each entry
    avg_quality_scores = []
    
    for _, row in dataset_df.iterrows():
        # Collect quality scores for this entry across all answer indices
        quality_scores_per_entry = []
        for answer_idx in range(1, 101):
            col_name = f"{METHOD}_{rate_str}_answer{answer_idx}"
            
            if col_name not in df.columns:
                continue
            
            quality_score = row[col_name]
            quality_scores_per_entry.append(quality_score)
        
        assert len(quality_scores_per_entry) == 100, f"Expected 100 quality scores for {METHOD}_{rate_str}_answer{answer_idx}, got {len(quality_scores_per_entry)}"
        avg_quality_score = np.mean(quality_scores_per_entry)
        avg_quality_scores.append(avg_quality_score)
    
    # Convert to numpy array
    avg_quality_scores = np.array(avg_quality_scores)
    quality_drops = 1 - avg_quality_scores
    
    print(f"\n{dataset_name}:")
    print(f"  Number of entries: {len(quality_drops)}")
    print(f"  Mean quality drop: {np.mean(quality_drops):.3f}")
    print(f"  Median quality drop: {np.median(quality_drops):.3f}")
    # calculate CV
    cv = np.std(quality_drops) / np.mean(quality_drops)
    print(f"  CV: {cv:.3f}")
    
    # Sort quality scores for CDF
    sorted_scores = np.sort(quality_drops)
    
    # CDF
    cdf_values = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    
    # Plot CDF
    axes[idx].plot(sorted_scores, cdf_values, linewidth=2, color='steelblue')
    
    # Add labels only for leftmost column and bottom row
    if idx % 3 == 0:  # Leftmost column
        axes[idx].set_ylabel('CDF', fontsize=font_sz)
    if idx >= 3:  # Bottom row
        axes[idx].set_xlabel('Relative Quality Drop', fontsize=font_sz)
    
    # Add dataset name inside the plot
    axes[idx].text(0.05, 0.95, dataset_name, transform=axes[idx].transAxes,
                   fontsize=font_sz, fontweight='bold', verticalalignment='top')
    
    # # Calculate dynamic x-axis range based on data
    # if len(sorted_scores) > 0:
    #     min_val = np.min(sorted_scores)
    #     max_val = np.max(sorted_scores)
    #     range_width = max_val - min_val
        
    #     # Add padding factor (0.1 = 10% on each side)
    #     padding_factor = 0.1
    #     x_min = min_val - range_width * padding_factor
    #     x_max = max_val + range_width * padding_factor
        
    #     # Ensure x_min doesn't go below 0 (since quality drop is between 0 and 1)
    #     x_min = max(0, x_min)
    #     # Ensure x_max doesn't go above 1
    #     x_max = min(1, x_max)
        
    #     axes[idx].set_xlim([x_min, x_max])
    # else:
    #     axes[idx].set_xlim([0, 1])
    axes[idx].set_xlim([0, 1])
    
    axes[idx].set_ylim([0, 1.05])
    axes[idx].grid(True, alpha=0.3)

# Hide extra subplots
for idx in range(len(datasets), len(axes)):
    axes[idx].axis('off')

# plt.suptitle(f'Quality Score CDF - {METHOD} (Rate: {COMPRESSION_RATE}, Answers 1-100 avg)', 
#              fontsize=font_sz, y=0.995)
plt.tight_layout()

output_filename = f'{METHOD}_rate{rate_str}_answers1-100_quality_drop_cdf.pdf'
plt.savefig(PLOT_DIR + output_filename, bbox_inches='tight')
print(f"\nSaved plot to {output_filename}")

