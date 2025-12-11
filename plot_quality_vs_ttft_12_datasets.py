import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ============================
# Styling (from droidspeak_example.ipynb)
# ============================
font_sz = 8
font = "Arial"

plt.rcParams['font.family'] = font
plt.rcParams['font.size'] = font_sz

# ============================
# Configuration
# ============================
RECORDS_DIR = "/Users/xiaokun/Desktop/cacheserve/records"

# Dataset names in desired order
DATASET_NAMES = [
    '2wikimqa',
    'gov_report',
    'hotpotqa',
    'multi_news',
    'multifieldqa_en',
    'musique',
    'narrativeqa',
    'qasper',
    'qmsum',
    'samsum',
    'trec',
    'triviaqa'
]

# ============================
# Function to plot single subplot for one dataset
# ============================
def plot_dataset_subplot(ax, dataset_name, csv_paths, xlabel=False, ylabel=False):
    """
    Plot quality vs TTFT for a single dataset across all CSV files
    Each CSV file contributes one point (average TTFT vs average score for that dataset)
    """
    ttfts = []
    scores = []
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        
        # Filter for this specific dataset
        df_dataset = df[df['dataset'] == dataset_name]
        
        if df_dataset.empty:
            print(f"[WARN] No data for dataset {dataset_name} in {os.path.basename(csv_path)}")
            continue
        
        # Calculate average TTFT and average score for this dataset in this CSV
        avg_ttft = df_dataset['ttft'].mean()
        avg_score = df_dataset['score'].mean()
        
        ttfts.append(avg_ttft)
        scores.append(avg_score)
        
        print(f"[LOAD] {os.path.basename(csv_path)} -> {dataset_name}: TTFT={avg_ttft:.4f}, Score={avg_score:.4f}")
    
    assert len(ttfts) == len(scores), f"Number of TTFTs and scores do not match for {dataset_name}"
    assert len(ttfts) == 7, f"Number of TTFTs and scores should be 7 for {dataset_name}"
    
    # Plot the line connecting the 7 points
    ax.plot(ttfts, scores, marker='o', markersize=2, linewidth=1, color='C0', label=dataset_name)
    
    # Formatting
    ax.set_title(dataset_name, fontsize=python)
    
    if xlabel:
        ax.set_xlabel("Average TTFT (s)", fontsize=font_sz, labelpad=1)
    if ylabel:
        ax.set_ylabel("Average score", fontsize=font_sz, labelpad=1)
    
    ax.set_xlim(left=0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis='both', which='major', pad=1)
    ax.grid(True, alpha=0.3, linewidth=0.5)

# ============================
# Load all CSV files
# ============================
csv_paths = sorted(glob.glob(os.path.join(RECORDS_DIR, "*.csv")))
if not csv_paths:
    print(f"[ERROR] No CSV files found in {RECORDS_DIR}")
    exit(1)

print(f"[INFO] Found {len(csv_paths)} CSV files")
for csv_path in csv_paths:
    print(f"  - {os.path.basename(csv_path)}")

# ============================
# Create 4x3 subplot figure
# ============================
fig, axes = plt.subplots(4, 3, figsize=(7, 6))
wspace = 0.35
hspace = 0.35
plt.subplots_adjust(wspace=wspace, hspace=hspace)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Plot each dataset in a separate subplot
for idx, dataset_name in enumerate(DATASET_NAMES):
    if idx >= 12:
        break
    
    ax = axes_flat[idx]
    
    # Determine if this subplot should have axis labels
    xlabel = idx >= 9  # Bottom row (indices 9, 10, 11)
    ylabel = idx % 3 == 0  # Left column (indices 0, 3, 6, 9)
    
    plot_dataset_subplot(ax, dataset_name, csv_paths, xlabel=xlabel, ylabel=ylabel)

# Save as PDF
output_path = "quality_vs_ttft_12_datasets.pdf"
plt.savefig(output_path, bbox_inches="tight")
print(f"[SAVE] Figure saved to {output_path}")
