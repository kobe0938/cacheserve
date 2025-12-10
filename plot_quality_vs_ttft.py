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
# Legend customization
# ============================
# Mapping for legend labels
LEGEND_LABEL_MAP = {
    'keydiff': 'keydiff+LRU-based eviction',
    'knorm': 'knorm+LRU-based eviction',
    'snapkv': 'snapkv+LRU-based eviction',
    'offload': 'eviction only',
}

# Order and markers for legends (7 legends total)
LEGEND_ORDER = ['keydiff', 'knorm', 'snapkv', 'impress', 'offload', 'prefill', 'ours']
LEGEND_MARKERS = {
    'keydiff': 'v',      # triangle down
    'knorm': '^',        # triangle up
    'snapkv': 's',       # square
    'impress': 'o',      # circle
    'offload': 'D',      # diamond
    'prefill': 'p',      # pentagon
    'ours': '*',         # star
}

# ============================
# Configuration: Fill in your directories
# ============================
# Each directory should contain CSV files with total_ttft and avg_score columns
DIRS = [
    "intermediate_results/Llama-3p1-8B-Instruct/results",
    "intermediate_results/Mistral-7B-Instruct-v0p3/results", 
    "intermediate_results/Qwen2p5-14B-Instruct/results",
    # Placeholders for future directories (will use same data as first three)
    "intermediate_results/Llama-3p1-8B-Instruct/results",
    "intermediate_results/Mistral-7B-Instruct-v0p3/results", 
    "intermediate_results/Qwen2p5-14B-Instruct/results",
]

# ============================
# Function to plot single subplot (original logic)
# ============================
def plot_quality_vs_ttft_subplot(ax, directory, xlabel=False, ylabel=False, show_legend=False):
    """
    Plot quality vs TTFT for a single directory/subplot
    Uses original plotting logic - one line per CSV file
    """
    csv_paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csv_paths:
        print(f"[WARN] No CSV found under {directory}")
        ax.text(0.5, 0.5, "No CSV files", ha='center', va='center', 
                transform=ax.transAxes, fontsize=font_sz)
        return
    
    series_list = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if not {"total_ttft", "avg_score"}.issubset(df.columns):
            print(f"[SKIP] {csv_path}: missing total_ttft / avg_score")
            continue
        
        # label: file name (without extension)
        file_stem = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"[LOAD] {csv_path}  -> label = {file_stem}")
        series_list.append((df, file_stem))
    
    if not series_list:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', 
                transform=ax.transAxes, fontsize=font_sz)
        return
    
    # Plot each CSV as a separate line (original logic)
    for df, label in series_list:
        # Drop NaN and sort by total_ttft
        df_plot = (
            df[["total_ttft", "avg_score"]]
            .dropna(subset=["total_ttft", "avg_score"])
            .sort_values("total_ttft")
        )
        
        if df_plot.empty:
            print(f"[SKIP] {label}: empty after dropping NaN")
            continue
        
        # Get custom legend label
        display_label = LEGEND_LABEL_MAP.get(label, label)
        
        # Get marker shape
        marker = LEGEND_MARKERS.get(label, 'o')
        
        # if marker shape is pentagon,marker size is 4
        ax.plot(
            df_plot["total_ttft"],
            df_plot["avg_score"],
            marker=marker,
            markersize=3.3 if marker == 'p' else 3 if marker == '*' else 2,
            linewidth=1,
            label=display_label,
        )
    
    # Formatting
    # Get model name (parent directory name)
    parent_dir = os.path.dirname(os.path.normpath(directory))
    model_name = os.path.basename(parent_dir)
    ax.set_title(model_name, fontsize=font_sz)
    
    if xlabel:
        ax.set_xlabel("Total TTFT (s)", fontsize=font_sz, labelpad=1)
    if ylabel:
        ax.set_ylabel("Average score", fontsize=font_sz, labelpad=1)
    
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis='both', which='major', pad=1)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add legend above the plot (like in droidspeak_example)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Reorder handles and labels according to LEGEND_ORDER
            # Create a mapping from display label to (handle, original_key)
            label_to_handle = {}
            for h, lbl in zip(handles, labels):
                # Find the original key for this display label
                original_key = lbl
                for key, display_lbl in LEGEND_LABEL_MAP.items():
                    if display_lbl == lbl:
                        original_key = key
                        break
                label_to_handle[original_key] = (h, lbl)
            

            # Reorder according to LEGEND_ORDER
            ordered_handles = []
            ordered_labels = []
            for key in LEGEND_ORDER:
                if key in label_to_handle:
                    h, lbl = label_to_handle[key]
                    ordered_handles.append(h)
                    ordered_labels.append(lbl)
            
            # Matplotlib legend uses column-major order with ncols parameter
            # We need to rearrange for row-major display
            # With 7 items and ncols=3: we want 3 rows (3, 3, 1 items per row)
            # But matplotlib fills columns first, so we need to transpose
            ncols = 3
            nrows = (len(ordered_handles) + ncols - 1) // ncols  # ceiling division
            
            # Rearrange from row-major to column-major
            reordered_handles = []
            reordered_labels = []
            for col in range(ncols):
                for row in range(nrows):
                    idx = row * ncols + col
                    if idx < len(ordered_handles):
                        reordered_handles.append(ordered_handles[idx])
                        reordered_labels.append(ordered_labels[idx])
            
            ax.legend(reordered_handles, reordered_labels, loc='upper left', fontsize=font_sz, frameon=True, 
                     bbox_to_anchor=(-0.15, 1.8), ncol=ncols)

# ============================
# Create 2x3 subplot figure
# ============================
fig, axes = plt.subplots(2, 3, figsize=(7, 3.5))
wspace = 0.35
hspace = 0.35
plt.subplots_adjust(wspace=wspace, hspace=hspace)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Plot each directory in a separate subplot
for idx, directory in enumerate(DIRS[:6]):  # Only use first 6
    if idx >= 6:
        break
    
    ax = axes_flat[idx]
    
    # Determine if this subplot should have axis labels
    xlabel = idx >= 3  # Bottom row (indices 3, 4, 5)
    ylabel = idx % 3 == 0  # Left column (indices 0, 3)
    show_legend = idx == 0  # Only show legend on first subplot
    
    if os.path.exists(directory):
        plot_quality_vs_ttft_subplot(ax, directory, xlabel=xlabel, ylabel=ylabel, show_legend=show_legend)
    else:
        print(f"[WARN] Directory not found: {directory}")
        parent_dir = os.path.dirname(os.path.normpath(directory))
        model_name = os.path.basename(parent_dir)
        ax.text(0.5, 0.5, f"{model_name}\n(not found)", ha='center', va='center', 
                transform=ax.transAxes, fontsize=font_sz)
        ax.set_title(model_name, fontsize=font_sz)
        if xlabel:
            ax.set_xlabel("Total TTFT (s)", fontsize=font_sz, labelpad=1)
        if ylabel:
            ax.set_ylabel("Average score", fontsize=font_sz, labelpad=1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(True, alpha=0.3, linewidth=0.5)

# Save as PDF
output_path = "quality_vs_ttft.pdf"
plt.savefig(output_path, bbox_inches="tight")
print(f"[SAVE] Figure saved to {output_path}")
