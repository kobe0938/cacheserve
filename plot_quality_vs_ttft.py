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
        
        ax.plot(
            df_plot["total_ttft"],
            df_plot["avg_score"],
            marker="o",
            markersize=2,
            linewidth=1,
            label=label,
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
    ax.grid(True)
    
    # Add legend above the plot (like in droidspeak_example)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper left', fontsize=font_sz, frameon=True, 
                     bbox_to_anchor=(-0.15, 1.5), ncols=len(handles))

# ============================
# Create 2x3 subplot figure
# ============================
fig, axes = plt.subplots(2, 3, figsize=(7, 3.5))
wspace = 0.35
hspace = 0.6
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
        ax.grid(True)

# Save as PDF
output_path = "quality_vs_ttft.pdf"
plt.savefig(output_path, bbox_inches="tight")
print(f"[SAVE] Figure saved to {output_path}")
