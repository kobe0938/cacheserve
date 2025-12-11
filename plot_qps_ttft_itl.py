import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ============================
# Styling (from plot_quality_vs_ttft.py)
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
# Configuration: Model directories
# ============================
MODEL_DIRS = [
    "cache-serve/Llama-3p1-8B-Instruct",
    "cache-serve/Mistral-7B-Instruct-v0p3",
    "cache-serve/Qwen2p5-14B-Instruct",
]

# ============================
# Helper functions (from visualize.py)
# ============================
def parse_method_rate_score(dirname: str):
    """
    Given a directory name like 'ours_0p7_0p86337',
    return ('ours', '0p7', '0p86337').
    """
    name = os.path.basename(dirname.rstrip("/"))
    method, rate, score = name.rsplit("_", 2)
    return method, rate, score


def load_qps_stats(run_dir: str):
    """
    Given directory:
        <directory_name>/<method>_<rate>_<score>/
    Load all <qps>.csv files and compute mean ttft and mean itl per QPS.

    Returns:
        DataFrame with columns:
            qps, mean_ttft, mean_itl
    """
    csvs = glob.glob(os.path.join(run_dir, "*.csv"))
    rows = []

    for path in csvs:
        base = os.path.basename(path)
        qps_str = base.replace(".csv", "")
        try:
            qps = float(qps_str)
        except ValueError:
            continue

        df = pd.read_csv(path)

        if "ttft" not in df.columns:
            print(f"[WARN] Missing 'ttft' in {path}")
            continue
        if "itl" not in df.columns:
            print(f"[WARN] Missing 'itl' in {path}")
            continue

        rows.append({
            "qps": qps,
            "mean_ttft": df["ttft"].mean(),
            "mean_itl": df["itl"].mean(),
        })

    if not rows:
        return None

    out = pd.DataFrame(rows)
    out = out.sort_values("qps")
    out = out.reset_index(drop=True)
    return out


def apply_ttft_cutoff(df: pd.DataFrame, cutoff: float = 10.0) -> pd.DataFrame:
    """
    For a subdf belonging to a single method/score:
    - Keep all rows where mean_ttft <= cutoff.
    - Find the first row where mean_ttft > cutoff.
    - Keep that one row, drop all later rows.
    """
    df = df.sort_values("qps").reset_index(drop=True)

    below = df[df["mean_ttft"] <= cutoff]
    above = df[df["mean_ttft"] > cutoff]

    if above.empty:
        return df  # No clipping needed.

    first_above = above.iloc[0:1]  # keep only first > cutoff row

    # Only rows below cutoff, plus first_above
    clipped = pd.concat([below, first_above], ignore_index=True)
    return clipped


def score_str_to_float_str(score_str: str) -> str:
    score_float = float(score_str.replace("p", "."))
    # truncate to 4 decimal places
    return f"{score_float:.4f}"


def add_manual_data_points(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Add manually specified data points for specific models.
    """
    # Manual data points for Llama-3p1-8B-Instruct "ours" method
    if model_name == "Llama-3p1-8B-Instruct":
        manual_points = [
            {"qps": 11, "mean_ttft": 0.1565, "mean_itl": 0.0411},
            {"qps": 12, "mean_ttft": 0.1984, "mean_itl": 0.0607},
            {"qps": 13, "mean_ttft": 0.3325, "mean_itl": 0.1141},
            {"qps": 14, "mean_ttft": 2.5497, "mean_itl": 0.2652},
            {"qps": 15, "mean_ttft": 11.8935, "mean_itl": 0.2778},
        ]
        
        # Find the "ours" method in the dataframe
        ours_mask = df['method'] == 'ours'
        if ours_mask.any():
            # Get the rate and score from existing "ours" data
            ours_data = df[ours_mask].iloc[0]
            rate = ours_data['rate']
            score = ours_data['score']
            
            # Remove existing data points for these QPS values
            manual_qps = [p['qps'] for p in manual_points]
            df = df[~((df['method'] == 'ours') & (df['qps'].isin(manual_qps)))]
            
            # Add manual data points
            for point in manual_points:
                new_row = pd.DataFrame([{
                    'qps': point['qps'],
                    'mean_ttft': point['mean_ttft'],
                    'mean_itl': point['mean_itl'],
                    'method': 'ours',
                    'rate': rate,
                    'score': score,
                }])
                df = pd.concat([df, new_row], ignore_index=True)
    
    return df.sort_values(['method', 'score', 'qps']).reset_index(drop=True)


def load_model_data(directory_name: str):
    """
    Load all data for a given model directory.
    Returns combined DataFrame with columns: qps, mean_ttft, mean_itl, method, rate, score
    """
    base = directory_name
    model_name = os.path.basename(directory_name)

    # Find all <method>_<rate>_<score> directories
    run_dirs = [
        x for x in glob.glob(os.path.join(base, "outputs", "*"))
        if os.path.isdir(x)
    ]

    all_stats = []

    for run_dir in run_dirs:
        method, rate, score = parse_method_rate_score(run_dir)

        df_stats = load_qps_stats(run_dir)
        if df_stats is None:
            continue

        df_stats["method"] = method
        df_stats["rate"] = rate
        df_stats["score"] = score
        all_stats.append(df_stats)

    if not all_stats:
        return None

    combined = pd.concat(all_stats, ignore_index=True)

    # Add manual data points for specific models
    combined = add_manual_data_points(combined, model_name)

    # Apply TTFT clipping per (method, score)
    clipped_groups = []
    for (method, score), subdf in combined.groupby(["method", "score"]):
        clipped = apply_ttft_cutoff(subdf, cutoff=10.0)
        clipped_groups.append(clipped)

    combined_clipped = pd.concat(clipped_groups, ignore_index=True)
    return combined_clipped


# ============================
# Plotting functions
# ============================
def plot_qps_ttft_subplot(ax, directory_name, xlabel=False, ylabel=False, show_legend=False):
    """
    Plot QPS vs TTFT for a single model
    """
    data = load_model_data(directory_name)
    
    # Get clean model name (remove "cache-serve/" prefix if present)
    model_name = os.path.basename(directory_name)
    
    if data is None:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center',
                transform=ax.transAxes, fontsize=font_sz)
        ax.set_title(model_name, fontsize=font_sz)
        return
    
    # Plot each method/score combination
    for idx, ((method, score), subdf) in enumerate(data.groupby(["method", "score"])):
        jitter = (idx * 0.02)
        x = subdf["qps"] + jitter
        y = subdf["mean_ttft"]
        
        # Get custom legend label
        display_label = LEGEND_LABEL_MAP.get(method, method)
        
        # Get marker shape
        marker = LEGEND_MARKERS.get(method, 'o')
        
        ax.plot(
            x, y,
            marker=marker,
            markersize=3 if marker == '*' else 2,
            linewidth=1,
            label=display_label,
        )
    
    # Formatting
    ax.set_title(model_name, fontsize=font_sz)
    
    if xlabel:
        ax.set_xlabel("QPS", fontsize=font_sz, labelpad=1)
    if ylabel:
        ax.set_ylabel("Mean TTFT (s)", fontsize=font_sz, labelpad=1)
    
    ax.set_xlim(left=0)
    ax.set_ylim(0, 5.0)  # show TTFT from 0 to 5 seconds
    ax.tick_params(axis='both', which='major', pad=1)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add legend above the plot (like in plot_quality_vs_ttft)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Reorder handles and labels according to LEGEND_ORDER
            # Create a mapping from display label to (handle, original_key)
            label_to_handle = {}
            for h, lbl in zip(handles, labels):
                # Find the original key for this display label
                original_key = lbl
                # Find the method key that matches this display label
                for key, display_lbl in LEGEND_LABEL_MAP.items():
                    if display_lbl == lbl:
                        original_key = key
                        break
                # If not in map, it's a raw method name
                if original_key not in LEGEND_ORDER:
                    for key in LEGEND_ORDER:
                        if key in original_key.lower():
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


def plot_qps_itl_subplot(ax, directory_name, xlabel=False, ylabel=False, show_legend=False):
    """
    Plot QPS vs ITL for a single model
    """
    data = load_model_data(directory_name)
    
    # Get clean model name (remove "cache-serve/" prefix if present)
    model_name = os.path.basename(directory_name)
    
    if data is None:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center',
                transform=ax.transAxes, fontsize=font_sz)
        ax.set_title(model_name, fontsize=font_sz)
        return
    
    # Plot each method/score combination
    for idx, ((method, score), subdf) in enumerate(data.groupby(["method", "score"])):
        jitter = (idx * 0.02)
        x = subdf["qps"] + jitter
        y = subdf["mean_itl"]
        
        # Get custom legend label
        display_label = LEGEND_LABEL_MAP.get(method, method)
        
        # Get marker shape
        marker = LEGEND_MARKERS.get(method, 'o')
        
        ax.plot(
            x, y,
            marker=marker,
            markersize=3 if marker == '*' else 2,
            linewidth=1,
            label=display_label,
        )
    
    # Formatting
    ax.set_title(model_name, fontsize=font_sz)
    
    if xlabel:
        ax.set_xlabel("QPS", fontsize=font_sz, labelpad=1)
    if ylabel:
        ax.set_ylabel("Mean ITL (s)", fontsize=font_sz, labelpad=1)
    
    ax.set_xlim(left=0)
    ax.set_ylim(0, 0.2)  # show ITL from 0 to 0.2 seconds
    ax.tick_params(axis='both', which='major', pad=1)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add legend above the plot (like in plot_quality_vs_ttft)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Reorder handles and labels according to LEGEND_ORDER
            # Create a mapping from display label to (handle, original_key)
            label_to_handle = {}
            for h, lbl in zip(handles, labels):
                # Find the original key for this display label
                original_key = lbl
                # Find the method key that matches this display label
                for key, display_lbl in LEGEND_LABEL_MAP.items():
                    if display_lbl == lbl:
                        original_key = key
                        break
                # If not in map, it's a raw method name
                if original_key not in LEGEND_ORDER:
                    for key in LEGEND_ORDER:
                        if key in original_key.lower():
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
fig, axes = plt.subplots(2, 3, figsize=(7, 3))
wspace = 0.35
hspace = 0.35
plt.subplots_adjust(wspace=wspace, hspace=hspace)

# First row: QPS vs TTFT for three models
for idx, model_dir in enumerate(MODEL_DIRS):
    ax = axes[0, idx]
    xlabel = False  # No xlabel for top row
    ylabel = (idx == 0)  # Only leftmost subplot has ylabel
    show_legend = (idx == 0)  # Only show legend on first subplot
    
    model_name = os.path.basename(model_dir)
    
    if os.path.exists(model_dir):
        print(f"[LOAD] Plotting QPS vs TTFT for {model_name}")
        plot_qps_ttft_subplot(ax, model_dir, xlabel=xlabel, ylabel=ylabel, show_legend=show_legend)
    else:
        print(f"[WARN] Directory not found: {model_dir}")
        ax.text(0.5, 0.5, f"{model_name}\n(not found)", ha='center', va='center',
                transform=ax.transAxes, fontsize=font_sz)
        ax.set_title(model_name, fontsize=font_sz)
        if ylabel:
            ax.set_ylabel("Mean TTFT (s)", fontsize=font_sz, labelpad=1)
        ax.grid(True, alpha=0.3, linewidth=0.5)

# Second row: QPS vs ITL for three models
for idx, model_dir in enumerate(MODEL_DIRS):
    ax = axes[1, idx]
    xlabel = True  # xlabel for bottom row
    ylabel = (idx == 0)  # Only leftmost subplot has ylabel
    show_legend = False  # No legend in second row
    
    model_name = os.path.basename(model_dir)
    
    if os.path.exists(model_dir):
        print(f"[LOAD] Plotting QPS vs ITL for {model_name}")
        plot_qps_itl_subplot(ax, model_dir, xlabel=xlabel, ylabel=ylabel, show_legend=show_legend)
    else:
        print(f"[WARN] Directory not found: {model_dir}")
        ax.text(0.5, 0.5, f"{model_name}\n(not found)", ha='center', va='center',
                transform=ax.transAxes, fontsize=font_sz)
        ax.set_title(model_name, fontsize=font_sz)
        if xlabel:
            ax.set_xlabel("QPS", fontsize=font_sz, labelpad=1)
        if ylabel:
            ax.set_ylabel("Mean ITL (s)", fontsize=font_sz, labelpad=1)
        ax.grid(True, alpha=0.3, linewidth=0.5)

# Save as PDF
output_path = "qps_ttft_itl.pdf"
plt.savefig(output_path, bbox_inches="tight")
print(f"[SAVE] Figure saved to {output_path}")
