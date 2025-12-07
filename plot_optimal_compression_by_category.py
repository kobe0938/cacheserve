import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_utility_length_dist import calculate_utility, load_data

# Configuration
DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/scores_tokens.csv' # mistral 7b
PLOT_DIR = '/Users/xiaokun/Desktop/cacheserve/optimal_compression_plots/'
METHODS = ['keydiff', 'knorm', 'snapkv']
ANSWER_INDEX = 1
alpha = 0.5
COMPRESSION_RATES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Dataset categories
CATEGORIES = {
    'Multi-hop Contexts': ['hotpotqa', 'multi_news'],
    'Single-hop Contexts': ['triviaqa', 'narrativeqa', 'gov_report', 'qasper'],
    'Dialogue': ['samsum', 'qmsum']
}
# CATEGORIES = {
#     'Multi-hop Contexts': ['qmsum'],
#     'Single-hop Contexts': ['narrativeqa'],
#     'Dialogue': ['samsum']
# }

def find_optimal_compression_per_entry(df):
    """
    For each entry in the specified datasets, find the best method and compression rate with highest utility.
    Uses average utility across answer indices 1-50.
    Returns dict: {category: [list of optimal compression rates]}
    """
    L = df['length'].max()  # Global max length
    
    # Initialize results dictionary
    results = {category: [] for category in CATEGORIES.keys()}
    
    # Get all datasets we care about
    all_datasets = []
    for datasets in CATEGORIES.values():
        all_datasets.extend(datasets)
    
    # Filter to only the datasets we care about
    df_filtered = df[df['dataset'].isin(all_datasets)]
    
    print(f"Total entries to process: {len(df_filtered)}")
    
    # For each entry, find the optimal method and compression rate
    for idx, row in df_filtered.iterrows():
        dataset = row['dataset']
        context_length = row['length']
        
        # Determine which category this dataset belongs to
        category = None
        for cat_name, datasets in CATEGORIES.items():
            if dataset in datasets:
                category = cat_name
                break
        
        if category is None:
            continue
        
        # Calculate utility for all methods and compression rates
        best_rate = None
        best_avg_utility = -float('inf')
        
        # Loop through all methods
        for method in METHODS:
            # Loop through all compression rates
            for rate in COMPRESSION_RATES:
                rate_str = str(rate).replace('.', 'p')
                
                # Calculate average utility across answer indices 1-50
                utilities = []
                for answer_idx in range(1, 101):
                    col_name = f"{method}_{rate_str}_answer{answer_idx}"
                    
                    if col_name not in df.columns:
                        continue
                    
                    quality_score = row[col_name]
                    utility = calculate_utility(alpha, quality_score, rate, context_length, L)
                    utilities.append(utility)
                
                assert len(utilities) == 100, f"Expected 100 utilities for {method}_{rate_str}_answer{answer_idx}, got {len(utilities)}"
                avg_utility = np.mean(utilities)
                if avg_utility > best_avg_utility:
                    best_avg_utility = avg_utility
                    best_rate = rate
        
        # Add this entry's optimal compression rate to the category
        if best_rate is not None:
            results[category].append(best_rate)
    
    return results

def plot_optimal_compression_distribution(results):
    """
    Plot box plots of optimal compression rate distributions for each category.
    results: dict with category -> [list of optimal compression rates]
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    data = []
    labels = []
    colors = ['steelblue', 'darkorange', 'forestgreen']
    
    for category in ['Multi-hop Contexts', 'Single-hop Contexts', 'Dialogue']:
        if category in results and len(results[category]) > 0:
            data.append(results[category])
            labels.append(category)
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Style the boxes with different colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Optimal Compression Ratio', fontsize=12)
    ax.set_xlabel('Context Type', fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    methods_str = '_'.join(METHODS)
    output_filename = f'optimal_compression_by_category_{methods_str}_answers1-100_alpha{alpha}.png'
    plt.savefig(PLOT_DIR + output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_filename}")
    
    # Print statistics
    print("\nOptimal compression rate statistics by category:")
    for category in ['Multi-hop Contexts', 'Single-hop Contexts', 'Dialogue']:
        if category in results and len(results[category]) > 0:
            rates = results[category]
            print(f"\n{category}:")
            print(f"  Entries: {len(rates)}")
            print(f"  Mean: {np.mean(rates):.3f}")
            print(f"  Median: {np.median(rates):.3f}")
            # print(f"  Min: {np.min(rates):.3f}")
            # print(f"  Max: {np.max(rates):.3f}")
            # print(f"  25th percentile: {np.percentile(rates, 25):.3f}")
            # print(f"  75th percentile: {np.percentile(rates, 75):.3f}")

def main():
    """Main function to run the analysis."""
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} total entries")
    print(f"Max context length (L): {df['length'].max()}")
    
    print(f"\nUsing methods: {METHODS}, answer indices: 1-100 (averaged), alpha: {alpha}")
    print(f"Compression rates to evaluate: {COMPRESSION_RATES}")
    
    # Find optimal compression rate for each entry (across all methods and answer indices)
    results = find_optimal_compression_per_entry(df)
    
    # Plot distributions
    plot_optimal_compression_distribution(results)

if __name__ == "__main__":
    main()

