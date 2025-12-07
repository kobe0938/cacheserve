import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/presses_scores_1and2and3.csv'
PLOT_DIR = '/Users/xiaokun/Desktop/cacheserve/length_distribution_plots/'
ANSWER_INDEX = 1
A = 1  # Utility function constant
COMPRESSION_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METHODS = ['kvzip', 'knorm', 'snapkv']

def load_data():
    """Load the CSV data."""
    df = pd.read_csv(DATA_DIR)
    return df

def calculate_utility(quality_score, compression_rate, context_length, L):
    """
    Calculate utility for a single entry.
    utility = a * quality_score - (remaining_length / L)
    """
    remaining_length = (1 - compression_rate) * context_length
    utility = A * quality_score - (remaining_length / L)
    return utility

def find_best_method_per_entry(df, compression_rate):
    """
    For each entry, find the method with highest utility at the given compression rate.
    Returns dict: {method: [list of lengths where this method was best]}
    """
    L = df['length'].max()  # Global max length
    rate_str = str(compression_rate).replace('.', 'p')
    
    # Initialize results dictionary
    results = {method: [] for method in METHODS}
    
    # For each entry, find the best method
    for idx, row in df.iterrows():
        context_length = row['length']
        best_method = None
        best_utility = -float('inf')
        
        # Calculate utility for each method
        for method in METHODS:
            col_name = f"{method}_{rate_str}_answer{ANSWER_INDEX}"
            
            if col_name not in df.columns:
                continue
            
            quality_score = row[col_name]
            utility = calculate_utility(quality_score, compression_rate, context_length, L)
            
            if utility > best_utility:
                best_utility = utility
                best_method = method
        
        # Add this entry's length to the best method
        if best_method is not None:
            results[best_method].append(context_length)
    
    return results

def plot_length_distributions_grid(all_results):
    """
    Plot box plots of length distributions for each compression rate in a 3x3 grid.
    all_results: dict with compression_rate -> {method: [list of lengths]}
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, compression_rate in enumerate(COMPRESSION_RATES):
        ax = axes[idx]
        results = all_results[compression_rate]
        
        # Prepare data for box plot
        data = []
        labels = []
        
        for method in METHODS:
            if method in results and len(results[method]) > 0:
                data.append(results[method])
                labels.append(method)
        
        if len(data) > 0:
            # Create box plot
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Style the boxes
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.7)
        
        ax.set_ylabel('length', fontsize=10)
        ax.set_title(f'Rate {compression_rate}', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=8)
    
    plt.suptitle('Length Distribution per Method at Different Compression Rates', fontsize=16, y=0.995)
    plt.tight_layout()
    output_filename = f'length_distribution_grid_answer{ANSWER_INDEX}.png'
    plt.savefig(PLOT_DIR + output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_filename}")

def main():
    """Main function to run the analysis."""
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} entries")
    print(f"Max context length (L): {df['length'].max()}")
    
    # Process all compression rates
    all_results = {}
    
    for compression_rate in COMPRESSION_RATES:
        print(f"\n{'='*60}")
        print(f"Processing compression rate: {compression_rate}")
        print(f"{'='*60}")
        
        # Find best method for each entry at this compression rate
        results = find_best_method_per_entry(df, compression_rate)
        all_results[compression_rate] = results
        
        # Print statistics
        print("\nEntries won by each method:")
        total_entries = 0
        for method in METHODS:
            if method in results and len(results[method]) > 0:
                count = len(results[method])
                avg_length = np.mean(results[method])
                total_entries += count
                print(f"  {method}: {count} entries, avg_length={avg_length:.1f}")
            else:
                print(f"  {method}: 0 entries")
        print(f"Total: {total_entries} entries")
    
    # Plot all results in a 3x3 grid
    plot_length_distributions_grid(all_results)

if __name__ == "__main__":
    main()
