import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/presses_scores_1and2and3.csv'
PLOT_DIR = '/Users/xiaokun/Desktop/cacheserve/length_distribution_plots/'
ANSWER_INDEX = 12
A = 1  # Utility function constant
COMPRESSION_RATE = 0.7  # Fixed compression rate for analysis
METHODS = ['kvzip', 'knorm', 'snapkv']

def load_data():
    """Load the CSV data."""
    df = pd.read_csv(DATA_DIR)
    return df

def calculate_utility(alpha, quality_score, compression_rate, context_length, L):
    """
    Calculate utility for a single entry.
    utility = a * quality_score - (remaining_length / L)
    """
    remaining_length = (1 - compression_rate) * context_length
    utility = alpha * quality_score - (remaining_length / L)
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
            alpha = A
            utility = calculate_utility(alpha, quality_score, compression_rate, context_length, L)
            
            if utility > best_utility:
                best_utility = utility
                best_method = method
        
        # Add this entry's length to the best method
        if best_method is not None:
            results[best_method].append(context_length)
    
    return results

def plot_length_distributions(results, compression_rate):
    """
    Plot box plots of length distributions for each method.
    results: dict with method -> [list of lengths]
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    data = []
    labels = []
    colors = ['steelblue', 'darkorange', 'forestgreen']
    
    for method in METHODS:
        if method in results and len(results[method]) > 0:
            data.append(results[method])
            rate_str = str(compression_rate).replace('.', 'p')
            labels.append(f"{method}\nmethod_{rate_str}")
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Style the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('length', fontsize=12)
    ax.set_title('Length Distribution per Method', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_filename = f'length_distribution_best_method_rate{rate_str}_answer{ANSWER_INDEX}.png'
    plt.savefig(PLOT_DIR + output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_filename}")
    
    # Print results
    print("\nEntries won by each method:")
    total_entries = 0
    for method in METHODS:
        if method in results:
            count = len(results[method])
            total_entries += count
            print(f"{method}: {count} entries")
    print(f"Total entries: {total_entries}")

def main():
    """Main function to run the analysis."""
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} entries")
    print(f"Max context length (L): {df['length'].max()}")
    
    # Find best method for each entry
    results = find_best_method_per_entry(df, COMPRESSION_RATE)
    # print the average length of the entries won by each method
    for method in METHODS:
        if method in results:
            print(f"{method}: {np.mean(results[method])}")
    
    # Plot distributions
    plot_length_distributions(results, COMPRESSION_RATE)

if __name__ == "__main__":
    main()

