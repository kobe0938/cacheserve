import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Font configuration
font_sz = 8
font = "Arial"
plt.rcParams['font.family'] = font
plt.rcParams['font.size'] = font_sz

# Configuration
DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/presses_scores_1and2and3.csv' # llama 8b
PLOT_DIR = '/Users/xiaokun/Desktop/cacheserve/length_distribution_plots/'
# ANSWER_INDEX = 12
alpha = 1  # Utility function constant, does not affect the results in this graph
COMPRESSION_RATE = 0.6  # Fixed compression rate for analysis
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
    Uses average utility across answer indices 1-50.
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
        best_avg_utility = -float('inf')
        
        # Calculate utility for each method
        for method in METHODS:
            # Calculate average utility across answer indices 1-50
            utilities = []
            for answer_idx in range(1, 51):
                col_name = f"{method}_{rate_str}_answer{answer_idx}"
                
                if col_name not in df.columns:
                    continue
                
                quality_score = row[col_name]
                utility = calculate_utility(alpha, quality_score, compression_rate, context_length, L)
                utilities.append(utility)
            
            # if idx % 100 == 0:
            #     print(f"Processing entry {idx}...")
            #     print(f"Method: {method}")
            #     print(f"Utilities: {utilities}")
            assert len(utilities) == 50, f"Expected 50 utilities for {method}_{rate_str}_answer{answer_idx}, got {len(utilities)}"
            # If we have utilities for this method, take the average
            avg_utility = np.mean(utilities)
            if avg_utility > best_avg_utility:
                best_avg_utility = avg_utility
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
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    
    rate_str = str(compression_rate).replace('.', 'p')
    
    # Prepare data for box plot
    data = []
    labels = []
    colors = ['steelblue', 'darkorange', 'forestgreen']
    
    for method in METHODS:
        if method in results and len(results[method]) > 0:
            data.append(results[method])
            labels.append(f"{method}")
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Style the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Method', fontsize=font_sz)
    ax.set_ylabel('Length', fontsize=font_sz)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_filename = f'length_distribution_best_method_rate{rate_str}_answers1-50_alpha{alpha}_compression_rate{compression_rate}.pdf'
    plt.savefig(PLOT_DIR + output_filename, bbox_inches='tight')
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
    
    # Find best method for each entry (averaged across answer indices 1-50)
    results = find_best_method_per_entry(df, COMPRESSION_RATE)
    
    # Print the average length of the entries won by each method
    print("\nAverage length of entries won by each method:")
    for method in METHODS:
        if method in results and len(results[method]) > 0:
            print(f"{method}: {np.mean(results[method]):.1f}")
    
    # Plot distributions
    plot_length_distributions(results, COMPRESSION_RATE)

if __name__ == "__main__":
    main()

