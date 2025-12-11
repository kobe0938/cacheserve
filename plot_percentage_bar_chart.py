import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Font configuration
font_sz = 8
font = "Arial"
plt.rcParams['font.family'] = font
plt.rcParams['font.size'] = font_sz

# Configuration
DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/mistral7b_555.csv'
PLOT_DIR = '/Users/xiaokun/Desktop/cacheserve/percentage_bar_chart_plots/'
COMPRESSION_RATE = 0.6  # Fixed compression rate for analysis
METHODS = ['keydiff', 'knorm', 'snapkv']

def load_data():
    """Load the CSV data."""
    df = pd.read_csv(DATA_DIR)
    return df

def find_best_method_per_entry(df, compression_rate):
    """
    For each entry, find the method with highest average quality score at the given compression rate.
    Uses average quality score across answer indices 51-100.
    Returns dict: {method: count} - count of how many entries each method won
    """
    rate_str = str(compression_rate).replace('.', 'p')
    
    # Initialize results dictionary
    results = {method: 0 for method in METHODS}
    
    # For each entry, find the best method
    for idx, row in df.iterrows():
        best_method = None
        best_avg_quality = -float('inf')
        
        # Calculate average quality score for each method
        for method in METHODS:
            # Calculate average quality across answer indices 51-100
            quality_scores = []
            for answer_idx in range(51, 101):
                col_name = f"{method}_{rate_str}_answer{answer_idx}"
                
                if col_name not in df.columns:
                    continue
                
                quality_score = row[col_name]
                quality_scores.append(quality_score)
            
            assert len(quality_scores) == 50, f"Expected 50 quality scores for {method}_{rate_str}_answer{answer_idx}, got {len(quality_scores)}"
            # If we have quality scores for this method, take the average
            avg_quality = np.mean(quality_scores)
            if avg_quality > best_avg_quality:
                best_avg_quality = avg_quality
                best_method = method
        
        # Count this entry towards the best method
        if best_method is not None:
            results[best_method] += 1
    
    return results

def plot_bar_chart(results, compression_rate):
    """
    Plot bar chart showing percentage of entries won by each method.
    results: dict with method -> count
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    
    rate_str = str(compression_rate).replace('.', 'p')
    
    # Calculate total entries
    total_entries = sum(results.values())
    
    # Prepare data for bar chart
    methods = []
    percentages = []
    colors = ['steelblue', 'darkorange', 'forestgreen']
    
    for method in METHODS:
        if method in results:
            methods.append(method)
            percentage = (results[method] / total_entries) * 100
            percentages.append(percentage)
    
    # Create bar chart
    bars = ax.bar(methods, percentages, color=colors, alpha=0.7)
    
    ax.set_xlabel('Method', fontsize=font_sz)
    ax.set_ylabel('Percentage (%)', fontsize=font_sz)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_filename = f'percentage_bar_chart_rate{rate_str}_answers51-100_compression_rate{compression_rate}.pdf'
    plt.savefig(PLOT_DIR + output_filename, bbox_inches='tight')
    print(f"Saved plot to {output_filename}")
    
    # Print results
    print("\nEntries won by each method:")
    for method in METHODS:
        if method in results:
            count = results[method]
            percentage = (count / total_entries) * 100
            print(f"{method}: {count} entries ({percentage:.1f}%)")
    print(f"Total entries: {total_entries}")

def main():
    """Main function to run the analysis."""
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} entries")
    print(f"Max context length (L): {df['length'].max()}")
    
    # Find best method for each entry (averaged across answer indices 51-100)
    results = find_best_method_per_entry(df, COMPRESSION_RATE)
    
    # Plot bar chart
    plot_bar_chart(results, COMPRESSION_RATE)

if __name__ == "__main__":
    main()

