import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_utility_length_dist import calculate_utility

# Font configuration
font_sz = 8
font = "Arial"
plt.rcParams['font.family'] = font
plt.rcParams['font.size'] = font_sz

# Configuration
# DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/scores_tokens.csv' # mistral 7b
DATA_DIR = '/Users/xiaokun/Desktop/cacheserve/scores_tokens.csv' # mistral 7b
PLOT_DIR = '/Users/xiaokun/Desktop/cacheserve/optimal_compression_plots/'
METHODS = ['keydiff', 'knorm', 'snapkv']
ANSWER_INDEX = 1
alpha = 0.5
COMPRESSION_RATES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Dataset categories
# CATEGORIES = {
#     'Multi-hop Contexts': ['hotpotqa', 'multi_news'],
#     'Single-hop Contexts': ['triviaqa', 'narrativeqa', 'gov_report', 'qasper'],
#     'Dialogue': ['samsum', 'qmsum']
# }
# CATEGORIES = {
#     'Multi-hop Contexts': ['qmsum'],
#     'Single-hop Contexts': ['narrativeqa'],
#     'Dialogue': ['samsum']
# }

# Draw 
CATEGORIES = {
    'samsum': ['samsum'],
    'triviaqa': ['triviaqa'],
    'multi_news': ['multi_news'],
    '2wikimqa': ['2wikimqa'],
    'qasper': ['qasper'],
    'narrativeqa': ['narrativeqa']
}

# Calculate the average length of each category
def calculate_average_length(df):
    """
    Calculate the average length of each category.
    """
    for category in CATEGORIES.keys():
        df_category = df[df['dataset'].isin(CATEGORIES[category])]
        average_length = df_category['length'].mean()
        print(f"Average length of {category}: {average_length}")

def load_data():
    """Load the CSV data."""
    df = pd.read_csv(DATA_DIR)
    return df

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
        # print("current idx: ", idx)
        dataset = row['dataset']
        context_length = row['length']
        
        # Determine which category this dataset belongs to
        category = None
        for cat_name, datasets in CATEGORIES.items():
            if dataset in datasets:
                category = cat_name
                break
        
        assert category is not None, f"Category is None for entry {idx} in dataset {dataset}"
        
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
                    
                    assert col_name in df.columns, f"Column {col_name} not found in dataframe"

                    quality_score = row[col_name]
                    assert quality_score is not None, f"Quality score is None for entry {idx} in dataset {dataset}, col name: {col_name}"
                    utility = calculate_utility(alpha, quality_score, rate, context_length, L)
                    # if idx == 30:
                    #     # print alpha, quality_score, rate, context_length, L
                    #     print(f"method: {method}, rate: {rate}, answer_idx: {answer_idx}, alpha: {alpha}, quality_score: {quality_score}, rate: {rate}, context_length: {context_length}, L: {L}")
                    #     print(f"utility: {utility}")
                    assert utility is not None, f"Utility is None for entry {idx} in dataset {dataset}, col name: {col_name}"
                    utilities.append(utility)
                
                assert len(utilities) == 100, f"Expected 100 utilities for {method}_{rate_str}_answer{answer_idx}, got {len(utilities)}"
                avg_utility = np.mean(utilities)
                assert avg_utility is not None, f"Average utility is None for entry {idx} in dataset {dataset}, col name: {col_name}"
                if avg_utility > best_avg_utility:
                    best_avg_utility = avg_utility
                    best_rate = rate
        # if idx == 30:
        #     print(f"utilities: {utilities}")
        #     print(f"avg_utility: {avg_utility}")
        #     print(f"best_avg_utility: {best_avg_utility}")
        #     print(f"best_rate: {best_rate}")
        #     print(f"category: {category}")
        #     print(f"dataset: {dataset}")
        #     print(f"context_length: {context_length}")
        # Add this entry's optimal compression rate to the category
        # print(f"Best rate: {best_rate} for entry {idx} in dataset {dataset}")
        # assert best_rate > 0
        results[category].append(best_rate)
    
    return results

def plot_optimal_compression_distribution(results):
    """
    Plot bar chart with error bars of optimal compression rate distributions for each category.
    results: dict with category -> [list of optimal compression rates]
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Prepare data for bar chart
    means = []
    errors = []
    labels = []
    
    for category in CATEGORIES.keys():
        if category in results and len(results[category]) > 0:
            rates = results[category]
            means.append(np.mean(rates))
            # Use standard error of the mean for error bars
            errors.append(np.std(rates))
            labels.append(category)
    
    # Create bar chart with error bars
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, 
                   color='steelblue', alpha=0.7, 
                   error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax.set_ylabel('Optimal Compression Ratio', fontsize=font_sz)
    # ax.set_xlabel('Dataset', fontsize=font_sz)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    methods_str = '_'.join(METHODS)
    output_filename = f'optimal_compression_by_category_{methods_str}_answers1-100_alpha{alpha}.pdf'
    plt.savefig(PLOT_DIR + output_filename, bbox_inches='tight')
    print(f"\nSaved plot to {output_filename}")
    
    # Print statistics
    print("\nOptimal compression rate statistics by category:")
    for category in CATEGORIES.keys():
        if category in results and len(results[category]) > 0:
            rates = results[category]
            print(f"\n{category}:")
            print(f"  Entries: {len(rates)}")
            print(f"  Mean: {np.mean(rates):.3f}")
            print(f"  Median: {np.median(rates):.3f}")
            print(f"  Standard deviation: {np.std(rates):.3f}")
            # calculate CV
            cv = np.std(rates) / np.mean(rates)
            print(f"  CV: {cv:.3f}")

def main():
    """Main function to run the analysis."""
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} total entries")
    print(f"Max context length (L): {df['length'].max()}")
    print(f"Average length of each category: {calculate_average_length(df)}")
    print(f"\nUsing methods: {METHODS}, answer indices: 1-100 (averaged), alpha: {alpha}")
    print(f"Compression rates to evaluate: {COMPRESSION_RATES}")
    
    # Find optimal compression rate for each entry (across all methods and answer indices)
    results = find_optimal_compression_per_entry(df)
    
    # Plot distributions
    plot_optimal_compression_distribution(results)

if __name__ == "__main__":
    main()

