#!/bin/bash

# A values to process (0.1 to 1.0 in increments of 0.1)
for a_value in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "Processing: alpha = $a_value"
    # Create temp file with modified A
    sed "s/^alpha = .*/alpha = $a_value/" plot_optimal_compression_by_category.py > temp_plot_optimal.py
    /Users/xiaokun/miniconda3/envs/VidGen/bin/python3 temp_plot_optimal.py
    rm temp_plot_optimal.py
done

echo "Done!"

