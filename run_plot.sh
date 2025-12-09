#!/bin/bash

# Methods to process
for compression_rate in 0.1 0.2 0.4 0.6 0.8 0.9; do
    echo "Processing: compression_rate = $compression_rate"
    # Create temp file with modified A
    sed "s/^COMPRESSION_RATE = .*/COMPRESSION_RATE = $compression_rate/" plot_cdf.py > temp_plot.py
    /Users/xiaokun/miniconda3/envs/VidGen/bin/python3 temp_plot.py
    rm temp_plot.py
done

echo "Done!"