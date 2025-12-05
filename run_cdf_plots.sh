#!/bin/bash

# Methods to process
methods=("keydiff" "knorm" "snapkv")

for method in "${methods[@]}"; do
    echo "Processing: $method"
    # Create temp file with modified METHOD
    sed "s/^METHOD = .*/METHOD = '$method'/" plot_keydiff_cdf.py > temp_plot.py
    /Users/xiaokun/miniconda3/envs/VidGen/bin/python3 temp_plot.py
    rm temp_plot.py
done

echo "Done!"
