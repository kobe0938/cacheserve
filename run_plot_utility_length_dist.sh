#!/bin/bash

# Answer indices to process (1 to 50)
for answer_idx in 12; do
    echo "Processing: answer index $answer_idx"
    # Create temp file with modified ANSWER_INDEX
    sed "s/^ANSWER_INDEX = .*/ANSWER_INDEX = $answer_idx/" plot_utility_length_dist.py > temp_plot.py
    /Users/xiaokun/miniconda3/envs/VidGen/bin/python3 temp_plot.py
    rm temp_plot.py
done

echo "Done!"
