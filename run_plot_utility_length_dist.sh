# #!/bin/bash

# # Answer indices to process (1 to 50)
# for answer_idx in {1..50}; do
#     echo "Processing: answer index $answer_idx"
#     # Create temp file with modified ANSWER_INDEX
#     sed "s/^ANSWER_INDEX = .*/ANSWER_INDEX = $answer_idx/" plot_utility_length_dist.py > temp_plot.py
#     /Users/xiaokun/miniconda3/envs/VidGen/bin/python3 temp_plot.py
#     rm temp_plot.py
# done

# echo "Done!"

#!/bin/bash

# A values to process (0.1 to 0.9 in increments of 0.1)
for compression_rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "Processing: compression_rate = $compression_rate"
    # Create temp file with modified A
    sed "s/^COMPRESSION_RATE = .*/COMPRESSION_RATE = $compression_rate/" plot_utility_length_dist.py > temp_plot.py
    /Users/xiaokun/miniconda3/envs/VidGen/bin/python3 temp_plot.py
    rm temp_plot.py
done

echo "Done!"