#!/bin/bash

echo "--- 1. Starting Training ---"
# This will run for a long time and create q_table.pkl and rewards.pkl
python3 train.py

echo "--- 2. Starting Evaluation ---"
# This will load q_table.pkl and run 10 visible episodes
python3 evaluate.py

echo "--- 3. Generating Plot ---"
# This will load rewards.pkl and create training_performance.png
python3 plot.py

echo "--- All steps complete. ---"
