#!/bin/bash

# This script runs the full pipeline:
# 1. Trains the agent (using default seed 42)
# 2. Evaluates the agent (using default seed 1337)
# 3. Generates plots from the training data

echo "--- 1. Starting Training (Seed 42) ---"
# This will run for a long time and create q_table.pkl and training_data.pkl
python3 train.py --seed 42

echo "--- 2. Starting Evaluation (Seed 1337) ---"
# This will load q_table.pkl and run 10 visible episodes
python3 evaluate.py --seed 1337

echo "--- 3. Generating Plots ---"
# This will load training_data.pkl and create two .png files
python3 plot.py

echo "--- All steps complete. ---"
