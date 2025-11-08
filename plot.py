import pickle
import numpy as np
import matplotlib.pyplot as plt

REWARDS_FILE = "rewards.pkl"
PLOT_FILE = "training_performance.png"
MOVING_AVG_WINDOW = 100

def moving_average(data, window_size):
    """Calculates the moving average of a list."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def main():
    # --- 1. Load the Data ---
    try:
        with open(REWARDS_FILE, 'rb') as f:
            episode_rewards = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Could not find rewards file: {REWARDS_FILE}")
        print("Please run train.py first to generate the rewards.")
        return

    # --- 2. Process the Data ---
    # Calculate the moving average
    avg_rewards = moving_average(episode_rewards, MOVING_AVG_WINDOW)
    
    # Create an x-axis that aligns with the averaged data
    # (e.g., the 100th data point is the avg of episodes 1-100)
    x_axis = np.arange(MOVING_AVG_WINDOW, len(episode_rewards) + 1)
    
    print(f"Data loaded. Total episodes: {len(episode_rewards)}")
    print(f"Moving average calculated with window {MOVING_AVG_WINDOW}.")

    # --- 3. Create the Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, avg_rewards, label=f'{MOVING_AVG_WINDOW}-Episode Moving Average')
    plt.title('Agent Training Performance (4-Feature Model)')
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    
    # --- 4. Save the Plot ---
    plt.savefig(PLOT_FILE)
    print(f"Plot saved to {PLOT_FILE}")
    print("You can now add this image to your report.")

if __name__ == "__main__":
    main()
