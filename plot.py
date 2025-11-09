import pickle
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "training_data.pkl"
PLOT_FILE_1 = "training_performance.png"
PLOT_FILE_2 = "score_distribution.png"
MOVING_AVG_WINDOW = 100

def moving_average(data, window_size):
    """Calculates the moving average of a list."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def main():
    # --- 1. Load the Data ---
    try:
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Could not find data file: {DATA_FILE}")
        print("Please run train.py first to generate the data.")
        return
        
    episode_rewards = data["rewards"]
    epsilon_values = data["epsilons"]
    
    print(f"Data loaded. Total episodes: {len(episode_rewards)}")

    # --- 2. Create Plot 1: Performance vs. Epsilon ---
    
    # Process the reward data
    avg_rewards = moving_average(episode_rewards, MOVING_AVG_WINDOW)
    
    # Create an x-axis that aligns with the averaged data
    x_axis_avg = np.arange(MOVING_AVG_WINDOW, len(episode_rewards) + 1)
    
    # Create the plot with two Y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot 1: Average Reward
    color = 'tab:blue'
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel(f'{MOVING_AVG_WINDOW}-Episode Avg Reward', color=color)
    ax1.plot(x_axis_avg, avg_rewards, color=color, label='Avg. Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create the second Y-axis for Epsilon
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Epsilon (Exploration Rate)', color=color)
    ax2.plot(np.arange(1, len(epsilon_values) + 1), epsilon_values, color=color, linestyle='--', label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Agent Performance vs. Exploration During Training')
    fig.tight_layout()  # Adjust plot to prevent labels from overlapping
    
    # Save the first plot
    plt.savefig(PLOT_FILE_1)
    print(f"Plot 1 saved to {PLOT_FILE_1}")

    # --- 3. Create Plot 2: Score Distribution ---
    plt.figure(figsize=(10, 6))
    plt.hist(episode_rewards, bins=100, color='tab:green')
    plt.title('Distribution of Final Episode Scores (All Episodes)')
    plt.xlabel('Final Score')
    plt.ylabel('Frequency (Number of Episodes)')
    plt.grid(axis='y', alpha=0.75)
    
    # Save the second plot
    plt.savefig(PLOT_FILE_2)
    print(f"Plot 2 saved to {PLOT_FILE_2}")

if __name__ == "__main__":
    main()
