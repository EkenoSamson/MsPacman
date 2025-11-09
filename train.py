import gymnasium as gym
import ale_py
from feature_engineer import FeatureEngineer
from agent import Agent
import time
import numpy as np
import pickle
import argparse  # For command-line arguments
import random

# --- 1. Register Environment ---
gym.register_envs(ale_py)

# --- 2. Hyperparameters ---
NUM_EPISODES = 50000      # Set to 50000 for full run
LEARNING_RATE = 0.1       # alpha
DISCOUNT_FACTOR = 0.99    # gamma
EPSILON = 1.0             # Initial exploration rate
EPSILON_MIN = 0.01        # Minimum exploration rate
EPSILON_DECAY = 0.9999    # Decay rate
SAVE_EVERY = 500          # Save the Q-Table every 500 episodes

# Filepaths for saving/loading
Q_TABLE_FILE = "q_table.pkl"
DATA_FILE = "training_data.pkl" # Will store rewards AND epsilons

def main(args):
    # --- 3. Initialization ---
    
    # --- Set the Seed (for rubric) ---
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    
    env = gym.make("ALE/MsPacman-v5", obs_type="ram")
    num_actions = env.action_space.n
    
    fe = FeatureEngineer()
    agent = Agent(
        action_space_n=num_actions,
        alpha=LEARNING_RATE,
        gamma=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY
    )
    
    print("--- Starting Training ---")
    print(f"Running for {NUM_EPISODES} episodes with Seed: {seed}")
    
    episode_rewards = []
    epsilon_values = [] # List to store epsilon values

    # --- 4. The Main Training Loop ---
    for episode in range(NUM_EPISODES):
        
        # Seed the reset for reproducibility
        (obs, info) = env.reset(seed=seed + episode)
        state = fe.get_state(obs)
        
        total_reward = 0
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action = agent.choose_action(state)
            (next_obs, reward, terminated, truncated, info) = env.step(action)
            next_state = fe.get_state(next_obs)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
        
        # --- End of Episode ---
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        epsilon_values.append(agent.epsilon) # Store current epsilon
        
        # --- Logging ---
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {(episode + 1):5d} | "
                  f"Avg Reward (last 100): {avg_reward:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # --- Save Progress ---
        if (episode + 1) % SAVE_EVERY == 0:
            agent.save_q_table(Q_TABLE_FILE)

    # --- 5. Clean Up ---
    env.close()
    agent.save_q_table(Q_TABLE_FILE) # Final save
    
    # --- Save the Data for Plotting ---
    training_data = {
        "rewards": episode_rewards,
        "epsilons": epsilon_values
    }
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(training_data, f)
        
    print(f"--- Training Finished. Data saved to {DATA_FILE} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Q-Learning agent for Ms. Pac-Man.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    main(args)
