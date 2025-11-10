import gymnasium as gym
import ale_py
from feature_engineer import FeatureEngineer
from agent import Agent
import time
import argparse
from gymnasium.wrappers import RecordVideo # <-- 1. IMPORT THE WRAPPER

# --- 1. Register Environment ---
gym.register_envs(ale_py)

# --- 2. Parameters ---
NUM_EPISODES = 10
Q_TABLE_FILE = "q_table.pkl"
VIDEO_FOLDER = "videos" # <-- 2. DEFINE A FOLDER FOR VIDEOS

def main(args):
    # --- 3. Initialization ---
    seed = args.seed
    
    # We must create the base environment *first*
    # Notice render_mode="rgb_array"
    # This tells gym to prepare the frames for the video recorder
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array", obs_type="ram")
    
    # --- 4. WRAP THE ENVIRONMENT FOR VIDEO ---
    # This is the key change.
    # It will record all 10 episodes into the VIDEO_FOLDER
    env = RecordVideo(
        env,
        VIDEO_FOLDER,
        episode_trigger=lambda e: True, # Record every episode
        name_prefix=f"eval-seed-{seed}" # Name files clearly
    )
    
    num_actions = env.action_space.n
    fe = FeatureEngineer()
    
    agent = Agent(
        action_space_n=num_actions,
        alpha=0, gamma=0, epsilon=0,
        epsilon_min=0, epsilon_decay=0
    )
    
    # --- 5. Load the Trained Brain ---
    try:
        agent.load_q_table(Q_TABLE_FILE)
    except FileNotFoundError:
        print(f"ERROR: Could not find Q-Table file: {Q_TABLE_FILE}")
        env.close()
        return

    print("--- Starting Evaluation (Recording to Video) ---")
    print(f"Loading agent from {Q_TABLE_FILE}")
    print(f"Running for {NUM_EPISODES} episodes with Epsilon=0.")
    print(f"Videos will be saved in '{VIDEO_FOLDER}'")

    # --- 6. The Evaluation Loop ---
    for episode in range(NUM_EPISODES):
        
        (obs, info) = env.reset(seed=seed + episode)
        state = fe.get_state(obs)
        
        total_reward = 0
        terminated = False
        truncated = False
        
        print(f"\n--- Starting Episode {episode + 1}/{NUM_EPISODES} ---")

        while not terminated and not truncated:
            action = agent.choose_action(state)
            (next_obs, reward, terminated, truncated, info) = env.step(action)
            next_state = fe.get_state(next_obs)
            state = next_state
            total_reward += reward
            
            # We don't need time.sleep() anymore
        
        print(f"Episode {episode + 1} Finished. Final Score: {total_reward}")

    # --- 7. Clean Up ---
    env.close()
    print("\n--- Evaluation Finished ---")
    print(f"Video files are saved in the '{VIDEO_FOLDER}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained agent and record videos.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for evaluation.")
    args = parser.parse_args()
    main(args)
