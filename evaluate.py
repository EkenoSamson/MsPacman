import gymnasium as gym
import ale_py
from feature_engineer import FeatureEngineer
from agent import Agent  # We reuse our Agent class
import time

# --- 1. Register Environment ---
gym.register_envs(ale_py)

# --- 2. Parameters ---
NUM_EPISODES = 10         # How many games to watch
Q_TABLE_FILE = "q_table.pkl" # The "brain" we trained

def main():
    # --- 3. Initialization ---
    
    # !! THIS IS THE CORRECTED LINE !!
    # Changed "MsPacMan" to "MsPacman" (lowercase 'm')
    env = gym.make("ALE/MsPacman-v5", render_mode="human", obs_type="ram")
    
    num_actions = env.action_space.n
    
    fe = FeatureEngineer()
    
    # We initialize the agent, but the learning parameters
    # don't matter since we won't be training.
    agent = Agent(
        action_space_n=num_actions,
        alpha=0,
        gamma=0,
        epsilon=0, # <-- CRITICAL: Set Epsilon to 0
        epsilon_min=0,
        epsilon_decay=0
    )
    
    # --- 4. Load the Trained Brain ---
    try:
        agent.load_q_table(Q_TABLE_FILE)
    except FileNotFoundError:
        print(f"ERROR: Could not find Q-Table file: {Q_TABLE_FILE}")
        print("Please run train.py first to create the file.")
        env.close()
        return

    print("--- Starting Evaluation ---")
    print(f"Loading trained agent from {Q_TABLE_FILE}")
    print(f"Running for {NUM_EPISODES} episodes with Epsilon=0 (100% exploitation).")

    # --- 5. The Evaluation Loop ---
    for episode in range(NUM_EPISODES):
        
        (obs, info) = env.reset()
        state = fe.get_state(obs)
        
        total_reward = 0
        terminated = False
        truncated = False
        
        print(f"\n--- Starting Episode {episode + 1}/{NUM_EPISODES} ---")

        while not terminated and not truncated:
            
            # 1. Agent chooses the BEST action (no randomness)
            action = agent.choose_action(state)
            
            # 2. Environment executes the action
            (next_obs, reward, terminated, truncated, info) = env.step(action)
            
            # 3. Get the new state
            next_state = fe.get_state(next_obs)
            
            # 4. We DO NOT call agent.update(). We are only evaluating.
            
            # 5. Prepare for next iteration
            state = next_state
            total_reward += reward
            
            # Add a small delay so we can watch
            time.sleep(0.01)
        
        # --- End of Episode ---
        print(f"Episode {episode + 1} Finished. Final Score: {total_reward}")

    # --- 6. Clean Up ---
    env.close()
    print("\n--- Evaluation Finished ---")


if __name__ == "__main__":
    main()
