import gymnasium as gym
import ale_py
import time
from feature_engineer import FeatureEngineer # Import our class

gym.register_envs(ale_py)

print("Creating Ms. Pac-Man environment...")
env = gym.make("ALE/MsPacman-v5", render_mode="human", obs_type="ram")
print("Environment created.")

# --- 2. Create your Feature Engineer ---
fe = FeatureEngineer()
print("FeatureEngineer object created.")

(obs, info) = env.reset()
print("Environment reset (game starting).")
print("-" * 30)
print("Running main loop...")
print("Now printing the simplified 3-feature state tuple:")
print("(GhostDist, FruitDist, PlayerDir)") # <-- New labels

# --- 4. The Main Loop ---
for i in range(2000): 
    
    action = env.action_space.sample() 
    (obs, reward, terminated, truncated, info) = env.step(action)

    # --- 4c. GET THE SIMPLIFIED STATE! ---
    state = fe.get_state(obs) 
    
    # Print our new, simple state
    # \r (carriage return) makes it update on one line
    print(f"Step: {i:04d} | State: {state}                ", end="\r")

    if terminated or truncated:
        print("\nGame over! Resetting...")
        (obs, info) = env.reset()

    time.sleep(0.01)

# --- 5. Clean Up ---
env.close()
print("\nLoop finished. Environment closed.")
