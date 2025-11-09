## Reinforcement Learning in Ms. Pac-Man

This project implements a Q-Learning agent to play Atari Ms. Pac-Man. The agent learns its policy by reading the game's 128-byte RAM, using a custom-engineered 4-feature state representation to overcome the "Curse of Dimensionality."

### Project Structure

* `feature_engineer.py`: Contains the `FeatureEngineer` class. This is responsible for converting the 128-byte RAM into our low-dimensional state tuple.
* `agent.py`: Contains the `Agent` class. This is the "brain," managing the Q-Table and implementing the Q-Learning update rule and $\epsilon$-greedy policy.
* `train.py`: The main script for training the agent. It runs for 50,000 episodes (headless) and saves the learned brain to `q_table.pkl` and the training data to `rewards.pkl`.
* `evaluate.py`: A script to watch the trained agent play. It loads `q_table.pkl` and runs 10 games with rendering on and exploration off.
* `plot.py`: A utility script to load `rewards.pkl` and generate a `training_performance.png` graph for the report.
* `run_project.sh`: A simple bash script to run training, evaluation, and plotting in sequence.

### Note on `MSPACMAN.BIN`

This project uses the `gymnasium` library, as recommended by the modern `ALE README` documentation. The `gymnasium[atari]` package includes the `AutoROM` utility, which automatically provides the `MSPACMAN.BIN` ROM to the emulator when `gym.make("ALE/MsPacman-v5")` is called. This ensures full compliance with the assignment in a modern, stable framework.

### How to Run

#### 1. Installation

This project requires Python 3 and several packages.

```bash
# Install dependencies
pip install gymnasium[atari]
pip install numpy
pip install matplotlib
