import numpy as np
import pickle

class Agent:
    """
    The Q-Learning Agent. This is the "brain" that learns
    to play Ms. Pac-Man.
    """
    
    def __init__(self, action_space_n, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        """
        Initialize the Agent.
        
        Parameters:
        - action_space_n: The number of possible actions (e.g., 9 for Ms. Pac-Man)
        - alpha: The learning rate (e.g., 0.1)
        - gamma: The discount factor (e.g., 0.99)
        - epsilon: The initial exploration rate (e.g., 1.0)
        - epsilon_min: The minimum exploration rate (e.g., 0.01)
        - epsilon_decay: The rate at which epsilon decays (e.g., 0.999)
        """
        self.action_space_n = action_space_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # The Q-Table is a dictionary where:
        # Key = state tuple (e.g., (3, 2, 56))
        # Value = a list of Q-values for each action [Q(s, a0), Q(s, a1), ...]
        self.q_table = {}

    def get_q_table(self, state):
        """
        A helper function to get the Q-values for a state.
        If the state is new, it initializes Q-values to 0.
        """
        if state not in self.q_table:
            # Initialize Q-values for a new state to all zeros
            self.q_table[state] = [0.0] * self.action_space_n
        
        return self.q_table[state]

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        # 1. EXPLORATION (take a random action)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_n)
        
        # 2. EXPLOITATION (take the best known action)
        q_values = self.get_q_table(state)
        return np.argmax(q_values) # Returns the *index* of the best action

    def update(self, state, action, reward, next_state, terminated):
        """
        Implements the Q-Learning update rule.
        This is the core of the learning algorithm.
        
        The Bellman Equation (Q-Learning update):
        Q(s, a) <- Q(s, a) + alpha * (r + gamma * max(Q(s', a')) - Q(s, a))
        """
        
        # 1. Get the current Q-value: Q(s, a)
        current_q = self.get_q_table(state)[action]
        
        # 2. Get the max future Q-value: max(Q(s', a'))
        # If the game is over (terminated), the future value is 0.
        max_future_q = 0.0
        if not terminated:
            max_future_q = np.max(self.get_q_table(next_state))
            
        # 3. Calculate the "target" value: r + gamma * max(Q(s', a'))
        target = reward + self.gamma * max_future_q
        
        # 4. Calculate the new Q-value using the update rule
        new_q = current_q + self.alpha * (target - current_q)
        
        # 5. Update the Q-Table
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """
        Reduces the exploration rate (epsilon) over time,
        so the agent exploits more as it learns.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filepath):
        """Saves the Q-Table to a file for later use."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-Table saved to {filepath}")

    def load_q_table(self, filepath):
        """Loads a pre-trained Q-Table from a file."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-Table loaded from {filepath}")
