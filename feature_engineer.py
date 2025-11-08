import numpy as np

class FeatureEngineer:
    """
    Transforms the 128-byte RAM into a simple state for our agent.
    
    Our new state is a 4-feature tuple:
    1. (f1) Binned distance to the nearest ghost (to AVOID)
    2. (f2) Binned distance to the fruit (to PURSUE)
    3. (f3) Player's current direction (for context)
    4. (f4) Binned number of dots eaten (to INCENTIVIZE EATING)
    """
    
    def __init__(self):
        # --- 1. Define RAM Addresses from ram_annotations.py ---
        
        # Player
        self.PLAYER_X_ADDR = 10
        self.PLAYER_Y_ADDR = 16
        self.PLAYER_DIRECTION_ADDR = 56
        
        # Ghosts (in lists for easy looping)
        self.GHOST_X_ADDRS = [9, 8, 7, 6]  # Blinky, Pinky, Inky, Sue
        self.GHOST_Y_ADDRS = [15, 14, 13, 12] # Blinky, Pinky, Inky, Sue
        
        # Fruit
        self.FRUIT_X_ADDR = 11
        self.FRUIT_Y_ADDR = 17
        
        # Dots Eaten
        self.DOTS_EATEN_ADDR = 119
        
        # --- 2. Define Discretization "Bins" ---
        # Bins for distances: (0-10), (10-30), (30-80), (80+)
        self.DISTANCE_BINS = [10, 30, 80]
        
        # Bins for dots eaten. There are ~250 dots.
        # Let's bin by quarters: (0-60), (60-120), (120-180), (180+)
        self.DOTS_BINS = [60, 120, 180]


    def _get_manhattan_distance(self, x1, y1, x2, y2):
        """
        Calculates the Manhattan distance (L1 norm).
        Casts to 'int' to prevent 8-bit overflow errors.
        """
        return abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))

    def _discretize(self, value, bins):
        """
        A general-purpose binning function.
        """
        return int(np.digitize(value, bins))


    def get_state(self, ram):
        """
        The main public function.
        Takes the 128-byte 'ram' and returns our 4-feature state tuple:
        (f1_ghost_dist, f2_fruit_dist, f3_player_dir, f4_dots_bin)
        """
        
        # Get Player's position
        player_x = ram[self.PLAYER_X_ADDR]
        player_y = ram[self.PLAYER_Y_ADDR]
        
        # --- f1: Distance to NEAREST GHOST (to AVOID) ---
        min_ghost_dist = float('inf')
        for i in range(4): # Loop over all 4 ghosts
            ghost_x = ram[self.GHOST_X_ADDRS[i]]
            ghost_y = ram[self.GHOST_Y_ADDRS[i]]
            dist = self._get_manhattan_distance(player_x, player_y, ghost_x, ghost_y)
            if dist < min_ghost_dist:
                min_ghost_dist = dist
        
        f1_ghost_dist = self._discretize(min_ghost_dist, self.DISTANCE_BINS)
        
        # --- f2: Distance to FRUIT (to PURSUE) ---
        fruit_x = ram[self.FRUIT_X_ADDR]
        fruit_y = ram[self.FRUIT_Y_ADDR]
        fruit_dist = self._get_manhattan_distance(player_x, player_y, fruit_x, fruit_y)
        f2_fruit_dist = self._discretize(fruit_dist, self.DISTANCE_BINS)
            
        # --- f3: Player's current direction ---
        f3_player_direction = ram[self.PLAYER_DIRECTION_ADDR]
        
        # --- f4: Dots Eaten Bin (to INCENTIVIZE EATING) ---
        dots_eaten = ram[self.DOTS_EATEN_ADDR]
        f4_dots_bin = self._discretize(dots_eaten, self.DOTS_BINS)
        
        
        # --- Return the final state ---
        return (f1_ghost_dist, f2_fruit_dist, f3_player_direction, f4_dots_bin)
