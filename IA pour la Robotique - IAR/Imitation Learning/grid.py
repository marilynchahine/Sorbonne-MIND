import numpy as np

# GridWorldEnvironment class
class GridWorldEnvironment:
    def __init__(self, grid_size=(5, 5), obstacles=[], stochastic=False):
        self.grid_size = grid_size
        self.state_space = np.prod(grid_size)
        self.action_space = 4  # Four possible actions: Up, Down, Left, Right
        self.state = None
        self.obstacles = obstacles
        self.set_mdp()

    def set_mdp(self):

        """ DEFINE THE MDP HERE"""

        self.mdp = None 

    def reset(self):
        # Randomly select an initial state that is not an obstacle
        # Add your code here
        random_state = np.random.choice(self.state_space)
        self.state = random_state
        return random_state


    def step(self, action):
        if self.state is None:
            raise Exception("You must call reset() before step()")

        # Define transitions for each action (move in the grid)
        next_state = self.state

        # Define transitions for each action (move in the grid)
        next_state = self.state

        # Add your code here

        # Update the state if the next state is not an obstacle
        if next_state not in self.obstacles:
            self.state = next_state
        else:
            return self.state, -1, False  # Negative reward for hitting an obstacle

        # Define the reward function (e.g., reaching a goal state)
        done = (self.state == self.state_space - 1)  # Goal state
        reward = 1 if done else 0

        return self.state, reward, done

    def print(self):
        
        """ IMPLEMENT THE PRINT METHOD HERE """




class IcyGridWorldEnvironment(GridWorldEnvironment):
    def __init__(self, grid_size=(5, 5), obstacles=[]):
        super(GridWorldEnvironment, self).__init__()
        self.grid_size = grid_size
        self.state_space = np.prod(grid_size)
        self.action_space = 4  # Four possible actions: Up, Down, Left, Right
        self.state = None
        self.obstacles = obstacles
        self.set_mdp()

    def set_mdp(self):

        """ DEFINE THE ICY-STOCHASTIC MDP HERE """