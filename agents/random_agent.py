class RandomAgent:
    """
    Baseline aléatoire: choisit uniformément parmi les actions disponibles.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()
