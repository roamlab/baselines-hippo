from .path import Path
import random

class ReplayBuffer:

    def __init__(self, capacity=0):
        self.capacity = capacity
        self.fill = 0
        self.paths = []
        # Note: capacity is measured in terms of number of transtions (not paths)

    def insert(self, path):
        self.paths.append(path)
        # prune if overflow occurs
        self.fill += len(path)
        while self.fill > self.capacity:
            path = self.paths.pop(1)
            self.fill -= len(path)

    def sample(self):
        """ randomly sample a path from the buffer """
        return random.choice(self.paths)

    def prune(self, min_iteration=None):
        for i , path in enumerate(self.paths):
            if path.iteration <= min_iteration:
                self.paths.remove(i)
