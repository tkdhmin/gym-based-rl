import pandas as pd
import numpy as np
import itertools
from collections import defaultdict

class SARSASolver():
    def __init__(self, knapsack_budget, actions):
        self.knapsack_budget = knapsack_budget
        self.alpha: float = 0.85
        self.gamma: float = 0.90
        self.epsilon: float = 0.8
        self.actions = actions
        self.q_tbl = defaultdict(int)
        self.terminated: bool = False

