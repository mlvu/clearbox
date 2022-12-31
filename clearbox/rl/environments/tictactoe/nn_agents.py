import torch
from torch import nn
import torch.nn.functional as F

from clearbox.rl.environments import Agent
from .ttt import TicTacToe

import numpy as np
import sys

"""
Neural network based agents. Uses torch for the time being.
"""

class MLPAgent(Agent):
    """
    Agent that uses a basic feedforward architecture.
    """

    def __init__(self, depth=1, hidden=64, parameters=None):

        modules = [nn.Linear(3*3*3, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            modules.append(nn.Linear(hidden, hidden))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden, 3*3))

        self.network = nn.Sequential(*modules)

        if parameters is not None:
            for source, target in zip(parameters, self.network.parameters()):
                target.data = source

    def move(self, state):

        # translate state to one-hot input
        inp = F.one_hot(torch.from_numpy(state), num_classes=3).to(torch.float)

        # forward pass
        out = self.network(inp.reshape(1, -1))

        # translate output to move
        out = out.reshape(3, 3)
        return np.unravel_index(out.argmax(), (3, 3))

    def parameters(self):

        return self.network.parameters()
