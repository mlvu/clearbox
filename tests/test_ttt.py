import unittest

import clearbox as cb
import numpy as np

from clearbox.rl.environments.tictactoe import Minimax

import torch

class TestTTT(unittest.TestCase):

    def test_mm(self):

        agent = Minimax()
        board = np.asarray([
            [1, 2, 0],
            [0, 2, 0],
            [0, 0, 0]
        ])

        print(agent.move(board))


