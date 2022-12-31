from clearbox.rl.environments import Agent
from .ttt import TicTacToe

import numpy as np
import random

"""
Implementations of some basic agents for tic tac toe
"""

VALUES = (0, 1, -1)

class RandomLegal(Agent):
    """
    Agent that always chooses a legal move when possible, but otherwise plays fully randomly.
    """

    def move(self, state):
        res = random.choice(TicTacToe.legalmoves(state))
        return res

class Minimax(Agent):
    """
    Minimax agent. Always plays the optimal strategy.

    Simple full-depth, depth-first search.

    -- This would be faster with alpha beta pruning, but this is the simplest possible implementation.
    """

    def move(self, state):

        moves, values = self.compute_values(state, player=1)

        mx = max(values)
        idx = values.index(mx)

        return moves[idx]

    def maximizing(self, player):
        """
        Whether the current player is maximizing.

        :param player:
        :return:
        """
        return player == 1

    def compute_values(self, state, player, d=0):
        moves = TicTacToe.legalmoves(state)
        random.shuffle(moves)
        values = []

        for move in moves:

            nextstate = TicTacToe.successor(state, move, player)

            if TicTacToe.is_finished(nextstate):

                winner = TicTacToe.check_winner(nextstate)

                values.append(VALUES[winner])

            else:
                nmoves, nvalues = self.compute_values(nextstate, other(player), d=d+1)

                if self.maximizing(other(player)):
                    values.append(max(nvalues)) # the value of nextstate is the maximum we can guarantee from that state
                else:
                    values.append(min(nvalues))

        return moves, values

class Rollouts(Agent):
    """
    Agent which plays by averaging values over random rollouts against a given opponent.

    """

    def __init__(self, num_rollouts=5_000, base_agent1=None, base_agent2=None):
        self.nr = num_rollouts
        self.base_agent1 = RandomLegal() if base_agent1 is None else base_agent1
        self.base_agent2 = RandomLegal() if base_agent2 is None else base_agent2

    def move(self, state):

        moves = TicTacToe.legalmoves(state)
        values = []
        for move in moves:
            sum = 0
            for _ in range(self.nr):
                sum += self.rollout(state) # rollout from this state with us as the first move

            values.append(sum / self.nr)

        # print(moves)
        # print(values)
        # print(np.argmax(values))

        return moves[np.argmax(values)]

    def rollout(self, state, player=1):
        """
        Plays a random rollout from the current state and returns the value.

        :param state: The starting state
        :param player: The player who makes the first move in this state
        :return:
        """

        agents = (None, self.base_agent1, self.base_agent2)

        while not TicTacToe.is_finished(state):

            move = agents[player].move(state)
            state = TicTacToe.successor(state, move, player)

            player = other(player)

        winner = TicTacToe.check_winner(state)

        return VALUES[winner]

def other(player):
    return 1 if player == 2 else 2
