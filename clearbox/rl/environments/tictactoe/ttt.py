from clearbox.rl.environments import Environment, Agent

from abc import ABC
import random

import numpy as np

def invert(board):
    """
    invert a board (turn 1s into 2s and 2s into 1s)
    :param board:
    :return:
    """
    board = board.copy()

    board[board == 1] = -1
    board[board == 2] = 1
    board[board == -1] = 2

    return board

class TicTacToe(Environment):
    """
    Environment for the basic game of tic-tac-toe. The environment player is referred to as "the opponent" and the player
    playing against the environment is referred to as "the agent".

    The state is a (3, 3) integer matrix with 0s representing empty squares, 1 representing moves the agent has made,
    and 2 representing moves the opponent has made. In the state presented to a player, the player is always player 1.
    That is, we invert the board before presenting it to the opponent.

    Illegal moves are allowed by the environment, but result in immediate loss of the game. This allwos us to train
    agents that learn the rules of the game as well as the gameplay.
    """

    @staticmethod
    def legalmoves(board):
        """
        Returns a list of legal moves (all the empty places on the board).

        :param board:
        :return:
        """
        x, y = np.where(board == 0)
        x, y = list(x), list(y)
        return list(zip(x, y))

    @staticmethod
    def successor(state, move, player=1, check=False):
        """
        Returns the successor state, together with the reward if the given player makes the given
        move in the given state.
        :return:
        """
        if check:
            raise #TODO

        state = state.copy()
        state[move] = player

        return state

    @staticmethod
    def is_finished(board):

        w = TicTacToe.check_winner(board)

        if w != 0:
            return True

        return not np.any(board == 0)

    @staticmethod
    def check_winner(board):
        """
        Check whether anybody has currently won.
        :return: 0 if no winner, other winner's id.
        """

        # check horizontal
        for row in range(3):
            if board[row, 0] == board[row, 1] == board[row, 2] != 0:
                return board[row, 0]

        # check vertical
        for column in range(3):
            if board[0, column] == board[1, column] == board[2, column] != 0:
                return board[0, column]

        # check diagonals
        if board[0, 0] == board[1, 1] == board[2, 2] != 0:
            return board[1, 1]

        if board[0, 2] == board[1, 1] == board[2, 0] != 0:
            return board[1, 1]

        return 0

    def __init__(self, opponent : Agent, agent_starts=None):

        if agent_starts is None:
            agent_starts = random.choice([True, False])

        self.opponent = opponent
        self._dead = False # the board is dead if a player has made an illegal move

        self.board = np.full(fill_value=0, shape=(3, 3), dtype=np.int)

        if not agent_starts:
            move = self.opponent.move(invert(self.board))

            self.board[move] = 2

    def winner(self):
        if not self.finished():
            return None

        if self._dead: # game ended by illegal move
            return 1 if self._dead == 2 else 2

        return self.check_winner(self.board)

    def illegal_move(self):
        """
        True if the game has ended because of an illegal move
        :return:
        """
        return bool(self._dead)

    def act(self, action):

        assert len(action) == 2, f'{action=}'

        if not self._is_legal(action): # illegal move results in loss of the game
            self._dead = 1
            return -1

        self.board[action] = 1

        w = self.winner()
        if w is not None:
            return [0, 1, -1][w]

        # Opponent move
        action = self.opponent.move(invert(self.board))

        if not self._is_legal(action): # illegal move results in loss of the game
            self._dead = 2
            return 1

        self.board[action] = 2

        w = self.winner()
        if w is not None:
            return [0, 1, -1][w]

        return 0 # Game still in progress, or draw, no reward

    def finished(self):
        if bool(self._dead):
            return True

        return self.is_finished(self.board)

    def _is_legal(self, action):
        """
        Check if the given action is a legal move in the current state. Should not be called by agents if the aim is to
        learn the rules and the gameplay.

        :param action:
        :return:
        """

        assert len(action) == 2

        if 0 > action[0] or action[0] > 2:
            return False

        if 0 > action[1] or action[1] > 2:
            return False

        if self.board[action] != 0:
            return False

        return True

    def state(self):
        return self.board

    def full_state(self):
        return self.state()