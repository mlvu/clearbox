from .agents import RandomLegal, Minimax, Rollouts
from .nn_agents import MLPAgent
from .ttt import TicTacToe

from clearbox.rl.environments import Agent

from colorama import Fore, Back, Style
import pickle

LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

def play(opponent='random'):
    """
    Start a command-line game against an opponent
    :return:
    """
    print(f'Playing tic-tac-toe against opponent: {opponent}.')

    opponent = load_opponent(opponent)
    env = TicTacToe(opponent)

    while not env.finished():

        moves = TicTacToe.legalmoves(env.state())

        print('Current board state:')
        print_board(env.state())

        move = None
        while move is None:
            rmove = input("Your move?")
            move = process(rmove, moves)

        reward = env.act(move)
        print(f'-- made move {move}, received reward {reward}')

    print('Final board state:')
    print_board(env.state())

    if env.illegal_move():
        print('Illegal move made.')

    w = env.winner()
    if w == 0 or w is None:
        print('Game ended in a draw')
    elif w == 1:
        print('You won.')
    elif w == 2:
        print('You lost.')

def process(rmove, moves):
    rmove = rmove.lower().strip()

    if rmove not in LETTERS[:len(moves)]:
        return None

    return moves[LETTERS.index(rmove)]

def print_board(board):

    moves = TicTacToe.legalmoves(board)

    for r in range(3):
        for c in range(3):
            if (r, c) in moves:
                chr = Fore.LIGHTWHITE_EX + LETTERS[moves.index((r, c))]
            else:
                chr = Fore.BLUE+'o' if (board[r, c] == 1) else Fore.RED+'x'

            print(chr + ' ', end='')
        print(Fore.BLACK)

def load_opponent(opponent):

    if type(opponent) == str:

        if opponent.lower().strip() == 'random':
            return RandomLegal()

        if opponent.lower().strip() == 'minimax':
            return Minimax()

        if opponent.lower().strip() == 'mlp':
            return MLPAgent()

        if opponent.lower().strip() == 'rollout':
            return Rollouts()

        if opponent.startswith('file:'):
            # Load a pickled opponent to play against
            with open(opponent[5:], 'rb') as file:
                return pickle.load(file)



    if type(opponent) == Agent:
        return opponent()

    return opponent

