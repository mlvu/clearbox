import clearbox as cb
from clearbox.rl.environments import tictactoe as ttt

from colorama import init

def play(args):

    init()
    command = args[0].lower().strip()

    if command == 'ttt':
        ttt.play(*args[1:])
    else:
        raise Exception(f'Command not recognized {command}')
