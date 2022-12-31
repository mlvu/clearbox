"""
Command line tools for various parts of the clearbox library
"""
import clearbox.rl
import sys

def main():

    args = sys.argv
    # print(args)

    if args[1].lower().strip() == 'rl':

        clearbox.rl.play(args[2:])

    else:
        raise Exception(f'Command not recognized: {args[1]}')

