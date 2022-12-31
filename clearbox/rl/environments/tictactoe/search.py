import random

from tqdm import trange
from .nn_agents import MLPAgent
from .agents import RandomLegal

from .ttt import TicTacToe

from clearbox.rl.environments import Agent
import torch

from fire import Fire
import pickle, os, sys

"""
Search for a good tic-tac-toe agent by random search.
"""

def sim_annealing(alpha=0.0, step_size=1.0, iterations=50_000, dir='.', agent_class=MLPAgent,
                  games_per_eval=250, opponent_prob=0.0, save_every=1_000, depth=3, hidden=128):

    opponent_store = [RandomLegal()]
    current = agent_class(depth=depth, hidden=hidden)
    current_score = float('-inf')

    for i in range(iterations):

        # random step
        parameters = [p.data.clone() for p in current.network.parameters()]
        parameters = [p + torch.randn(p.size()) * step_size for p in parameters]
        next = agent_class(depth=depth, hidden=hidden, parameters=parameters)

        # compute evaluation
        scores = evaluate(next, opponents=opponent_store + [], num_games=games_per_eval)
        next_score = sum(scores)

        if i% 10 == 0:
            print(next_score)

        if next_score > current_score or random.random() < alpha:
            current = next
            current_score = next_score

        if random.random() < opponent_prob:
            # print('new opponent')
            opponent_store.append(next)

        if (i!= 0 and i % save_every == 0) or i == iterations - 1:
            with open(dir + os.sep + f'agent.{i:07}.cbx', 'wb') as file:
                pickle.dump(current, file)

def breed(model1, model2, mutation=1e-8):
    """
    "Breeds" two pytorch models, simply by taking their average in parameter space, and adding some noise.

    :param model1:
    :param model2:
    :return:
    """
    nparms = []
    for p1, p2 in zip(model1.parameters(), model2.parameters()):

        size = p1.size()
        assert p1.size() == p2.size()

        np = p1.data * 0.5 + p2.data * 0.5 + torch.randn(*size) * mutation
        nparms.append(np)

    return nparms

def evo_search(popsize=100, mutation=0.05, iterations=50_000, dir='.', agent_class=MLPAgent,
                  games_per_eval=200, opponent_prob=0.01, save_every=100, depth=2, hidden=32):
    """
    Ad-hoc evolutionary method.

    :return:
    """

    init = 5
    opponent_store = [agent_class(depth=depth, hidden=hidden) for _ in range(popsize)]
    population = [agent_class(depth=depth, hidden=hidden) for _ in range(popsize)]

    for i in range(iterations):

        # compute evaluation
        scores = [sum(evaluate(p, opponents=opponent_store, num_games=games_per_eval)) for p in population]

        # if i% 1 == 0:
        # print(sum(scores) / len(scores))

        if random.random() < opponent_prob:
            print('new opponent')
            opponent_store.append(random.choice(population))

        if (i != 0 and i % save_every == 0) or i == iterations - 1:
            with open(dir + os.sep + f'agent.{i:07}.cbx', 'wb') as file:
                pickle.dump(population[0], file)
            print(f'Saved agent.{i:07}.cbx')

        # sort
        z = zip(population, scores)
        z = sorted(z, key=lambda x : -x[1])
        top = z[:popsize//2]
        population, scores = zip(*top)
        print(scores)

        # breed
        npop = []
        for _ in range(popsize):
            mother, father = random.choice(population), random.choice(population)
            newparms = breed(mother.network, father.network, mutation=mutation)
            npop.append(MLPAgent(depth=depth, hidden=hidden, parameters=newparms))

        population = npop

def evaluate(player : Agent, opponents, num_games):
    """
    Plays a number of games against randomly drawn opponents.

    :param player:
    :param opponents:
    :param num_games:
    :return:
    """

    scores = []
    for _ in range(num_games):

        opponent = random.choice(opponents)
        env = TicTacToe(opponent=opponent)

        reward = 0
        while not env.finished():
            move = player.move(env.state())
            reward += env.act(move)

        scores.append(reward)

    return scores