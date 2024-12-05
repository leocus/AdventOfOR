import numpy as np
import pandas as pd
from copy import deepcopy
from joblib import Parallel, delayed

X = np.loadtxt('./day4_instance_refactored.txt').reshape(4, 4, 4)

class Individual(object):
    """docstring for Individual"""
    def __init__(self, n_slots):
        super(Individual, self).__init__()
        self._n_slots = n_slots

        self.init()

    def init(self):
        self.M = np.random.choice([0, 1], size=(self._n_slots, 4, 4, 4), p=[0.999, 0.001])

    def mutate(self):
        point = np.random.randint(0, self._n_slots)
        r = np.random.randint(0, 4)
        if np.random.uniform() < 0.3:
            # Bitflip
            c = np.random.randint(0, 4)
            t = np.random.randint(0, 4)

            self.M[point, r, c, t] = abs(1 - self.M[point, r, c, t])
        elif np.random.uniform() < 0.5:
            # Row swap
            c = np.random.randint(0, 4)
            t1 = np.random.randint(0, 4)
            t2 = np.random.randint(0, 4)

            tmp = self.M[point, r, c, t1]
            self.M[point, r, c, t1] = self.M[point, r, c, t2]
            self.M[point, r, c, t2] = tmp
        else:
            # Col swap
            t = np.random.randint(0, 4)
            c1 = np.random.randint(0, 4)
            c2 = np.random.randint(0, 4)

            tmp = self.M[point, r, c1, t]
            self.M[point, r, c1, t] = self.M[point, r, c2, t]
            self.M[point, r, c2, t] = tmp

    def clone(self):
        return deepcopy(self)

    def crossover(self, other):
        "One point"
        cp = np.random.randint(self.M.shape[0])

        self.M[cp:] = other.M[cp:].copy()

    def repair(self):
        idx = np.where(self.M.reshape(self.M.shape[0], 4, -1).sum(-1) > 1)
        self.M[idx[0], idx[1], ...] = 0
        idx = np.where(self.M.sum(1).sum(2) > 1)
        self.M[idx[0], :, idx[1]] = 0
        idx = np.where(self.M.sum(2).sum(1) > 1)
        self.M[idx[0], :, :, idx[1]] = 0


def tournament_selection(fitnesses, num_to_select, tournament_size):
    """
    Perform tournament selection on a population.

    Parameters:
    fitnesses (numpy array): The fitnesses of the individuals in the population.
    num_to_select (int): The number of individuals to select.
    tournament_size (int): The number of individuals to consider in each tournament.

    Returns:
    selected_indices (numpy array): The indices of the selected individuals.
    """
    fitnesses = np.array(fitnesses)
    num_individuals = len(fitnesses)
    # Create a 2D array where each row is a tournament
    tournaments = np.random.choice(num_individuals, size=(num_to_select, tournament_size), replace=True)
    # Get the fitnesses of the individuals in each tournament
    tournament_fitnesses = fitnesses[tournaments]
    # Get the indices of the winners in each tournament
    winner_indices = np.argmin(tournament_fitnesses, axis=1)
    # Get the indices of the selected individuals
    selected_indices = tournaments[np.arange(num_to_select), winner_indices]

    fitnesses = list(fitnesses)
    selected_indices = list(selected_indices)

    return selected_indices


def fitness(genotype):
    ind = genotype.M
    F = np.sum(ind, 0)
    if np.min(X - F) < 0:
        return 1000
    
    closeness = np.sum(X - F)/X.sum()
    length = ind.reshape(ind.shape[0], -1).sum(1)
    return closeness


popsize = 100
generations = 10000
slots = 30

pop = [Individual(slots) for _ in range(popsize)]
[i.repair() for i in pop]
fitnesses = Parallel(-1)(delayed(fitness)(ind) for ind in pop)

for g in range(generations):
    indices = tournament_selection(fitnesses, popsize, 5)
    pop = [pop[i] for i in indices]
    fitnesses = [fitnesses[i] for i in indices]
    off = [pop[i].clone() for i in range(len(pop))]
    [o.crossover(np.random.choice(off)) for o in off]
    [o.mutate() for o in off]
    [o.repair() for o in off]
    off_f = Parallel(-1)(delayed(fitness)(ind) for ind in off)

    pop += off
    fitnesses += off_f

    print(g, *map(lambda x: x(fitnesses), (np.min, np.mean, np.max)))

best = np.argmin(fitnesses)
x = pop[best].M

df = {f'Room{i}': [] for i in range(4)}
df['Slot'] = []

for s, slot in enumerate(x):
    df['Slot'].append(s)
    for r, room in enumerate(slot):
        idx = np.where(room == 1)
        if len(idx[0]) > 0:
            df[f'Room{r}'].append(f'Class{idx[0]}/Teacher{idx[1]}')
        else:
            df[f'Room{r}'].append('-')

df = pd.DataFrame(df)
df.to_csv('solution.csv')
breakpoint()
print()
