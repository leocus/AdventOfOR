#!/usr/bin/env python3
import numpy as np
from tqdm import trange
from itertools import permutations
from collections import defaultdict


def check_solution(solution: list, constraints: dict):
    """
    Check quality of a solution.
    It returns (max n. rooms, n. slots, n. constraints violated), otherwise it

    :solution: 2d list. Each outer index is a day/timestep, each inner index is a room
    """
    violations = 1
    n_rooms = 0
    for events in solution:
        for i, event in enumerate(events):
            for event2 in events[i+1:]:
                if event2 in constraints[event]:
                    violations += 1
        n_rooms = max(n_rooms, len(events))

    return n_rooms, len(solution), violations


print('Loading file')
cnst = np.loadtxt('./day1_instance_refactored.txt')
print('Computing constraints')
constraints = defaultdict(list)
for a, b in cnst:
    constraints[a].append(b)
    constraints[b].append(a)


BUDGET = 10000

print('Approach 1: Trivial Solution')
indices = [[x + 1] for x in range(len(constraints))]
n_rooms, n_slots, violations = check_solution(indices, constraints)
print(f'The trivial solution needs {n_rooms} rooms for {n_slots} slots. It violates {violations} constraints.')


print('Approach 2: Random Search')
best = float('inf')
argbest = None
indices = [x + 1 for x in range(len(constraints))]
for i in trange(BUDGET, desc='Random Search'):
    candidate = []
    perm = np.random.permutation(indices)
    
    added = 0
    while added < len(perm):
        n_rooms = 1 if (len(perm) - added) == 1 else np.random.randint(1, len(perm) - added)
        candidate.append(perm[added:added+n_rooms])
        added += n_rooms

    n_rooms, n_slots, violations = check_solution(candidate, constraints)
    if violations == 0:
        if n_rooms < best:
            best = n_rooms
            argbest = candidate
print(f'Best solution found by random search needs {best} rooms')

print('Approach 3: Valid Random Search')
best = float('inf')
argbest = None
indices = [x + 1 for x in range(len(constraints))]
for i in trange(BUDGET, desc='Random Search'):
    candidate = []
    perm = np.random.permutation(indices)
    
    added = 0
    while added < len(perm):
        n_rooms = 1 if (len(perm) - added) == 1 else np.random.randint(1, len(perm) - added)
        new_row = perm[added:added+n_rooms]
        candidate.append(perm[added:added+n_rooms])
        added += n_rooms

    n_rooms, n_slots, violations = check_solution(candidate, constraints)
    if violations == 0:
        if n_rooms < best:
            best = n_rooms
            argbest = candidate
print(f'Best solution found by valid random search needs {best} rooms')
