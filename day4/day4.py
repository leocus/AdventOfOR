import numpy as np
import pandas as pd
from tqdm import trange
from copy import deepcopy
from queue import PriorityQueue


X = np.loadtxt('./day4_instance_refactored.txt')
rooms = 4
classes = 4
teachers = 4
X = X.reshape(rooms, classes, teachers).astype(np.int32)
max_steps = 4 * np.sum(X)


meetings = []
for room in range(rooms):
    for cl in range(classes):
        for t in range(teachers):
            for _ in range(X[room, cl, t]):
                meetings.append((room, cl, t))

# Now, we need to assign meetings to slots
# meetings = sorted(meetings, key=lambda x: x[0])
trials = 10000
best_efficiency = 0
best_timetable = None

for i in trange(trials):
    meetings = list(np.random.permutation(meetings))

    slots = []
    while len(meetings) > 0:
        slot = [None] * rooms
        indices = []
        for i in range(len(meetings)):
            tmp_room = meetings[i][0]
            if slot[tmp_room] is None:
                free = True
                # Check if teacher and class are free
                for assignment in slot:
                    if assignment is not None:
                        if assignment[1] == meetings[i][1] or assignment[2] == meetings[i][2]:
                            free = False
                if free:
                    slot[tmp_room] = meetings[i]
                    indices.append(i)

        for i in sorted(indices, reverse=True):
            del meetings[i]

        slots.append(slot)

    timetable = {f'Room{i}': [] for i in range(rooms)}
    timetable['Slot'] = []

    for i, slot in enumerate(slots):
        timetable['Slot'].append(i)
        for j, assignment in enumerate(slot):
            assert assignment is None or j == assignment[0], (j, assignment[0])
            if assignment is not None:
                timetable[f'Room{assignment[0]}'].append(f'Class {assignment[1]}/Teacher {assignment[2]}')
            else:
                timetable[f'Room{j}'].append(f'Empty')

    timetable = pd.DataFrame(timetable)
    efficiency = 1-(np.sum((timetable == 'Empty').values)/(len(timetable)*rooms))

    if efficiency > best_efficiency:
        best_timetable = timetable
        best_efficiency = efficiency

print(best_timetable)
print("Greedy - Efficiency:", best_efficiency)
