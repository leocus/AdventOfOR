import numpy as np
from tqdm import trange


def load_file(fname):
    costs = []
    n = None
    tmp = []
    with open(fname) as file:
        n = int(file.readline())

        for line in file:
            tmp += list(map(int, line.strip().split(' ')))

            if len(tmp) == n:
                costs.append(tmp)
                tmp = []
    return costs


def greedy(X, order=None):
    assignment = {}
    total_cost = 0
    if order is None:
        order = [*range(X.shape[0])]
    for t in order:
        i = np.argmin(X[t])
        total_cost += X[t, i]
        assignment[t] = i
        X[:, i] = np.inf
        X[t, :] = np.inf

    assert len(set(assignment.values())) == X.shape[0], f"Incorrect assignments ({len(set(assignment.values()))} are valid)\n{assignment}"
    return assignment, total_cost


def main():
    X = load_file('./day3_instance_refactored.txt')
    X = np.array(X).astype(np.float16)
    bak = np.array(X)
    print("Cost matrix has shape:", X.shape)

    #####################
    #  Greedy solution  #
    #####################

    assignment, total_cost = greedy(X)
    print(f"Greedy finds an assignment with cost {total_cost}.")

    ##################################
    #  Greedy with Randomized order  #
    ##################################

    BUDGET = 10000000
    indices = [*range(X.shape[0])]
    best_cost = np.inf
    best_sol = []
    for i in trange(BUDGET+1, desc='Randomized greedy'):
        X = bak.copy()
        order = np.random.permutation(indices)
        assignment, total_cost = greedy(X, order)
        if total_cost < best_cost:
            best_cost = total_cost
            best_sol = assignment
        if np.log10(i) % 1.0 == 0.0:
            print(f'Iteration: {i}; Best cost: {best_cost}')
    print(f"Randomized Greedy finds an assignment with cost {best_cost}.")


    # # ###################################
    # # #  Random Mutation Hill Climbing  #
    # # ###################################

    # X = bak.copy()
    # RMHCB = int(1e7)
    # for i in trange(RMHCB, desc='RMHC'):
    #     p1 = np.random.randint(0, len(best_sol))
    #     p2 = np.random.randint(0, len(best_sol))

    #     new_fit = best_cost - X[p1, best_sol[p1]] - X[p2, best_sol[p2]]
    #     new_fit += X[p2, best_sol[p1]] + X[p1, best_sol[p2]]

    #     if new_fit < best_cost:
    #         best_cost = new_fit
    #         best_sol[p1], best_sol[p2] = best_sol[p2], best_sol[p1]
    #         print(f"Found new best: {best_cost}")

    #######################################################
    #  Simulated Annealing Random Mutation Hill Climbing  #
    #######################################################

    # X = bak.copy()
    # SAB = int(1e7)
    # for i in trange(SAB, desc='SA-RMHC'):
    #     p1 = np.random.randint(0, len(best_sol))
    #     p2 = np.random.randint(0, len(best_sol))
        
    #     T = np.sqrt(2)*0.5/np.sqrt(i+1)

    #     new_fit = best_cost - X[p1, best_sol[p1]] - X[p2, best_sol[p2]]
    #     new_fit += X[p2, best_sol[p1]] + X[p1, best_sol[p2]]

    #     if np.random.uniform() > (new_fit - best_cost)/best_cost/T:
    #         best_cost = new_fit
    #         best_sol[p1], best_sol[p2] = best_sol[p2], best_sol[p1]
    #         print(f"Found new best: {best_cost}; T={T}")

    #     if new_fit < 0:
    #         breakpoint()
    # print(best_sol)


    # ######################
    # #  Branch and Bound  #
    # ######################

    # X = bak.copy()
    # queue = [([], 0)]
    # best_known_value = 270000
    # best_solution = None

    # iteration = 0
    # while len(queue) > 0:
    #     if iteration % 1000 == 0:
    #         print(f'Iteration {iteration}; length of the queue: {len(queue)}; Best known value: {best_known_value}.\n{queue[0]}')
    #     node, cur_cost = queue.pop(0)

    #     # Get index
    #     var_i = len(node)

    #     if len(node) == X.shape[1]:
    #         if cur_cost < best_known_value:
    #             best_known_value = cur_cost
    #             best_solution = node
    #             print(f"New best value: {best_known_value}\n{best_solution}")
    #     # Expand (branch)
    #     tmp = len(queue)
    #     tmp_remaining = X[var_i + 1:]
    #     for i in range(X.shape[1]):
    #         if i not in node:
    #             remaining = tmp_remaining[:, [j for j in range(X.shape[1]) if j not in node and j != i]]
    #             if remaining.shape[0] > 0 and remaining.min(1).sum() + cur_cost + X[var_i, i] < best_known_value and \
    #                 remaining.min(0).sum() + cur_cost + X[var_i, i] < best_known_value:
    #                 queue.insert(0, (node + [i], cur_cost + X[var_i, i]))
    #             else:
    #                 if remaining.shape[0] == 0:
    #                     queue.insert(0, (node + [i], cur_cost + X[var_i, i]))
    #     assert len(queue) - tmp <= X.shape[1], (len(queue), tmp)
        
    #     iteration += 1

    # print(f'Branch and bound finds an assignment with cost {best_known_value}')

    # ###################
    # #  Random Search  #
    # ###################
    
    # indices = [*range(X.shape[0])]
    # best_cost = np.inf
    # best_sol = []
    # for i in trange(100000000+1, desc='Random search'):
    #     X = bak.copy()
    #     order = np.random.permutation(indices)
    #     total_cost = X[indices, order].sum()
    #     if total_cost < best_cost:
    #         best_cost = total_cost
    #         print(f'Iteration: {i}; Best cost: {best_cost}')
    # print(f"Random Search finds an assignment with cost {best_cost}.")


if __name__ == '__main__':
    main()

