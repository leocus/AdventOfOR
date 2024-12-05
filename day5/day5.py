import pulp


def read_file(fname):
    with open(fname) as file:
        n_wh, n_c = list(map(int, file.readline().strip().split(' ')))
        clients_demands = []
        wh_capacity = []
        wh_cost = []
        wh_c_cost = []
        for i in range(n_wh):
            cap, cost = list(map(float, file.readline().strip().split(' ')))
            wh_capacity.append(cap)
            wh_cost.append(cost)
        for i in range(n_c):
            clients_demands += list(map(float, file.readline().strip().split(' ')))
            wh_c_cost.append(list(map(float, file.readline().strip().split(' '))))
        
        return n_wh, n_c, wh_capacity, wh_cost, clients_demands, wh_c_cost

n_wh, n_c, wh_capacity, wh_cost, clients_demands, wh_c_cost = read_file('./instance_refactored.txt')


# Minimization model
model = pulp.LpProblem("WarehousesPositioning", pulp.LpMinimize)

# Decision variables
wh_used = pulp.LpVariable.dicts("warehouse_used", (u for u in range(n_wh)), cat=pulp.LpBinary)
serve_matrix = pulp.LpVariable.dicts(
    "load_distribution",
    ((w, c) for w in range(n_wh) for c in range(n_c)),
    lowBound=0
)

# Minimize cost of warehouses
model += pulp.lpSum(sum((c * u) for c, u in zip(wh_cost, wh_used)))

# Minimize distributions costs
model += pulp.lpSum(wh_c_cost[c][w] * serve_matrix[w,c] for w in range(n_wh) for c in range(n_c))

# Constraint: Warehouse Capacity
for w in range(n_wh):
    model += pulp.lpSum(serve_matrix[w,c] for c in range(n_c)) <= wh_capacity[w] * wh_used[w]

# Constraint: Meet clients' needs
for c in range(n_c):
    model += pulp.lpSum(serve_matrix[w,c] for w in range(n_wh)) == clients_demands[c]

model.solve(pulp.PULP_CBC_CMD(msg=True))
if pulp.LpStatus[model.status] == 'Optimal':
    print([wh_used[w] for w in range(n_wh)])
    # print(serve_matrix)
