import networkx as nx

from logic.foc import *

# available = {
#     "RED": 0,
#     "BLUE": 1,
#     "GREEN": 2,
#     "BLACK": 3
# }

a0 = Property("RED", "x")

a1 = Property("BLUE", "y")
a2 = NEG(Role(relation="EDGE", variable1="x", variable2="y"))
a3 = AND(a1, a2)

# a2 = Property("BLUE", "y")
# a2 = Property("RED", "x")
# a3 = Role(relation="EDGE", variable1="x", variable2="y")

a4 = Exist(variable="y", expression=a3, lower=1, upper=1)
a5 = AND(a0, a4)

G = nx.path_graph(5)

nx.set_node_attributes(
    G, dict(zip(G, [1, 0, 1, 0, 0])), name="color")


# print(G.nodes(data=True))

formula = FOC(a5)
print(formula)
print(formula(G, variable="x"))
