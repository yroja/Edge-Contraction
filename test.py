import random
from typing import List, Tuple
import greedy_joining_spanning_forest

import greedy_joining_spanning_forest2

def generate_random_graphs(num_graphs: int, num_nodes: int, edge_prob: float = 0.5) -> List[Tuple[List[Tuple[int, int]], List[float]]]:
    """
    Erzeugt eine Liste zufälliger, ungerichteter Graphen mit Gewichtungen.
    
    Args:
        num_graphs (int): Anzahl der zu erzeugenden Graphen.
        num_nodes (int): Anzahl der Knoten pro Graph.
        edge_prob (float): Wahrscheinlichkeit für Existenz einer Kante zwischen zwei Knoten.

    Returns:
        List[Tuple[List[Tuple[int, int]], List[float]]]: 
            Liste von Graphen, wobei jeder Graph ein Tupel aus:
            - Kantenliste [(u, v), ...]
            - Kostenliste [c1, c2, ...]
            ist.
    """
    graphs = []
    for _ in range(num_graphs):
        edges = []
        costs = []
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):  # ungerichteter Graph, also nur (u, v) mit u < v
                if random.random() < edge_prob:
                    edges.append((u, v))
                    costs.append(round(random.uniform(-1.0, 1.0), 3))  # z.B. Kosten im Bereich [-1, 1]
        graphs.append((edges, costs))
    return graphs

if __name__ == '__main__':
    amount_graph = 1
    n = 5
    tau = 1
    graphs = generate_random_graphs(amount_graph, n, 0.5)

    for graph in graphs:
        value_of_cut, labeling = greedy_joining_spanning_forest.greedy_joining_spanning_forest(n, graph[0], graph[1], tau)
        
        value_of_cut2, labeling2 = greedy_joining_spanning_forest2.greedy_joining_spanning_forest2(n, graph[0], graph[1], tau)
        print(f'cost: {value_of_cut2}\nlabeling: {labeling2}')
