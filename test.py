import random
from typing import List, Tuple
import greedy_joining_spanning_forest

import random
from typing import List, Tuple

def generate_random_graphs_with_unique_weights(
    num_graphs: int,
    num_nodes: int,
    edge_prob: float = 0.5,
    min_weight: int = -10000,
    max_weight: int = 10000
) -> List[Tuple[List[Tuple[int, int]], List[int]]]:
    """
    Erzeugt zufällige Graphen mit eindeutigem, ganzzahligem Kantengewicht.

    Args:
        num_graphs: Anzahl der zu erzeugenden Graphen.
        num_nodes: Anzahl der Knoten pro Graph.
        edge_prob: Wahrscheinlichkeit, dass eine Kante existiert.
        min_weight: Untere Grenze für Gewicht.
        max_weight: Obere Grenze für Gewicht.

    Returns:
        Liste von Tupeln (Kantenliste, Kostenliste)
    """
    graphs = []
    total_possible_edges = num_nodes * (num_nodes - 1) // 2

    if max_weight - min_weight + 1 < total_possible_edges:
        raise ValueError("Nicht genug einzigartige Gewichtswerte für mögliche Kantenanzahl.")

    for _ in range(num_graphs):
        # Liste aller möglichen Gewichte im erlaubten Bereich
        possible_weights = list(range(min_weight, max_weight + 1))
        random.shuffle(possible_weights)

        edges = []
        costs = []
        weight_index = 0  # Index in der gemischten Gewichtsliste

        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if random.random() < edge_prob:
                    edges.append((u, v))
                    costs.append(possible_weights[weight_index])
                    weight_index += 1

        graphs.append((edges, costs))

    return graphs

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    n = int(lines[0])
    edges = []
    costs = []

    for i, line in enumerate(lines[1:]):
        weights = [int(x) for x in line.split()]
        for j, w in enumerate(weights):
            if w != 0:  # Nur Kanten mit Gewicht != 0 speichern
                edges.append([i, i + j + 1])
                noise = random.uniform(-0.1, 0.1)
                costs.append(float(w))
                #costs.append(float(w) + noise)

    return n, edges, costs



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
                    costs.append(random.randint(-10, 10))  # z.B. Kosten im Bereich [-1, 1]
        graphs.append((edges, costs))
    return graphs

if __name__ == '__main__':
    amount_graph = 10000
    n = 5
    tau = 1
    #graphs = generate_random_graphs(amount_graph, n, 0.7)
    graphs = generate_random_graphs_with_unique_weights(amount_graph, n, 0.4, -10, 10)
    error = []
    vals = []
    #n, edges, costs = read_data('test.txt')

    #value_of_cut, labeling = greedy_joining_spanning_forest.greedy_joining_extended(n, edges, costs, tau, 'f')
    #value_of_cut2, labeling2 = greedy_joining_spanning_forest.greedy_joining_extended(n, edges, costs, tau, 'm')
    #value_of_cut3, labeling3 = greedy_joining_spanning_forest.greedy_joining(n, edges, costs)

    #print(f'value of cut 1 (our): {value_of_cut}, value of cut 2: {value_of_cut2}, value of cut 3: {value_of_cut3}\nlabeling1: {labeling}, labeling2: {labeling2}, labeling 3: {labeling3}')
    
    for graph in graphs:
        #value_of_cut, labeling = greedy_joining_spanning_forest.greedy_joining_spanning_forest(n, graph[0], graph[1], tau)
        
        #value_of_cut2, labeling2 = greedy_joining_spanning_forest.greedy_joining_spanning_forest2(n, graph[0], graph[1], tau)
        value_of_cut_f, labeling_f = greedy_joining_spanning_forest.greedy_joining_extended(n, graph[0], graph[1], tau, 'f')
        value_of_cut_m, labeling_m = greedy_joining_spanning_forest.greedy_joining_extended(n, graph[0], graph[1], tau, 'm')
        value_of_cut_gaec, labeling_gaec = greedy_joining_spanning_forest.greedy_joining(n, graph[0], graph[1])



        if value_of_cut_f != value_of_cut_m and value_of_cut_f != value_of_cut_gaec and value_of_cut_m != value_of_cut_gaec:
            #print('************')
            error.append(graph)
            vals.append((value_of_cut_f, value_of_cut_m, value_of_cut_gaec))
            #print(graph)
            
            #print(f'(our) cost: {value_of_cut}\nlabeling: {labeling}')
        
            #print(f'cost2: {value_of_cut2}\nlabeling2: {labeling2}')


    for i in range(len(error)):
        print('**********')
        print(f'graph: {error[i]}')
        print(f'objective_f: {vals[i][0]}, objective_m: {vals[i][1]}, objective_g: {vals[i][2]}')            

    if(len(error) == 0):
        print('no missmatches found')
   
    
    
        
    

