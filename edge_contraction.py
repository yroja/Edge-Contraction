import torch
import numpy as np
import networkx as nx
import time
from partition import Partition
from alg import mst_without_conflicts
from maximum_matching import luby_jones_handshake
from maximum_forest import find_maximum_spanning_forest_without_conflicts



def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    n = int(lines[0])

    adj_matrix = np.zeros((n, n))

    for i, line in enumerate(lines[1:]):
        w = [int(x) for x in line.split()]
        adj_matrix[i, i+1:] = w
    
    adj_matrix += adj_matrix.T

    return torch.as_tensor(adj_matrix, dtype=torch.float64, device=device)

def read_data2(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    
    edges = []
    costs = []
    nodes = set()

    for line in lines[1:]:
        vals = [float(x) for x in line.split()]
        edges.append([int(vals[0]), int(vals[1])])
        costs.append(vals[2])
        nodes.add(vals[0])
        nodes.add(vals[1])
    
    n = len(nodes)

    indices = torch.tensor(edges, dtype=torch.int64).t()  # (2, num_edges)
    values = torch.tensor(costs, dtype=torch.float32)
    adj_sparse = torch.sparse_coo_tensor(indices, values, size=(n, n))

    # Symmetriere durch Addition der Transponierten
    adj_sym = adj_sparse + adj_sparse.transpose(0, 1)

    return adj_sym



def generate_noise_matrix(n, seed, epsilon):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.empty((n, n), dtype=torch.float64).uniform_(-epsilon, epsilon, generator=generator)
    noise = torch.triu(noise, diagonal=1)
    noise = noise + noise.T
    noise.fill_diagonal_(0)
    return noise

def generate_noise_vector(n, seed, epsilon):
    """
    Erzeugt für alle n*(n-1)/2 Paare (i<j) eine uniforme Noise in [-epsilon, +epsilon],
    deterministisch über `seed`.
    Returns:
      idx_pairs: List of (i,j) in lexicographischer Ordnung
      noise_vals: Tensor of shape (m,) mit dtype=torch.float64
    """
    # 1) Index-Paare in definierter Reihenfolge
    idx_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(idx_pairs)
    # 2) Generator initialisieren
    gen = torch.Generator().manual_seed(seed)
    # 3) Vektor erzeugen
    noise_vals = torch.empty(m, dtype=torch.float64).uniform_(-epsilon, epsilon, generator=gen)
    #noise_vals = torch.round(noise_vals * 10000) / 10000
    return idx_pairs, noise_vals

def add_noise_adj(adj_matrix, seed):
    adj = adj_matrix.to(torch.float64).clone()
    n = adj.shape[0]
    # Anzahl Kanten für epsilon
    num_edges = torch.count_nonzero(adj).item() // 2
    epsilon = 1 / num_edges if num_edges > 0 else 0

    idx_pairs, noise_vals = generate_noise_vector(n, seed, epsilon)         #np.argwhere(adj != 0)

    # Noise ins Matrixformat füllen
    for k, (i, j) in enumerate(idx_pairs):
        val = noise_vals[k].item()
        adj[i, j] += val
        adj[j, i] += val

    adj.fill_diagonal_(0)
    return adj

def add_noise(adj_matrix, seed):
    num_edges = torch.count_nonzero(adj_matrix).item() // 2
    epsilon = 1 / num_edges if num_edges > 0 else 0  # Verhindert Division durch 0
    #epsilon = 0.1
    generator = torch.Generator().manual_seed(seed)  # Zufallszahlengenerator mit Seed

    noise = torch.empty((adj_matrix.shape[0], adj_matrix.shape[0]), dtype=torch.float64).uniform_(-epsilon, epsilon, generator=generator).triu(diagonal=1)
    noise = noise + noise.T
    edge_indices = np.argwhere(adj_matrix.cpu().numpy() != 0)
    adj_matrix_noisy = adj_matrix.clone()
    c = float(0)
    amount = 0
    for i, j in edge_indices:
        adj_matrix_noisy[i, j] += noise[i, j]
        if(i < j):
            c += noise[i, j]
            amount += 1

    print(f'costs: {c}')
    print(f'amount additions: {amount}')
    # Diagonale wieder auf 0 setzen
    adj_matrix_noisy.fill_diagonal_(0)

    return adj_matrix_noisy

def extract_sorted_edges(adj_matrix):
    n = adj_matrix.shape[0]
    indices = torch.triu_indices(n, n, offset=1)
    weights = adj_matrix[indices[0], indices[1]]

    mask = weights != 0
    weights = weights[mask]
    u = indices[0][mask]
    v = indices[1][mask]

    w_np = weights.cpu().numpy()
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()

    # Lexsort mit Reihenfolge v aufsteigend, u aufsteigend, w absteigend
    sort_idx = np.lexsort((v_np, u_np, -w_np))

    edges = torch.stack((weights, u, v), dim=1)
    edges = edges[sort_idx]

    # Umwandeln in Liste von Tuples (float, int, int)
    edges_list = [(float(w), int(ui), int(vi)) for w, ui, vi in edges.tolist()]
    return edges_list

def add_noise_from_edges(n, adj_matrix, edges, seed):
    num_edges = len(edges)
    epsilon = 1 / num_edges if num_edges > 0 else 0
    generator = torch.Generator().manual_seed(seed)

    noise = torch.empty((n, n), dtype=torch.float64).uniform_(-epsilon, epsilon, generator=generator).triu(diagonal=1)
    noise = noise + noise.T

    adj_matrix_noisy = adj_matrix.clone().double()  # wichtig: float64, gleiche Typen!
    c = 0
    amount = 0

    for w, u, v in edges:
        adj_matrix_noisy[u, v] += noise[u, v].item()
        adj_matrix_noisy[v, u] += noise[u, v].item()
        if u < v:
            c += noise[u, v].item()
            amount += 1

    adj_matrix_noisy.fill_diagonal_(0)
    print(f'costs: {c}')
    print(f'amount additions: {amount}')

    return adj_matrix_noisy



def add_noise2(adj_matrix, seed):
    adj_matrix = adj_matrix.to(torch.float64)
    num_edges = torch.count_nonzero(adj_matrix).item() // 2  # Durch 2 teilen, um doppelte Zählung zu korrigieren

    epsilon = 1 / num_edges if num_edges > 0 else 0  # Verhindert Division durch 0
    #epsilon = 0.1
    generator = torch.Generator().manual_seed(seed)  # Zufallszahlengenerator mit Seed

    noise = torch.empty_like(adj_matrix).uniform_(-epsilon, epsilon, generator=generator)


    adj_matrix_noisy = adj_matrix + noise

    adj_matrix_noisy.fill_diagonal_(0)
    unchanged = (adj_matrix_noisy == adj_matrix).sum().item()
    print(f"Anzahl unveränderter Werte: {unchanged}")

    return adj_matrix_noisy

def find_highest_cost_edge(adj_matrix):
    index = torch.argmax(adj_matrix)
    # Konvertiere den linearen Index in (Zeile, Spalte)
    row, col = divmod(index.item(), adj_matrix.shape[1])
    #print(f'row: {row}, col: {col}, value: {adj_matrix[row, col]}')
    return [(row, col)]

# Blossom algorithm
def maximum_weighted_matching(adj_matrix):
    G = nx.from_numpy_array(adj_matrix.numpy())
    return nx.max_weight_matching(G)


def maximum_spanning_forest(adj_matrix):
    positive_adj_matrix = torch.clamp(adj_matrix, min = 0)
    G = nx.from_numpy_array(positive_adj_matrix.numpy())
    return nx.maximum_spanning_tree(G, algorithm='kruskal')

def remove_conflicts_from_forest(adj_matrix, forest):
    #print(nx.has_path(forest, 0, 4))
    #print(adj_matrix)
    n = adj_matrix.shape[0]
    negative_edges = []

    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i, j].item() < 0:
                negative_edges.append((i, j))

    for (i, j) in negative_edges:
        if nx.has_path(forest, source = i, target = j):
            path = nx.shortest_path(forest, source = i, target = j)

            min_edge = None
            min_edge_weight = np.inf
            path_edges = list(zip(path, path[1:]))
            #print(f'path: {path}')
            #print(f'same path: {path_edges}')

            for (u, v) in path_edges:
                w = forest.get_edge_data(u, v)['weight']
                #print(f'check edge ({u}, {v}) with weight {w}')
                if w < min_edge_weight:
                    min_edge = (u, v)
                    min_edge_weight = w
                    
            
            forest.remove_edge(*min_edge)
            #print(f'removed edge {min_edge} with weight {w}')
    sorted_edges = [(u, v) for u, v, _ in sorted(forest.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)]

    return sorted_edges
            



def find_contraction_mapping(adj_matrix, S):
    partition = Partition(adj_matrix.shape[0])
    
    for (node_1, node_2) in S:
        partition.merge(node_1, node_2)
    
    return [partition.find(i) for i in range(adj_matrix.shape[0])]

def construct_contraction_matrix(contraction_mapping):
    unique_nodes, contraction_mapping_indices = np.unique(contraction_mapping, return_inverse=True)
    n = len(contraction_mapping_indices)
    m = len(unique_nodes)
    
    contraction_matrix = np.zeros((n, m))
    contraction_matrix[np.arange(n), contraction_mapping_indices] = 1
    
    return torch.as_tensor(contraction_matrix, dtype=torch.float32, device=device)

def construct_contraction_matrix2(contraction_mapping):
    unique_nodes, contraction_mapping_indices = np.unique(contraction_mapping, return_inverse=True)
    n = len(contraction_mapping_indices)
    m = len(unique_nodes)

    #indices = torch.tensor([np.arange(n), contraction_mapping_indices], dtype=torch.long)
    indices_np = np.vstack([np.arange(n), contraction_mapping_indices])
    indices = torch.tensor(indices_np, dtype=torch.long)

    values = torch.ones(n, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, (n, m), dtype=torch.float64)

def construct_contraction_matrix_test(contraction_mapping):
    unique_nodes, contraction_mapping_indices = np.unique(contraction_mapping, return_inverse=True)
    n = len(contraction_mapping_indices)
    m = len(unique_nodes)

    #indices = torch.tensor([np.arange(n), contraction_mapping_indices], dtype=torch.long)
    indices_np = np.vstack([np.arange(n), contraction_mapping_indices])
    indices = torch.tensor(indices_np, dtype=torch.long)

    values = torch.ones(n, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, (n, m))


def contract(adj_matrix, contraction_matrix):
    tmp = torch.matmul(torch.matmul(contraction_matrix.T, adj_matrix), contraction_matrix).fill_diagonal_(0).triu()
    tmp = tmp + tmp.T
    print(f'type: {tmp.dtype}')
    return tmp

def cost(adj_matrix):
    return torch.triu(adj_matrix).sum()

def check_tree_cost(adj_matrix, forest):
    cost = 0
    for edge in forest:
        cost += adj_matrix[edge[0], edge[1]]
    return cost



def has_path(forest, node1, node2):
    # Baums als Nachbarschaftsliste darstellen
    adj = {}
    for edge in forest:
        if edge[0] not in adj:
            adj[edge[0]] = []
        if edge[1] not in adj:
            adj[edge[1]] = []
        adj[edge[0]].append(edge[1])
        adj[edge[1]].append(edge[0])

    # DFS-Funktion zur Überprüfung des Pfades
    def dfs(current, target, visited):
        if current == target:
            return True
        visited.add(current)
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                if dfs(neighbor, target, visited):
                    return True
        return False

    # Prüfen, ob ein Pfad von node1 zu node2 existiert
    return dfs(node1, node2, set())

# Funktion zur Überprüfung auf Konflikte
def check_conflicts(adj_matrix, forest):
    conflicts = []
    
    # Iteriere über alle negativen Kanten im Graphen
    for i in range(adj_matrix.size(0)):
        for j in range(i + 1, adj_matrix.size(1)):
            if adj_matrix[i, j] < 0:  # Falls es eine negative Kante ist
                # Überprüfe, ob ein Pfad im Forest zwischen i und j existiert
                if has_path(forest, i, j):
                    conflicts.append((i, j))  # Konflikt gefunden
    
    return conflicts

def check_duplicates(adj_matrix):
    unique_values, counts = torch.unique(adj_matrix, return_counts=True)

    duplicates = unique_values[counts > 2]

    print(f'duplicate edge weights: {duplicates}')  


if __name__ == '__main__':
    filename = 'p2000-2.txt'
    device = 'cpu'
    print(f'cuda is available: {torch.cuda.is_available()}')
    print("CUDA-Geräteanzahl:", torch.cuda.device_count())
    print("Aktuelles Gerät:", torch.cuda.current_device())
    print("Gerätename:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print('read data ...')
    adj_matrix = read_data(filename)
    #adj_matrix = adj_matrix.coalesce()
    seed = 40
    #adj_matrix = add_noise_from_edges(adj_matrix.shape[0], adj_matrix, extract_sorted_edges(adj_matrix), seed)
    print(adj_matrix)
    #adj_matrix = add_noise_adj(adj_matrix, seed)
    adj_matrix = add_noise(adj_matrix, seed)
    print(f'adjacency matrix device: {adj_matrix.device}')
    times = [0, 0, 0, 0, 0]
    iterations = 0
    #print('start algorithm ...')
    begin_check_matrix = time.time()
    start_time = time.time()
    #print(remove_conflicts_from_forest(adj_matrix, maximum_spanning_forest(adj_matrix)))
    #print(f'initial adjacency matrix:\n{adj_matrix}')
    n = adj_matrix.shape[0]
    frac = int(n * 0.01)
    if frac == 0:
        frac = 1

    while torch.any(adj_matrix > 0):
        #print(f'Adjacency matrix, shape: ({adj_matrix.shape[0]}, {adj_matrix.shape[1]})')
        #print(adj_matrix)
        #check_duplicates(adj_matrix)
        
        times[0] += time.time() - begin_check_matrix
        #print('hello')
        begin_find_S = time.time()
        #S = find_highest_cost_edge(adj_matrix)
        #S = maximum_weighted_matching(adj_matrix)
        #S = luby_jones_handshake(adj_matrix)
        #S = remove_conflicts_from_forest(adj_matrix, maximum_spanning_forest(adj_matrix))
        #S = find_maximum_spanning_forest_without_conflicts(adj_matrix)
        #print(f'adj_matrix: \n{adj_matrix}')
        S = mst_without_conflicts(adj_matrix)
        if(len(S) == 0):
            print(f'adj_matrix: \n{adj_matrix}')
            break
        #S_1 = mst_without_conflicts(adj_matrix)
        #print(f'S: {S}')
        #print(f'conflicts: {check_conflicts(adj_matrix, S)}')
        S = S[:frac]
        #S_1 = S_1[:frac]
        """tree1 = remove_conflicts_from_forest(adj_matrix, maximum_spanning_forest(adj_matrix))
        tree2 = kruskal(adj_matrix)
        #tree1_sorted = [tuple(sorted(edge)) for edge in tree1]
        #tree2_sorted = [tuple(sorted(edge)) for edge in tree2]
        print(f'tree: {sorted(tree1)}')
        print(f'cost: {check_tree_cost(adj_matrix, tree1)}')
        print(f'amount edges: {len(tree1)}')
        #print(f'has conflicts: {check_conflicts(adj_matrix, tree1)}')
        print('****************')
        print(f'tree: {sorted(tree2)}')
        print(f'cost: {check_tree_cost(adj_matrix, tree2)}')
        print(f'amount edges: {len(tree2)}')"""
        #print(f'has conflicts: {check_conflicts(adj_matrix, S)}')
        print(f'S: {S}, |S| = {len(S)}')
        #print(f'S_1: {S_1}, |S_1| = {len(S_1)}')
        #S = remove_conflicts_from_forest(adj_matrix, boruvka_algorithm(adj_matrix))
        times[1] += time.time() - begin_find_S
        #print(f'contraction set: {S}')
        #S = [(0,1), (1,2)]
        begin_find_cm = time.time()
        contraction_mapping = find_contraction_mapping(adj_matrix, S)
        times[2] += time.time() - begin_find_cm
        begin_construct = time.time()
        contraction_matrix = construct_contraction_matrix2(contraction_mapping)
        #contraction_matrix = construct_contraction_matrix_test(contraction_mapping)
        
        #print(f'contraction matrix: \n{contraction_matrix}')
        times[3] += time.time() - begin_construct
        #print(f'shape contraction matrix: {contraction_matrix.shape}\nshape adjacency matrix: {adj_matrix.shape}')
        begin_contract = time.time()
        adj_matrix = contract(adj_matrix, contraction_matrix)
        #torch.cuda.synchronize()
        times[4] += time.time() - begin_contract
        iterations += 1
        begin_check_matrix = time.time()
        #torch.cuda.synchronize()
        #print(f'contraction set: \n{S}\ncontraction matrix: \n{contraction_matrix}\nadjacency matrix: \n{adj_matrix}')
    times[0] += time.time() - begin_check_matrix
    
    print(f'iterations: {iterations}')
    print(f'cost: {cost(adj_matrix)}, \ntime: {time.time()-start_time :.2f}s')
    print(f'time checking adjacency matrix: {times[0] :.6f}\ntime finding S: {times[1] :.6f}\ntime finding contraction mapping: {times[2] :.6f}\ntime construct contraction matrix: {times[3] :.6f}\ntime contracting: {times[4] :.6f}')
    print(f'time sum: {sum(times) :.2f}s')
    print(f'adj_matrix < 0 für: \n{adj_matrix}')
    
    

    