import os
import torch
import networkx as nx
from torch_geometric.utils import to_networkx

def compute_or_load_hop_matrix(data, dataset_name, max_distance=3, save_dir='hop_matrices'):
    os.makedirs(save_dir, exist_ok=True)  # フォルダがなければ作成

    filename = f"{dataset_name}_hop_matrix_d{max_distance}.pt"
    path = os.path.join(save_dir, filename)

    if os.path.exists(path):
        print(f"✅ Loading hop matrix from {path}")
        hop_matrix = torch.load(path)
    else:
        print(f"🚀 Computing hop matrix for dataset: {dataset_name}")
        G = to_networkx(data, to_undirected=True)
        num_nodes = data.num_nodes
        hop_matrix = torch.full((num_nodes, num_nodes), float('inf'))
        for src in range(num_nodes):
            lengths = nx.single_source_shortest_path_length(G, source=src, cutoff=max_distance)
            for dst, d in lengths.items():
                hop_matrix[src][dst] = d
        torch.save(hop_matrix, path)
        print(f"✅ Saved hop matrix to {path}")

    return hop_matrix


from torch_geometric.datasets import Planetoid

dataset_name = 'Cora'
dataset = Planetoid(root='/tmp/Cora', name=dataset_name)
data = dataset[0]

# hop_matrices/ フォルダは事前に作成されている前提
hop_matrix_cpu = compute_or_load_hop_matrix(data, dataset_name, max_distance=10)
hop_matrix_gpu = hop_matrix_cpu.to('cuda')