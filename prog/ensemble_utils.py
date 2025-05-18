import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import networkx as nx

def torch_spread(GAT_out, pickup_nodes, edge_index, num_nodes, alpha=1.0, max_distance=3):
    device = GAT_out.device
    num_classes = GAT_out.size(1)

    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    adj = (adj + adj.T).clamp(max=1)

    P = len(pickup_nodes)
    dist = torch.full((num_nodes, P), float('inf'), device=device)
    visited = torch.zeros((num_nodes, P), dtype=torch.bool, device=device)
    dist[pickup_nodes, torch.arange(P, device=device)] = 0
    frontier = torch.zeros((num_nodes, P), device=device)
    frontier[pickup_nodes, torch.arange(P, device=device)] = 1

    for d in range(1, max_distance + 1):
        newly_reached = (frontier > 0) & (~visited)
        dist[newly_reached] = d
        visited |= newly_reached
        frontier = adj @ frontier  # 次のフロンティアへ

    weight = torch.exp(-alpha * dist)
    weight[torch.isinf(dist)] = 0.0

    total_weight = weight.sum(dim=1, keepdim=True) + 1e-8
    weighted_out = weight @ GAT_out
    final_pred = weighted_out / total_weight

    return final_pred


def build_pickup_edge_index(pickup_nodes, data, method="shortest_path", top_k=2):
    G_nx = nx.Graph()
    G_nx.add_edges_from(data.edge_index.t().tolist())
    pickup_list = pickup_nodes.tolist()

    new_edges = []

    if method == "shortest_path":
        for src in pickup_list:
            dists = {}
            for dst in pickup_list:
                if src == dst:
                    continue
                try:
                    d = nx.shortest_path_length(G_nx, source=src, target=dst)
                    dists[dst] = d
                except nx.NetworkXNoPath:
                    continue
            closest_2 = sorted(dists.items(), key=lambda x: x[1])[:top_k]
            for dst, _ in closest_2:
                edge = tuple(sorted((src, dst)))
                new_edges.append(edge)

    elif method == "feature_similarity":
        x = data.x[pickup_nodes]  # shape: [P, F]
        sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)  # [P, P]
        for i in range(len(pickup_list)):
            sims = sim[i].clone()
            sims[i] = -1  # 除外
            topk = torch.topk(sims, k=top_k).indices.tolist()
            for j in topk:
                u, v = sorted((pickup_list[i], pickup_list[j]))
                new_edges.append((u, v))

    elif method == "hybrid":
        x = data.x[pickup_nodes]  # shape: [P, F]
        sim_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)  # [P, P]
        old_to_new = {old: new for new, old in enumerate(pickup_list)}
        for src in pickup_list:
            scores = {}
            for dst in pickup_list:
                if src == dst:
                    continue
                try:
                    d = nx.shortest_path_length(G_nx, source=src, target=dst)
                    i = old_to_new[src]
                    j = old_to_new[dst]
                    sim = sim_matrix[i][j].item()
                    score = sim / ((1 + d)**2)
                    scores[dst] = score
                except nx.NetworkXNoPath:
                    continue
            top2 = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
            for dst, _ in top2:
                edge = tuple(sorted((src, dst)))
                new_edges.append(edge)


    else:
        raise ValueError("Unknown method for edge building")

    new_edges = list(set(new_edges))
    old_to_new = {old: new for new, old in enumerate(pickup_list)}
    remapped_edges = [[old_to_new[u], old_to_new[v]] for u, v in new_edges]
    edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
    
    return edge_index