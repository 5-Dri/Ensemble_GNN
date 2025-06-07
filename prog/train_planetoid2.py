from tqdm import tqdm
import numpy as np
import mlflow

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from torch_geometric.data import Data

from models.model_loader import load_net
from utils import fix_seed, accuracy, random_splits, index_to_mask, EarlyStopping
from ensemble_utils import build_pickup_edge_index
from ensemble_utils import torch_spread
from models.layer import orthonomal_loss


def train(cfg, data, model, optimizer, device):
    model.train()
    optimizer.zero_grad()

    h, _ = model(data.x, data.edge_index)
    prob_labels = F.log_softmax(h, dim=1)
    loss_train  = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    # if cfg.global_skip_connection == 'twin': # if it is proposal model
    #     loss_train += cfg.coef_orthonomal * orthonomal_loss(model, device)
    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    h, _ = model(data.x, data.edge_index)
    prob_labels_val = F.log_softmax(h, dim=1)
    loss_val = F.nll_loss(prob_labels_val[data.val_mask], data.y[data.val_mask])
    acc_val, _ = accuracy(prob_labels_val[data.val_mask], data.y[data.val_mask])

    return loss_val.item(), acc_val


def test(data, model):
    model.eval()
    h, alpha = model(data.x, data.edge_index)
    prob_labels_test = F.log_softmax(h, dim=1)
    acc, _ = accuracy(prob_labels_test[data.test_mask], data.y[data.test_mask])
    _, whole_node_correct = accuracy(prob_labels_test, data.y)

    return acc, alpha, whole_node_correct


# def ensemble(cfg, data, device):

#     model = load_net(cfg).to(device)
#     optimizer = torch.optim.Adam(params       = model.parameters(), 
#                                  lr           = cfg.learning_rate, 
#                                  weight_decay = cfg.weight_decay)
#     pickup_num = int(cfg.num_nodes * cfg.pickup_ratio)
#     available_indices = torch.where(data.train_mask | data.val_mask | data.test_mask)[0]
#     pred_sum = torch.zeros((cfg.num_nodes, cfg.n_class), device=device)
    

#     for _ in range(cfg.graphs_number):
#         if cfg.strategy == "random":
#             pickup_nodes = available_indices[torch.randperm(len(available_indices))[:pickup_num]]

#         else:
#             deg = degree(data.edge_index[0], num_nodes=cfg.num_nodes)
#             deg = deg[available_indices]

#             if cfg.strategy == "high_degree":
#                 prob = deg + 1e-6
#             elif cfg.strategy == "low_degree":
#                 prob = 1.0 / (deg + 1e-6)
#             else:
#                 raise ValueError("Unknown strategy")
            
#             prob = prob / prob.sum()
#             prob = prob.to(available_indices.device)
#             pickup_nodes = available_indices[torch.multinomial(prob, pickup_num, replacement=False)].to(device)

#         pickup_edge_index = build_pickup_edge_index(pickup_nodes, data, cfg)
#         pickup_x = data.x[pickup_nodes]
#         pickup_y = data.y[pickup_nodes]

#         pickup_mask = index_to_mask(pickup_nodes, size=cfg.num_nodes)

#         pickup_data = Data(
#             x=pickup_x,
#             y=pickup_y,
#             edge_index=pickup_edge_index.to(device),
#             train_mask=data.train_mask[pickup_mask],
#             val_mask=data.val_mask[pickup_mask],
#             test_mask=data.test_mask[pickup_mask]
#         )
#         early_stopping = EarlyStopping(patience=10)


#         for epoch in range(1, cfg.epochs+1):
#             loss_val, mini_acc_val = train(cfg, pickup_data, model, optimizer, device)
#             early_stopping(loss_val, model)
#             if early_stopping.early_stop:
#                 break
            
#         model.eval()
#         out, _ = model(pickup_data.x, pickup_data.edge_index)
#         spread_pred = torch_spread(cfg, out, pickup_nodes, data, alpha=1.0, max_distance=3)
#         log_pred = F.log_softmax(spread_pred, dim=1)
#         pred_sum += log_pred

#     final_log_pred = pred_sum / cfg.graphs_number
#     pred_class = final_log_pred.argmax(dim=1)
#     correct_val = (pred_class[data.val_mask] == data.y[data.val_mask]).sum().item()
#     correct_test = (pred_class[data.test_mask] == data.y[data.test_mask]).sum().item()
#     acc_val = correct_val / data.val_mask.sum().item()
#     acc_test = correct_test / data.test_mask.sum().item()

#     return acc_val, acc_test

def ensemble(cfg, data, device):
    """Run k independent sub‑graph trainings and ensemble their predictions.

    Key change: *model* と *optimizer* を **ループ内で毎回初期化**し、
    各サブグラフが独立した重みで学習されるようにする。
    """
    data.to(device)
    pickup_num = int(cfg.num_nodes * cfg.pickup_ratio)
    prob_sum   = torch.zeros((cfg.num_nodes, cfg.n_class), device=device)

    train_idx  = torch.where(data.train_mask)[0].to(device)
    other_idx  = torch.where(data.val_mask | data.test_mask)[0].to(device)

    # 各ループで再計算するので固定値として保持
    base_train_pickup = int(pickup_num * len(train_idx) / cfg.num_nodes)
    base_other_pickup = pickup_num - base_train_pickup

    for _ in range(cfg.graphs_number):
        # ---------------------------
        # ① ノードサンプリング
        # ---------------------------
        train_pickup_num  = min(base_train_pickup, len(train_idx))
        other_pickup_num  = min(base_other_pickup, len(other_idx))

        if cfg.strategy == "random":
            train_pickup = train_idx [torch.randperm(len(train_idx)) [:train_pickup_num]]
            other_pickup = other_idx [torch.randperm(len(other_idx)) [:other_pickup_num]]
        else:
            deg = degree(data.edge_index[0], num_nodes=cfg.num_nodes).to(device)

            # train 側
            tr_prob = deg[train_idx] + 1e-6 if cfg.strategy == "high_degree" else 1.0 / (deg[train_idx] + 1e-6)
            tr_prob = tr_prob / tr_prob.sum()
            train_pickup = train_idx[torch.multinomial(tr_prob, train_pickup_num, replacement=False)]

            # other 側
            ot_prob = deg[other_idx] + 1e-6 if cfg.strategy == "high_degree" else 1.0 / (deg[other_idx] + 1e-6)
            ot_prob = ot_prob / ot_prob.sum()
            if other_pickup_num == 0:
                other_pickup = torch.tensor([], dtype=torch.long, device=device)
            else:
                other_pickup = other_idx[torch.multinomial(ot_prob, other_pickup_num, replacement=False)]

        pickup_nodes = torch.cat([train_pickup, other_pickup])

        # ---------------------------
        # ② サブグラフ構築
        # ---------------------------
        edge_sub = build_pickup_edge_index(pickup_nodes, data, cfg).to(device)
        sub_data = Data(
            x          = data.x[pickup_nodes],
            y          = data.y[pickup_nodes],
            edge_index = edge_sub,
            train_mask = data.train_mask[pickup_nodes],
            val_mask   = data.val_mask  [pickup_nodes],
            test_mask  = data.test_mask [pickup_nodes]
        )

        # ---------------------------
        # ③ モデルとオプティマイザをループ内で初期化
        # ---------------------------
        model = load_net(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        stopper = EarlyStopping(patience=cfg.patience)

        for epoch in range(1, cfg.epochs + 1):
            val_loss, _ = train(cfg, sub_data, model, optimizer, device)  # train() returns (loss_val, acc_val)
            stopper(val_loss, model)  # 監視対象 = validation loss
            if stopper.early_stop:
                break

        # ---------------------------
        # ④ 推論して原グラフに拡散
        # ---------------------------
        model.eval()
        logits_pickup, _ = model(sub_data.x, sub_data.edge_index)
        spread = torch_spread(cfg, logits_pickup, pickup_nodes, data, alpha=1.0, max_distance=3)
        prob_sum += F.softmax(spread, dim=1)  # 確率で加算

    # ---------------------------
    # ⑤ 予測集約 & 評価
    # ---------------------------
    prob_final  = prob_sum / cfg.graphs_number
    pred_class  = prob_final.argmax(dim=1)

    val_acc  = (pred_class[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
    test_acc = (pred_class[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return val_acc, test_acc



def train_and_test(cfg, data, device):
    model = load_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg.learning_rate, 
                                 weight_decay = cfg.weight_decay)
    early_stopping = EarlyStopping(patience=10)

    for epoch in range(1, cfg.epochs+1):
        loss_val, acc_val = train(cfg, data, model, optimizer, device)
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            break

    acc_test, alpha, whole_node_correct = test(data, model)

    return acc_val, acc_test

def run(cfg, root, device):
    if cfg.x_normalize:
        transforms = T.NormalizeFeatures()
    else:
        transforms = None

    valid_acces, test_acces, artifacts = [], [], {}
    for tri in tqdm(range(cfg.n_tri)):
        if cfg.debug_mode:
            fix_seed(cfg.seed)
        else:
            fix_seed(cfg.seed + tri) 
        # [train, valid, test] is splited based on above seed
        dataset = Planetoid(root      = root + '/data/' + cfg.dataset,
                            name      = cfg.dataset,
                            transform = transforms)
        data = dataset[0].to(device)
        data, index = random_splits(data=data,num_classes=cfg.n_class,lcc_mask=None)

        if cfg.ensemble == True:
            valid_acc, test_acc = ensemble(cfg, data, device)
        elif cfg.ensemble == False:
            valid_acc, test_acc = train_and_test(cfg, data, device)


        valid_acces.append(valid_acc)
        test_acces.append(test_acc)
                           
        # valid_acces.append(valid_acc.to('cpu').item())
        # test_acces.append(test_acc.to('cpu').item())
        # artifacts['alpha_{}.npy'.format(tri)] = alpha
        # artifacts['correct_{}.npy'.format(tri)] = correct
        # artifacts['test_mask_{}.npy'.format(tri)] = data.test_mask

    return valid_acces, test_acces