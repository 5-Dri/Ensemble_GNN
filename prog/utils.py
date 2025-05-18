import mlflow
import numpy as np

import torch
from torch_geometric.utils import subgraph

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def log_params_from_omegaconf_dict(params):
    for param_name, value in params.items():
        print('{}: {}'.format(param_name, value))
        mlflow.log_param(param_name, value)

def log_artifacts(artifacts):
    if artifacts is not None:
        for artifact_name, artifact in artifacts.items():
            if artifact is not None:
                np.save(artifact_name, artifact.to('cpu').detach().numpy().copy())
                mlflow.log_artifact(artifact_name)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels), correct

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def random_splits(data, num_classes, lcc_mask=None):
    torch.manual_seed(42)
    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    val_size = 500 // num_classes
    test_size = 1000 // num_classes

    train_index = torch.cat([i[:40] for i in indices], dim=0)
    val_index = torch.cat([i[40:40+val_size] for i in indices], dim=0)
    test_index = torch.cat([i[40+val_size:40+val_size+test_size] for i in indices], dim=0)
    test_index = test_index[torch.randperm(test_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data, torch.cat((train_index, val_index), dim=0)


class EarlyStopping:
    def __init__(self, patience=10, path='best_model.pt'):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)