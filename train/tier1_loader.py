import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from graphs import chem2dataset
from utils import set_seed


def collate_graphs(samples):
    samples = [s for s in samples 
        if s is not None and hasattr(s, 'y')]
    if len(samples) == 0:
        return None

    graphs = [s for s in samples if isinstance(s, Data)]
    batch = Batch.from_data_list(graphs)
    batch.smiles = [s.smiles for s in samples]
    labels = [s.y for s in samples]
    batch.y = torch.stack(labels) if labels else None

    return batch


def graph_loader(
    train_smiles,
    val_smiles,
    test_smiles,
    y_train=None,
    y_val=None,
    y_test=None,
    batch_size=32,
    seed=None):

    if seed is not None:
        set_seed(seed)

    train_dataset = chem2dataset(train_smiles, y_train)
    val_dataset   = chem2dataset(val_smiles, y_val)
    test_dataset  = chem2dataset(test_smiles, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,    
        collate_fn=collate_graphs
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs
        )

    return train_loader, val_loader, test_loader


def graph_info(data_loader):
    node_feature_dim = None
    edge_feature_dim = None
    num_tasks = None

    for batch in data_loader:
        if batch is None:
            continue
        if isinstance(batch, Batch):
            if batch.y is not None and batch.y.size(0) > 0:
                num_tasks = batch.y.size(1
                    ) if batch.y.ndim > 1 else 1
            node_feature_dim = batch.x.size(1)
            edge_feature_dim = batch.edge_attr.size(1)
        else:
            if batch.y is not None and batch.y.size(0) > 0:
                num_tasks = batch.y.size(1
                    ) if batch.y.ndim > 1 else 1
            node_feature_dim = batch.x.size(1)
            edge_feature_dim = batch.edge_attr.size(1)

        break

    return node_feature_dim, edge_feature_dim, num_tasks