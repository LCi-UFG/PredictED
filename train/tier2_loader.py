import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from graphs import (
    chem2dataset, 
    pred2dataset
    )
from params import (
    load_params, 
    load_tier1_model
    )
from utils import (
    device, 
    set_seed
    )
from predictor import predict_tier1


def collate_graphs(batch):
    mol_list, pred_list, ys = [], [], []
    for (mol_graph, pred_graph), y in batch:
        mol_list.append(mol_graph)
        pred_list.append(pred_graph)
        ys.append(y)
    mol_batch = Batch.from_data_list(mol_list)
    pred_batch = Batch.from_data_list(pred_list)
    y = torch.stack(ys)
    mol_batch.y = y
    pred_batch.y = y
    return (mol_batch, pred_batch), y


class CombinedDataset(Dataset):
    def __init__(self, mol_dataset, pred_dataset):
        if len(mol_dataset) != len(pred_dataset):
            raise ValueError("Objects must have same length")
        self.mol_dataset = mol_dataset
        self.pred_dataset = pred_dataset

    def __len__(self):
        return len(self.mol_dataset)

    def __getitem__(self, idx):
        mol_data = self.mol_dataset[idx]
        pred_data = self.pred_dataset[idx]

        return (mol_data, pred_data), mol_data.y


def graph_loader(
    train_smiles, 
    val_smiles, 
    test_smiles,
    y_train=None, 
    y_val=None, 
    y_test=None,
    batch_size=32,
    hyperparams_path=None, 
    model_path=None,
    architecture_type=None,
    data_path=None, 
    feature_path=None,
    device=device,
    seed=None):

    if seed is not None:
        set_seed(seed)

    def _worker_init_fn(worker_id):
        base = 42 if seed is None else seed
        rs = base + worker_id
        random.seed(rs)
        np.random.seed(rs)
        torch.manual_seed(rs)

    g = torch.Generator()
    g.manual_seed(42 if seed is None else seed)

    mol_train_ds = chem2dataset(
        train_smiles, y_train
        )
    mol_val_ds = chem2dataset(
        val_smiles, y_val
        )
    mol_test_ds = chem2dataset(
        test_smiles, y_test
        )
    full_loader = DataLoader(
        mol_train_ds, 
        batch_size=len(mol_train_ds), 
        shuffle=False,
        collate_fn=lambda x: Batch.from_data_list(x),
        num_workers=0,
        worker_init_fn=_worker_init_fn,
        generator=g
        )
    mol_full = next(iter(full_loader))
    nd = mol_full.x.size(1)
    ed = mol_full.edge_attr.size(1) if hasattr(
        mol_full, 'edge_attr') else None

    params = load_params(hyperparams_path)
    ckpt = torch.load(
        model_path, map_location=device
        )
    nt = next(v.size(0) for k, v in ckpt.items() 
        if k.endswith('output_layer.weight')
        )
    tier1_model = load_tier1_model(
        model_path, 
        architecture_type, 
        params, nd, ed, nt).to(device)
    tier1_model.eval()

    pred_kwargs = dict(
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: Batch.from_data_list(x),
        num_workers=0,
        worker_init_fn=_worker_init_fn,
        generator=g
        )
    with torch.no_grad():
        tp, _, _ = predict_tier1(
            tier1_model, DataLoader(
            mol_train_ds, **pred_kwargs), device
            )
        vp, _, _ = predict_tier1(
            tier1_model, DataLoader(
            mol_val_ds, **pred_kwargs), device
            )
        sp, _, _ = predict_tier1(
            tier1_model, DataLoader(
            mol_test_ds, **pred_kwargs), device
            )
    pred_train_ds = pred2dataset(
        tp, labels=y_train, 
        data_path=data_path, 
        feature_path=feature_path
        )
    pred_val_ds   = pred2dataset(
        vp, labels=y_val, 
        data_path=data_path, 
        feature_path=feature_path
        )
    pred_test_ds  = pred2dataset(
        sp, labels=y_test,  
        data_path=data_path, 
        feature_path=feature_path
        )
    train_ds = CombinedDataset(
        mol_train_ds, pred_train_ds)
    val_ds = CombinedDataset(
        mol_val_ds, pred_val_ds)
    test_ds = CombinedDataset(
        mol_test_ds, pred_test_ds)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,
        worker_init_fn=_worker_init_fn,
        generator=g,
        drop_last=True
        )
    val_loader   = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
        worker_init_fn=_worker_init_fn,
        generator=g
        )
    test_loader  = DataLoader(
        test_ds,  
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
        worker_init_fn=_worker_init_fn,
        generator=g
        )

    return train_loader, val_loader, test_loader




def graph_loader_inference(
    smiles,
    batch_size=1,
    hyperparams_path=None,
    model_path=None,
    architecture_type=None,
    data_path=None,
    feature_path=None,
    device=device,
    seed=None):

    if seed is not None:
        set_seed(seed)

    if isinstance(smiles, str):
        smiles = [smiles]

    y_dummy = [0] * len(smiles)

    mol_ds = chem2dataset(smiles, y_dummy)

    dummy_loader = DataLoader(
        mol_ds,
        batch_size=len(mol_ds),
        shuffle=False,
        collate_fn=lambda x: Batch.from_data_list(x)
    )
    mol_full = next(iter(dummy_loader))
    nd = mol_full.x.size(1)
    ed = mol_full.edge_attr.size(1) if hasattr(
        mol_full, 'edge_attr') else None
    
    params = load_params(hyperparams_path)
    ckpt = torch.load(model_path, map_location=device)
    nt = next(v.size(0) for k, v in ckpt.items(
        ) if k.endswith('output_layer.weight'))
    tier1_model = load_tier1_model(
        model_path,
        architecture_type,
        params,
        nd, ed, nt).to(device)

    pred_loader = DataLoader(
        mol_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: Batch.from_data_list(x)
    )
    preds, _, _ = predict_tier1(tier1_model, pred_loader, device)

    pred_ds = pred2dataset(
        preds,
        labels=y_dummy,
        data_path=data_path,
        feature_path=feature_path
    )

    dataset = CombinedDataset(mol_ds, pred_ds)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_graphs
    )

    return loader


def graph_info(loader):

    for batch_data, y in loader:
        mol_batch, pred_batch = batch_data
        break
    node_dim = mol_batch.x.size(1)
    edge_dim = (mol_batch.edge_attr.size(1)
            if hasattr(mol_batch, 'edge_attr') and
                mol_batch.edge_attr is not None and
                mol_batch.edge_attr.dim() == 2
            else (1 if mol_batch.edge_attr is not None else None))
    node_pred = pred_batch.x.size(1)
    edge_pred = (pred_batch.edge_attr.size(1)
            if hasattr(pred_batch, 'edge_attr') and
                pred_batch.edge_attr is not None and
                pred_batch.edge_attr.dim() == 2
            else (1 if getattr(
                pred_batch, 'edge_attr', None) is not None else None))
    num_tasks = y.size(-1) if y.ndim > 1 else 1

    return (node_dim, edge_dim, 
            node_pred, edge_pred, num_tasks
           )
