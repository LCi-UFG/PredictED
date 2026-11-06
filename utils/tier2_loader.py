import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from features.graphs import prediction_graph, smiles2graph
from train.predictor import deterministic_embeddings


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


def graph_loader_inference(smiles, runtime):
    molecule = smiles2graph(smiles)
    molecule.y = torch.tensor(0.0, dtype=torch.float32)
    _, logits = deterministic_embeddings(runtime, smiles)
    pathway = prediction_graph(
        logits["tier1"].squeeze(0).detach().cpu(),
        runtime.edge_index.cpu(), runtime.edge_attr.cpu()
    )
    pathway.y = torch.tensor([0.0], dtype=torch.float32)
    dataset = CombinedDataset([molecule], [pathway])
    return DataLoader(dataset, batch_size=1, shuffle=False,
                      collate_fn=collate_graphs)
