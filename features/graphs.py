import torch
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from rules import ATOMIC_NUMBER
from utils import one_hot


def get_atomic_number():
    atomic_number_list = ATOMIC_NUMBER()
    def atomic_number(number):
        return one_hot(number, atomic_number_list)
    return atomic_number


def get_degree(degree):
    degree_categories = list(range(0, 8))
    return one_hot(degree, degree_categories)


def get_charge(charge):
    charge_categories = [-1, 0, 1, 2, 3, 4]
    return one_hot(charge, charge_categories)


def get_hybridization(hybridization):
    hybridization_categories = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
        ]
    if hybridization not in hybridization_categories:
        hybridization = Chem.rdchem.HybridizationType.SP3

    return one_hot(
        hybridization, 
        hybridization_categories
        )


def get_chirality(atom):
    chiral_tag = atom.GetChiralTag()
    if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        return 'R'
    elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return 'S'
    return None


def encode_stereo(chirality, stereo):
    bond_stereo_categories = [
        Chem.rdchem.BondStereo.STEREONONE, 
        Chem.rdchem.BondStereo.STEREOANY, 
        Chem.rdchem.BondStereo.STEREOZ,  
        Chem.rdchem.BondStereo.STEREOE, 
        'R', 'S']
    stereo = one_hot(chirality if chirality 
                   else stereo, 
                   bond_stereo_categories
                   )
    return stereo


def get_bond_stereo(stereo, atom=None):
    chirality = get_chirality(atom
            ) if atom else None
    stereo = encode_stereo(chirality, stereo)

    return stereo


def atom_features(atom):
    atomic_number = get_atomic_number()
    atom_feature = torch.cat([
        torch.tensor(atomic_number(
            atom.GetAtomicNum()), 
            dtype=torch.float),
        torch.tensor(get_degree(
            atom.GetDegree()), 
            dtype=torch.float),
        torch.tensor(get_charge(
            atom.GetFormalCharge()), 
            dtype=torch.float),
        torch.tensor(get_hybridization(
            atom.GetHybridization()), 
            dtype=torch.float),
        torch.tensor([float(
            atom.GetIsAromatic())], 
            dtype=torch.float)
            ]
        )
    return atom_feature


def bond_features(bond):
    bond_feature = torch.cat([
        torch.tensor(one_hot(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE, 
            Chem.rdchem.BondType.DOUBLE, 
            Chem.rdchem.BondType.TRIPLE, 
            Chem.rdchem.BondType.AROMATIC]), 
            dtype=torch.float),
        torch.tensor([float(
            bond.GetIsConjugated())], 
            dtype=torch.float),
        torch.tensor([float(
            bond.IsInRing())], 
            dtype=torch.float),
        torch.tensor(
            get_bond_stereo(bond.GetStereo()), 
            dtype=torch.float)
            ]
        )
    return bond_feature


def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_features_list = [
        atom_features(atom) 
        for atom in mol.GetAtoms()
        ]
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feature = bond_features(bond)
        edge_indices.append([i, j])
        edge_features.append(bond_feature)
    x = torch.stack(atom_features_list)
    edge_index = torch.tensor(
        edge_indices, 
        dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_features)

    return Data(x=x, edge_index=edge_index, 
                edge_attr=edge_attr
               )


class chem2dataset(Dataset):
    def __init__(self, smiles_list, labels=None):
        self.smiles_list = smiles_list
        self.labels = labels
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx
            ] if self.labels is not None else None
        data = smiles2graph(smiles)
        data.smiles = smiles
        if data is None:
            return None
        if label is not None:
            data.y = torch.tensor(
                label, dtype=torch.float32)
            
        return data
    

def load_task_data(
    data_path,
    feature_path):

    df_data = pd.read_csv(data_path)
    assays = df_data.columns[2:].tolist()
    df_info = (
        pd.read_csv(feature_path)
        .set_index("Assay")
        .loc[assays]
        .reset_index()
        )
    assert df_info["Assay"
        ].tolist() == assays
    return df_info


def prepare_thresholds(df_info):
    thresholds = torch.tensor(
        df_info["Threshold"]
        .astype(float)
        .values,
        dtype=torch.float32
        )
    recall = df_info["Recall"].astype(float).values
    spec = df_info["Specificity"].astype(float).values
    meta_feats = torch.tensor(
        np.stack([recall, spec], axis=1),
        dtype=torch.float32
        )
    return thresholds, meta_feats


def assign_task_hierarchy(df_info):
    ER = {"ERα", "ERα/ERβ", "ERβ"}
    AR = {"ARE", "Nrf2/ARE", "AR"}
    df_info["Group"] = df_info["Target"].apply(
        lambda t:
        "ER" if t in ER else
        ("AR" if t in AR else "Other")
        )
    def rule_lvl(r):
        ru = r.upper()
        if ru == "MIE":
            return 0
        if ru.startswith("KE"):
            try:
                return int(ru[2:])
            except:
                return 1
        kes = [x for x in df_info["Rule"]
            if x.upper().startswith("KE")
            ]
        max_k = max((rule_lvl(x) for x in kes), 
            default=0)
        return max_k + 1
    
    df_info["Rule_level"] = df_info["Rule"].apply(rule_lvl)


def build_interactions(
    df_info,
    pmi,
    q_mat):

    G = nx.Graph()
    n = len(df_info)
    G.add_nodes_from(range(n))
    ke_groups = {}
    for i, (grp, lvl) in enumerate(
        zip(df_info["Group"], df_info["Rule_level"])):
        ke_groups.setdefault((grp, lvl), []).append(i)
    for nodes in ke_groups.values():
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                u, v = nodes[i], nodes[j]
                G.add_edge(u, v, weight=None)
    gids = df_info["Group"].values
    lvls = df_info["Rule_level"].values
    for i in range(n):
        for j in range(n):
            if (gids[i] == gids[j] and
                abs(lvls[i] - lvls[j]) == 1):
                G.add_edge(i, j, weight=None)
    er_mies = df_info[
        (df_info["Group"]=="ER") &
        (df_info["Rule_level"]==0)].index.tolist()
    ar_mies = df_info[
        (df_info["Group"]=="AR") &
        (df_info["Rule_level"]==0)].index.tolist()
    for u in er_mies:
        for v in ar_mies:
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=None)
    src, dst = [], []
    for u, v in G.edges():
        src += [u, v]
        dst += [v, u]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attrs = [[pmi[u, v], q_mat[u, v]] for u, v in zip(src, dst)]
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    return edge_index, edge_attr


class pred2dataset(Dataset):
    def __init__(
        self,
        predictions,      
        labels=None,
        data_path=None,
        feature_path=None,
        alpha_edges: float = 5.0):
        
        self.predictions = np.asarray(predictions, dtype=np.float32)
        self.labels = labels
        self.num_tasks = self.predictions.shape[1]

        df_info = load_task_data(data_path, feature_path)
        self.thresholds, _ = prepare_thresholds(df_info)
        assign_task_hierarchy(df_info)
        logits_scaled = np.clip(
            self.predictions / alpha_edges, -40.0, 40.0
            )
        probs_edges = 1.0 / (1.0 + np.exp(-logits_scaled))
        cat = (probs_edges >= self.thresholds.numpy()).astype(int)

        self.prior_pos = torch.tensor(
            np.nanmean(cat, axis=0), 
            dtype=torch.float32).clamp_(1e-6, 1 - 1e-6)

        T = self.num_tasks
        p_i = np.nanmean(cat, axis=0)
        p_ij = np.zeros((T, T), dtype=np.float64)
        for a in range(T):
            for b in range(T):
                mask = ~np.isnan(cat[:, a]) & ~np.isnan(cat[:, b])
                if mask.sum() > 0:
                    p_ij[a, b] = np.mean(cat[mask, a] * cat[mask, b])

        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log(p_ij / (p_i[:, None] * p_i[None, :]))
            np.fill_diagonal(pmi, 0.0)
            npmi = pmi / (-np.log(p_ij + 1e-8))
            npmi = np.nan_to_num(npmi, nan=0.0, posinf=0.0, neginf=0.0)

        or_mat = np.zeros((T, T), dtype=np.float64)
        for a in range(T):
            for b in range(T):
                mask = ~np.isnan(cat[:, a]) & ~np.isnan(cat[:, b])
                if mask.sum() > 0:
                    ta, tb = cat[mask, a], cat[mask, b]
                    TP = ((ta == 1) & (tb == 1)).sum()
                    TN = ((ta == 0) & (tb == 0)).sum()
                    FP = ((ta == 0) & (tb == 1)).sum()
                    FN = ((ta == 1) & (tb == 0)).sum()
                    denom = FP * FN
                    if denom > 0:
                        or_mat[a, b] = (TP * TN) / denom

        q_mat = (or_mat - 1.0) / (or_mat + 1.0)
        q_mat = np.nan_to_num(q_mat, nan=0.0, posinf=0.0, neginf=0.0)
        npmi = (npmi + 1.0) / 2.0
        q_mat = (q_mat + 1.0) / 2.0

        self.edge_index, self.edge_attr = build_interactions(df_info, npmi, q_mat)

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, idx):
        logits = torch.tensor(self.predictions[idx], dtype=torch.float32)
        x = logits.unsqueeze(1)
        data = Data(
            x=x, edge_index=self.edge_index,
            edge_attr=self.edge_attr
            )
        if self.labels is not None:
            data.y = torch.tensor(
                self.labels[idx], 
                dtype=torch.float32).unsqueeze(0)
            
        return data