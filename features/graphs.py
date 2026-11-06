"""Molecular and pathway graphs used by the released checkpoints."""

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

from rules import ATOMIC_NUMBER
from common import one_hot


class MoleculeInputError(ValueError):
    """SMILES cannot be represented with the checkpoint's graph features."""


def atom_features(atom):
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hybrid = atom.GetHybridization()
    if hybrid not in hybridizations:
        hybrid = Chem.rdchem.HybridizationType.SP3
    return torch.tensor(
        one_hot(atom.GetAtomicNum(), ATOMIC_NUMBER())
        + one_hot(atom.GetDegree(), list(range(8)))
        + one_hot(atom.GetFormalCharge(), [-1, 0, 1, 2, 3, 4])
        + one_hot(hybrid, hybridizations)
        + [float(atom.GetIsAromatic())], dtype=torch.float32,
    )


def bond_features(bond):
    stereo = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        "R", "S",
    ]
    return torch.tensor(
        one_hot(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ])
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + one_hot(bond.GetStereo(), stereo), dtype=torch.float32,
    )


def smiles2graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise MoleculeInputError(f"Invalid SMILES: {smiles!r}")
    try:
        atoms = torch.stack([atom_features(atom) for atom in molecule.GetAtoms()])
        edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                 for bond in molecule.GetBonds()]
        edge_index = (torch.tensor(edges, dtype=torch.long).T.contiguous()
                      if edges else torch.empty((2, 0), dtype=torch.long))
        edge_attr = (torch.stack([bond_features(bond)
                                  for bond in molecule.GetBonds()])
                     if edges else torch.empty((0, 12), dtype=torch.float32))
    except ValueError as error:
        raise MoleculeInputError(str(error)) from error
    return Data(x=atoms, edge_index=edge_index, edge_attr=edge_attr,
                smiles=smiles)


def pathway_edges(info_path, assays):
    """Single-substrate pathway edges from the original inference algorithm.

    In a one-row pred2dataset, all normalized PMI edge values are 0.5 and
    all normalized odds-ratio values are 0.0. The topology still depends on
    assay target and rule metadata and is computed exactly as before.
    """
    info = pd.read_csv(info_path).set_index("Assay").loc[list(assays)]
    targets = info["Target"].tolist()
    rules = info["Rule"].astype(str).tolist()
    er = {"ERα", "ERα/ERβ", "ERβ"}
    ar = {"ARE", "Nrf2/ARE", "AR"}
    groups = ["ER" if target in er else
              "AR" if target in ar else "Other" for target in targets]
    ke_levels = [int(rule[2:]) for rule in rules
                 if rule.upper().startswith("KE") and rule[2:].isdigit()]
    ao_level = max(ke_levels, default=0) + 1
    levels = [0 if rule.upper() == "MIE" else
              int(rule[2:]) if rule.upper().startswith("KE")
              and rule[2:].isdigit() else ao_level for rule in rules]
    edges = []
    for first in range(len(assays)):
        for second in range(first + 1, len(assays)):
            same_group = groups[first] == groups[second]
            same_or_adjacent = (levels[first] == levels[second]
                                or abs(levels[first] - levels[second]) == 1)
            cross_mie = (levels[first] == levels[second] == 0
                         and {groups[first], groups[second]} == {"ER", "AR"})
            if (same_group and same_or_adjacent) or cross_mie:
                edges.extend(((first, second), (second, first)))
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    edge_attr = torch.tensor([[0.5, 0.0]] * len(edges), dtype=torch.float32)
    return edge_index, edge_attr


def prediction_graph(logits, edge_index, edge_attr):
    values = torch.as_tensor(logits, dtype=torch.float32)
    if values.ndim == 1:
        values = values[:, None]
    return Data(x=values, edge_index=edge_index, edge_attr=edge_attr)
