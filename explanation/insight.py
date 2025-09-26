import copy
import torch
import random
import networkx as nx
import matplotlib as mpl
import numpy as np
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import rdDepictor
from torch_geometric.loader import DataLoader
from torch.quasirandom import SobolEngine
from networkx.drawing.nx_pydot import pydot_layout
from torch_geometric.utils import to_scipy_sparse_matrix

from rules import (
    ATOMIC_NUMBER, 
    ATOMIC_SUBSTITUTIONS,
    CHARGE_SUBSTITUTIONS, 
    HYBRIDIZATION_SUBSTITUTIONS
    )
from tier2_loader import collate_graphs
from predictor import predict_tier2


def get_layout(G):
    pos_raw = pydot_layout(G, prog='sfdp')
    xs, ys = zip(*pos_raw.values())
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = (max_x - min_x) or 1.0
    dy = (max_y - min_y) or 1.0

    positions = {
        n: ((x - min_x) / dx, (y - min_y) / dy)
        for n, (x, y) in pos_raw.items()
    }

    return positions


def decode_symbol(x_feat):
    atomic_numbers = ATOMIC_NUMBER()
    periodic = {
        1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si',
        15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca',
        22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe',
        27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 33: 'As',
        34: 'Se', 35: 'Br', 53: 'I'
        }
    vec = x_feat[:len(atomic_numbers)]
    idx = vec.argmax().item()
    atomic_number = atomic_numbers[idx]
    symbol = periodic.get(
        atomic_number, str(atomic_number)
        )
    
    return symbol


def prepare_positions(
    mol_graph,
    graph_pred,
    left_padding,
    right_padding,
    bottom_padding,
    top_padding,
    band_gap,
    bottom_ratio,
    target_bond_frac):

    graph_mol, _ = build_mol_graph(mol_graph)
    total_v = 1 - bottom_padding - top_padding - band_gap
    bottom_h = total_v * bottom_ratio
    top_h = total_v * (1 - bottom_ratio)
    mol_offset = bottom_padding
    pred_offset = mol_offset + bottom_h + band_gap
    usable_w = 1 - left_padding - right_padding
    x0, x1 = -left_padding, 1 + right_padding
    y0, y1 = -bottom_padding, 1 + top_padding
    raw_pred = get_layout(graph_pred)
    xsr, ysr = zip(*raw_pred.values())
    dxr = (max(xsr) - min(xsr)) or 1.0
    dyr = (max(ysr) - min(ysr)) or 1.0
    pos_pred = {
        n: (left_padding + (x - min(xsr)) / dxr * usable_w,
            pred_offset + (y - min(ysr)) / dyr * top_h)
        for n, (x, y) in raw_pred.items()
        }
    mol = Chem.MolFromSmiles(mol_graph.smiles)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    raw_pos = {
        atom.GetIdx(): np.array([
            conf.GetAtomPosition(atom.GetIdx()).x,
            conf.GetAtomPosition(atom.GetIdx()).y])
        for atom in mol.GetAtoms()
        }
    dists = [
        np.linalg.norm(raw_pos[u] - raw_pos[v])
        for u, v in graph_mol.edges()
        ]
    avg_bond = float(np.mean(dists)) if dists else 1.0
    target_bond = usable_w * target_bond_frac
    scale = target_bond / avg_bond
    scaled = {n: raw_pos[n] * scale for n in raw_pos}
    xs, ys = zip(*scaled.values())
    mol_w = max(xs) - min(xs)
    mol_h = max(ys) - min(ys)
    x_off = left_padding + (
        usable_w - mol_w) / 2 - min(xs)
    y_off = mol_offset   + (
        bottom_h - mol_h) / 2 - min(ys)
    pos_mol = {
        n: (scaled[n][0] + x_off, scaled[n][1] + y_off)
        for n in scaled
        }
    layouts = {
        'pos_pred': pos_pred,
        'pos_mol': pos_mol,
        'xlim': (x0, x1),
        'ylim': (y0, y1)
        }
    
    return layouts

def build_pred_graph(data):
    graph = nx.Graph()
    node_count = data.x.size(0)
    graph.add_nodes_from(range(node_count))
    edges_array = data.edge_index.cpu().numpy().T
    for src, dst in edges_array:
        graph.add_edge(int(src), int(dst))

    return graph


def build_mol_graph(data):
    graph = nx.Graph()
    bond_orders = {}
    aromatic_edges = []

    edges_array = data.edge_index.cpu().numpy().T
    node_count = data.x.size(0)
    graph.add_nodes_from(range(node_count))

    for idx, (src, dst) in enumerate(edges_array):
        src, dst = int(src), int(dst)
        graph.add_edge(src, dst)
        one_hot = data.edge_attr[idx][:4].cpu().numpy()
        bond_type = one_hot.argmax()
        if bond_type == 3:
            aromatic_edges.append((src, dst))
            order = 1.0
        else:
            order = {0: 1.0, 1: 2.0, 2: 3.0}[bond_type]
        bond_orders[(src, dst)] = bond_orders[(dst, src)] = order

    for cycle in nx.cycle_basis(graph):
        if 5 <= len(cycle) <= 7:
            cycle_nodes = cycle + [cycle[0]]
            aromatic_cycle = [(u, v)
                for u, v in zip(cycle_nodes, cycle_nodes[1:])
                if (u, v) in aromatic_edges or (v, u) in aromatic_edges
                ]
            for idx, (u, v) in enumerate(aromatic_cycle):
                order = 2.0 if idx % 2 == 0 else 1.0
                bond_orders[(u, v)] = bond_orders[(v, u)] = order

    return graph, bond_orders


def draw_base_graph(
    ax, graph, positions, labels,
    bond_orders=None,
    node_color=None, 
    cmap=None, 
    vmin=None, 
    vmax=None,
    edge_color='k', 
    node_edge_color='k', 
    edge_alpha=1.0):
  
    if bond_orders is None:
        edges = nx.draw_networkx_edges(
            graph, positions, ax=ax,
            edge_color=edge_color, 
            alpha=edge_alpha
            )
        edges.set_zorder(1)
    else:
        for u, v in graph.edges():
            order = bond_orders.get((u, v), 1.0)
            lines = int(order)
            p1 = np.array(positions[u])
            p2 = np.array(positions[v])
            dx, dy = p2 - p1
            length = np.hypot(dx, dy)
            if length < 1e-6:
                offsets = [(0, 0)] * lines
            else:
                ux, uy = -dy / length, dx / length
                delta = 0.015 / lines
                offsets = [
                    ((i - (lines - 1) / 2) * 2 * delta * ux,
                     (i - (lines - 1) / 2) * 2 * delta * uy)
                    for i in range(lines)
                    ]
            for xo, yo in offsets:
                line, = ax.plot(
                    [p1[0] + xo, p2[0] + xo],
                    [p1[1] + yo, p2[1] + yo],
                    color=edge_color, 
                    alpha=edge_alpha, 
                    linewidth=1.2)
                line.set_zorder(1)

    nodes = list(graph.nodes())
    if node_color is not None and cmap is not None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = [cmap(norm(node_color[i])) for i in nodes]
    else:
        colors = ['lightgray'] * len(nodes)

    nodes_drawn = nx.draw_networkx_nodes(
        graph, positions, nodelist=nodes,
        node_color=colors,
        edgecolors=node_edge_color,
        node_size=120,
        ax=ax
        )
    nodes_drawn.set_zorder(2)

    for idx, node in enumerate(nodes):
        x, y = positions[node]
        rgba = mpl.colors.to_rgba(colors[idx])
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = 'white' if lum < 0.5 else 'black'
        ax.text(
            x, y, labels.get(node, ''),
            color=text_color,
            ha='center', va='center',
            fontsize=8, zorder=3
            )

    return ax


def draw_topk_edges(
    ax, 
    attention_matrix, 
    mask_src, 
    mask_tgt,
    positions_src, 
    positions_tgt,
    importances, 
    cmap, 
    vlim, 
    top_k):

    top_indices = np.argsort(-importances)[:top_k]
    src_indices = np.nonzero(mask_src)[0]
    tgt_indices = np.nonzero(mask_tgt)[0]

    for src_i in top_indices:
        weights = attention_matrix[src_i]
        best_targets = np.argsort(-weights)[:top_k]

        for tgt_j in best_targets:
            src_node = src_indices[src_i]
            tgt_node = tgt_indices[tgt_j]
            weight = attention_matrix[src_i, tgt_j]
            color = cmap((weight + vlim) / (2 * vlim))

            line, = ax.plot(
                [positions_src[src_node][0], 
                 positions_tgt[tgt_node][0]],
                [positions_src[src_node][1], 
                 positions_tgt[tgt_node][1]],
                color=color, linewidth=1.0, alpha=0.4)
            line.set_zorder(2.5)

    result = ax

    return result


def mask_node_lig(data_raw, idx, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    data = copy.deepcopy(data_raw)
    atomic_nums = ATOMIC_NUMBER()

    one_hot = data.x[idx, :len(atomic_nums)].cpu().numpy()
    original = atomic_nums[one_hot.argmax()]
    subs_atoms = ATOMIC_SUBSTITUTIONS.get(original, [original])
    new_atom = np.random.choice(subs_atoms)
    atom_idx = atomic_nums.index(new_atom)
    data.x[idx, :len(atomic_nums)] = 0
    data.x[idx, atom_idx] = 1

    c0, c1 = len(atomic_nums), len(atomic_nums) + 4
    old_charge = data.x[idx, c0:c1].cpu().numpy().argmax()
    subs_charge = CHARGE_SUBSTITUTIONS.get(new_atom, [old_charge])
    new_charge = np.random.choice(subs_charge)
    data.x[idx, c0:c1] = 0
    data.x[idx, c0 + subs_charge.index(new_charge)] = 1

    h0, h1 = c1, c1 + 5
    old_hybrid = data.x[idx, h0:h1].cpu().numpy().argmax()
    subs_hybrid = HYBRIDIZATION_SUBSTITUTIONS.get(
        new_atom, {'types': [old_hybrid]})['types']
    new_hybrid = np.random.choice(subs_hybrid)
    data.x[idx, h0:h1] = 0
    data.x[idx, h0 + subs_hybrid.index(new_hybrid)] = 1

    return data.to(data_raw.x.device)


def leave1atom(
    model, 
    mol_raw, 
    pred_raw, 
    device,
    perturbations=10, 
    task_index=0):
    
    model.to(device).eval()

    mol0 = copy.deepcopy(mol_raw).to(device)
    pred0 = copy.deepcopy(pred_raw).to(device)
    pred0.y = torch.zeros((1,), device=device)
    mol0.y = pred0.y

    p0_all, _, _ = predict_tier2(model,
        DataLoader([((mol0, pred0), pred0.y)],
            batch_size=1,
            collate_fn=collate_graphs),
        device,
        return_embeddings=False)
    baseline = float(p0_all[0, task_index])

    n_atoms = mol_raw.x.size(0)
    means = np.zeros(n_atoms)
    poss = np.zeros(n_atoms)
    negs = np.zeros(n_atoms)

    sobol = SobolEngine(1, scramble=True)
    sequences = sobol.draw(n_atoms * perturbations).view(
        n_atoms, perturbations).numpy()

    for atom_idx in range(n_atoms):
        deltas = []
        for v in sequences[atom_idx]:
            seed = int((v * (2**32 - 1)).item())
            pert_mol = mask_node_lig(
                mol_raw, atom_idx, seed).to(device)
            pert_pred = copy.deepcopy(pred_raw).to(device)
            pert_mol.y = pert_pred.y = pred0.y

            p2_all, _, _ = predict_tier2(model,
                DataLoader([((pert_mol, pert_pred), pert_pred.y)],
                    batch_size=1,
                    collate_fn=collate_graphs),
                device,
                return_embeddings=False)
            deltas.append(baseline - float(p2_all[0, task_index]))

        sorted_deltas = np.sort(deltas)
        k = int(len(sorted_deltas) * 0.1)
        if len(sorted_deltas) > 2 * k:
            sorted_deltas = sorted_deltas[k:-k]

        means[atom_idx] = sorted_deltas.mean()
        poss[atom_idx] = np.mean(sorted_deltas > 0)
        negs[atom_idx] = np.mean(sorted_deltas < 0)

    importances = means * (poss - negs)
    result = (importances - importances.mean()
        ) / (importances.std() + 1e-8)

    return result


def mask_node_assay(
    data_raw, 
    idx, 
    new_value, 
    column, 
    threshold=None):

    data = copy.deepcopy(data_raw)

    if column == 0:
        data.x[idx, 0] = new_value
    elif column == 1:
        if threshold is None:
            raise ValueError("Threshold is required for column 1")
        data.x[idx, 1] = 1.0 if new_value >= threshold else 0.0
    else:
        raise ValueError("Column must be 0 or 1")

    return data


def leave1assay(
    model,
    mol_raw,
    pred_raw,
    device,
    thresholds,
    perturbations=10,
    task_index=0,
    perturb_column=0):

    model.to(device).eval()

    mol0 = copy.deepcopy(mol_raw).to(device)
    pred0 = copy.deepcopy(pred_raw).to(device)
    pred0.y = torch.zeros((1,), device=device)
    mol0.y = pred0.y

    p0_all, _, _ = predict_tier2(
        model,
        DataLoader(
            [((mol0, pred0), pred0.y)],
            batch_size=1,
            collate_fn=collate_graphs),
        device, return_embeddings=False
        )
    baseline = float(p0_all[0, task_index])

    n_assays = pred_raw.x.size(0)
    means = np.zeros(n_assays)
    poss = np.zeros(n_assays)
    negs = np.zeros(n_assays)

    sobol = SobolEngine(1, scramble=True)
    sequences = sobol.draw(n_assays * perturbations).view(
        n_assays, perturbations).numpy()

    for assay_idx in range(n_assays):
        batch = []
        for v in sequences[assay_idx]:
            pert = mask_node_assay(
                pred_raw, assay_idx, float(v),
                column=perturb_column,
                threshold=(thresholds[task_index]
                    if perturb_column == 1 else None)).to(device)
            pert.y = pred0.y
            batch.append(((mol0, pert), pert.y))

        p2_all, _, _ = predict_tier2(
            model,
            DataLoader(
                batch,
                batch_size=perturbations,
                collate_fn=collate_graphs),
            device, return_embeddings=False
            )
        deltas = baseline - p2_all[:, task_index]
        sorted_deltas = np.sort(deltas)
        k = int(len(sorted_deltas) * 0.1)
        if len(sorted_deltas) > 2 * k:
            sorted_deltas = sorted_deltas[k:-k]

        means[assay_idx] = sorted_deltas.mean()
        poss[assay_idx] = np.mean(sorted_deltas > 0)
        negs[assay_idx] = np.mean(sorted_deltas < 0)

    return means, poss, negs


def extract_attention(
    model, 
    mol_batch, 
    pred_batch):

    outputs = model((mol_batch, pred_batch), 
                    return_attention=True)
    _, att_m2p_t, att_p2m_t, mask_m_t, mask_p_t = outputs

    att_m2p = att_m2p_t[0].detach().cpu().numpy()
    att_p2m = att_p2m_t[0].detach().cpu().numpy()
    mask_m = mask_m_t[0].cpu().numpy().astype(bool)
    mask_p = mask_p_t[0].cpu().numpy().astype(bool)

    if att_m2p.shape[0] != mask_m.sum():
        att_m2p = att_m2p.T
    if att_p2m.shape[0] != mask_p.sum():
        att_p2m = att_p2m.T

    attention = (
        att_m2p[mask_m][:, mask_p],
        att_p2m[mask_p][:, mask_m],
        mask_m,
        mask_p
        )
    return attention


def compute_importances(attention_matrix):
    row_means = attention_matrix.mean(axis=1)
    importances = row_means - row_means.mean()

    return importances


def cluster_predictions(scores, edge_index):
    n = scores.shape[0]
    pos = np.where(scores >= 0)[0]
    neg = np.where(scores < 0)[0]
    edges = edge_index.cpu().numpy().T.tolist()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    clusters = []
    for idx in (pos, neg):
        if idx.size:
            for comp in nx.connected_components(
                G.subgraph(idx)):
                clusters.append(list(comp))

    return clusters


def refine_importance(
    raw,
    edge_idx,
    cluster_alpha,
    smooth_alpha):

    if isinstance(raw, torch.Tensor):
        raw = raw.cpu().numpy()
    n = raw.shape[0]
    if n < 5:
        return raw
    clusters = cluster_predictions(raw, edge_idx)
    imp = np.zeros_like(raw)
    for comp in clusters:
        vals = raw[comp]
        m = vals.mean()
        imp[comp] = m + cluster_alpha * (vals - m)
    A = to_scipy_sparse_matrix(edge_idx, num_nodes=n)
    deg = np.array(A.sum(axis=1)).flatten()
    Dinv = sp.diags(1.0 / (deg + 1e-8))
    smooth = Dinv.dot(A.dot(imp))
    importance =  smooth_alpha * imp + (
        1 - smooth_alpha) * smooth

    return importance