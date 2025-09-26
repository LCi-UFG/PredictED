import os
import torch
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader 

from tier2_loader import collate_graphs
from insight import (
    build_pred_graph,
    decode_symbol,
    prepare_positions,
    build_mol_graph,
    extract_attention,
    compute_importances,
    draw_base_graph,
    draw_topk_edges,
    leave1atom,
    leave1assay,
    refine_importance
    )


def view_attention(
    model,
    loader,
    device,
    output_dir,
    figure_size=(10,5),
    top_k=3,
    left_padding=0.05,
    right_padding=0.05,
    bottom_padding=0.05,
    top_padding=0.05,
    band_gap=0.25,
    bottom_ratio=0.4,
    target_bond_frac=0.06):

    os.makedirs(output_dir, exist_ok=True)
    model.to(device).eval()

    for idx, ((mol_graph, pred_graph), _) in enumerate(
        loader.dataset, start=1):

        batch = DataLoader(
            [((mol_graph, pred_graph), torch.tensor(0))],
            batch_size=1,
            collate_fn=collate_graphs
            )
        (mol_batch, pred_batch), _ = next(iter(batch))
        mol_batch, pred_batch = mol_batch.to(
            device), pred_batch.to(device)

        A_m2p, A_p2m, mask_m, mask_p = extract_attention(
            model, mol_batch, pred_batch
            )
        imp_lig = compute_importances(A_m2p)
        imp_pred = compute_importances(A_p2m)
        graph_pred = build_pred_graph(pred_graph)
        graph_mol, bond_orders = build_mol_graph(mol_graph)

        layouts = prepare_positions(
            mol_graph,
            graph_pred,
            left_padding,
            right_padding,
            bottom_padding,
            top_padding,
            band_gap,
            bottom_ratio,
            target_bond_frac
            )
        pos_pred = layouts['pos_pred']
        pos_mol  = layouts['pos_mol']
        x0, x1   = layouts['xlim']
        y0, y1   = layouts['ylim']

        labels_pred = {n: str(n) for n in graph_pred.nodes()}
        labels_mol  = {n: decode_symbol(
            mol_graph.x[n]) for n in graph_mol.nodes()
            }
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figure_size
            )
        for ax in (ax1, ax2):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.axis('off')
        fig.patch.set_alpha(0)

        cmap   = plt.cm.bwr
        v_lig  = imp_lig.max()  + 1e-8
        v_pred = imp_pred.max() + 1e-8

        draw_base_graph(
            ax1, 
            graph_pred, 
            pos_pred, 
            labels_pred,
            edge_color='lightgrey', 
            node_edge_color='black',
            )
        draw_base_graph(
            ax1, 
            graph_mol, 
            pos_mol, 
            labels_mol,
            bond_orders=bond_orders,
            node_color=imp_lig, 
            cmap=cmap,
            vmin=-v_lig, 
            vmax=v_lig,
            edge_color='black', 
            node_edge_color='black',
            )
        draw_topk_edges(
            ax1, A_m2p, mask_m, mask_p,
            pos_mol, pos_pred,
            imp_lig, cmap, v_lig, top_k
            )
        draw_base_graph(
            ax2, 
            graph_pred, 
            pos_pred, 
            labels_pred,
            node_color=imp_pred, 
            cmap=cmap,
            vmin=-v_pred, 
            vmax=v_pred,
            edge_color='lightgrey', 
            node_edge_color='black',

            )
        draw_base_graph(
            ax2, 
            graph_mol, 
            pos_mol, 
            labels_mol,
            bond_orders=bond_orders,
            edge_color='black', 
            node_edge_color='black',
            )
        draw_topk_edges(
            ax2, A_p2m, mask_p, mask_m,
            pos_pred, pos_mol,
            imp_pred, cmap, v_pred, top_k
            )
        plt.tight_layout(pad=0.3)
        fig.savefig(
            os.path.join(output_dir, f"sample_{idx}.svg"),
            format='svg', transparent=True, dpi=300,
            bbox_inches='tight', pad_inches=0.1
            )
        plt.show(fig)


def view_counterfactuality(
    model,
    loader,
    data_file,
    feature_file,
    device,
    output_dir,
    perturbations=10,
    task_index=0,
    perturb_column=0,
    cluster_alpha=0.5,
    smooth_alpha=0.5,
    figure_size=(5,5),
    left_padding=0.05,
    right_padding=0.05,
    bottom_padding=0.05,
    top_padding=0.05,
    band_gap=0.25,
    bottom_ratio=0.4,
    target_bond_frac=0.06):

    os.makedirs(output_dir, exist_ok=True)

    df_info  = pd.read_csv(
        feature_file).set_index('Assay')
    thresholds = torch.tensor(
        df_info['Threshold'].values, 
        dtype=torch.float32
        )
    df = pd.read_csv(data_file)
    tasks = df.columns[2:].tolist()
    task = tasks[task_index]
    task_dir = os.path.join(output_dir, task)
    os.makedirs(task_dir, exist_ok=True)

    model.to(device).eval()

    for idx, ((mol_graph, pred_graph), _ 
        ) in enumerate(loader.dataset, start=1):
        raw_lig   = leave1atom(
            model, mol_graph, pred_graph, device,
            perturbations=perturbations,
            task_index=task_index
            )
        means_p, poss_p, negs_p = leave1assay(
            model, mol_graph, pred_graph, device,
            thresholds, perturbations,
            task_index, perturb_column
            )
        raw_pred  = means_p * (poss_p - negs_p)

        imp_lig = refine_importance(
            raw_lig,
            mol_graph.edge_index,
            cluster_alpha,
            smooth_alpha
            )
        imp_pred = refine_importance(
            raw_pred,
            pred_graph.edge_index,
            cluster_alpha,
            smooth_alpha
            )
        imp_lig = (imp_lig - imp_lig.mean()
            ) / (imp_lig.std() + 1e-8)
        imp_pred = (imp_pred - imp_pred.mean()
            ) / (imp_pred.std() + 1e-8)
        graph_pred = build_pred_graph(pred_graph)
        graph_mol, bond_orders = build_mol_graph(mol_graph)

        layouts = prepare_positions(
            mol_graph,
            graph_pred,
            left_padding,
            right_padding,
            bottom_padding,
            top_padding,
            band_gap,
            bottom_ratio,
            target_bond_frac
            )
        pos_pred = layouts['pos_pred']
        pos_mol = layouts['pos_mol']
        x0, x1 = layouts['xlim']
        y0, y1 = layouts['ylim']
        labels_pred = {n: str(n) for n in graph_pred.nodes()}
        labels_mol  = {n: decode_symbol(
                mol_graph.x[n]) 
                for n in graph_mol.nodes()
                }
        fig, ax = plt.subplots(1, 1, figsize=figure_size)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axis('off')
        fig.patch.set_alpha(0)
        cmap = plt.cm.bwr
        v_lig = np.max(np.abs(imp_lig)) + 1e-8
        v_pred = np.max(np.abs(imp_pred)) + 1e-8

        draw_base_graph(
            ax,
            graph_pred,
            pos_pred,
            labels_pred,
            node_color=imp_pred,
            cmap=cmap,
            vmin=-v_pred,
            vmax=v_pred,
            edge_color='lightgrey',
            node_edge_color='black',
            edge_alpha=0.5
            )
        draw_base_graph(
            ax,
            graph_mol,
            pos_mol,
            labels_mol,
            bond_orders=bond_orders,
            node_color=imp_lig,
            cmap=cmap,
            vmin=-v_lig,
            vmax=v_lig,
            edge_color='black',
            node_edge_color='black',
            edge_alpha=0.6
            )
        plt.tight_layout(pad=0.3)
        fig.savefig(
            os.path.join(task_dir, f"sample_{idx}.svg"),
            format='svg',
            transparent=True,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1
            )
        plt.show(fig)