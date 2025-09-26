import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import (
    global_add_pool,
    JumpingKnowledge
    )

from layers import (
    MPNNLayer,
    cross_attention_block
    )
from activation import get_activation


class MPNNMod(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        node_pred,
        edge_pred,
        agg_hidden_dims,
        num_agg_layers,
        lin_hidden_dims,
        num_lin_layers,
        activation,
        dropout_rate,
        co_heads,
        num_tasks):

        super(MPNNMod, self).__init__()

        self.edge_dim = edge_dim
        self.edge_pred_dim = edge_pred

        dims = [agg_hidden_dims[i] 
            for i in range(num_agg_layers)
            ]
        mol_in = [node_dim] + dims[:-1]
        pred_in = [node_pred] + dims[:-1]
        d_model = dims[-1]
 
        self.mol_agg_layers = nn.ModuleList([
            MPNNLayer(
                mol_in[i],
                dims[i],
                edge_dim,
                dropout_rate)
            for i in range(num_agg_layers)
            ])
        self.mol_jk = JumpingKnowledge(mode='cat')
        self.mol_virtualnode_embedding = nn.ParameterList([
                nn.Parameter(torch.zeros(1, mol_in[i]))
            for i in range(num_agg_layers)
            ])
        self.mol_virtualnode_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], mol_in[i]),
                nn.LayerNorm(mol_in[i]),
                nn.ReLU(),
                nn.Linear(mol_in[i], mol_in[i]))
            for i in range(num_agg_layers)
            ])
        self.mol_node_proj = nn.Linear(sum(dims), d_model)

        self.pred_agg_layers = nn.ModuleList([
            MPNNLayer(
                pred_in[i],
                dims[i],
                edge_pred,
                dropout_rate)
            for i in range(num_agg_layers)
            ])
        self.pred_jk = JumpingKnowledge(mode='cat')
        self.pred_virtualnode_embedding = nn.ParameterList([
                nn.Parameter(torch.zeros(1, pred_in[i]))
            for i in range(num_agg_layers)
            ])
        self.pred_virtualnode_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], pred_in[i]),
                nn.LayerNorm(pred_in[i]),
                nn.ReLU(),
                nn.Linear(pred_in[i], pred_in[i]))
            for i in range(num_agg_layers)
            ])
        self.pred_node_proj = nn.Linear(sum(dims), d_model)

        self.cross_mol2pred = nn.MultiheadAttention(
            embed_dim=d_model,
            kdim=d_model,
            vdim=d_model,
            num_heads=co_heads,
            batch_first=True,
            dropout=dropout_rate
            )
        self.cross_pred2mol = nn.MultiheadAttention(
            embed_dim=d_model,
            kdim=d_model,
            vdim=d_model,
            num_heads=co_heads,
            batch_first=True,
            dropout=dropout_rate
            )
        self.norm_mol = nn.LayerNorm(d_model)
        self.norm_pred = nn.LayerNorm(d_model)
        self.ff_mol = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout_rate)
            )
        self.ff_pred = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout_rate)
            )
        combined_dim = 2 * d_model
        self.final_mlp = nn.ModuleList()
        in_dim = combined_dim
        for i in range(num_lin_layers):
            out_dim = lin_hidden_dims[i]
            self.final_mlp.append(nn.Linear(in_dim, out_dim))
            self.final_mlp.append(nn.LayerNorm(out_dim))
            self.final_mlp.append(get_activation(activation))
            self.final_mlp.append(nn.Dropout(dropout_rate))
            in_dim = out_dim

        self.output_layer = nn.Linear(in_dim, num_tasks)
        self.saved_embeddings = []

    def forward(
        self,
        data,
        save_embeddings=False,
        return_penultimate=False,
        return_attention=False): 

        mol_data, pred_data = data
        x, edge_idx, edge_attr, batch = (
            mol_data.x,
            mol_data.edge_index,
            mol_data.edge_attr,
            mol_data.batch
            )
        B = int(batch.max()) + 1
        mol_vnodes = [
            emb.expand(B, -1)
            for emb in self.mol_virtualnode_embedding
            ]
        mol_xs = []
        for i, layer in enumerate(self.mol_agg_layers):
            x = layer(x + mol_vnodes[i][batch], 
                edge_idx, 
                edge_attr, 
                batch
                )
            mol_xs.append(x)
            pooled = global_add_pool(x, batch)
            mol_vnodes[i] = mol_vnodes[
                i] + self.mol_virtualnode_mlp[i](pooled)
        x = self.mol_jk(mol_xs)
        node_mol = self.mol_node_proj(x)

        xp, edge_idx_p, edge_attr_p, batch_p = (
            pred_data.x,
            pred_data.edge_index,
            pred_data.edge_attr,
            pred_data.batch
            )
        if edge_attr_p.dim() == 1:
            edge_attr_p = edge_attr_p.view(
                -1, self.edge_pred_dim
                )
        Bp = int(batch_p.max()) + 1
        pred_vnodes = [
            emb.expand(Bp, -1)
            for emb in self.pred_virtualnode_embedding
            ]
        pred_xs = []
        for i, layer in enumerate(self.pred_agg_layers):
            xp = layer(xp + pred_vnodes[i][batch_p],
                    edge_idx_p, 
                    edge_attr_p, 
                    batch_p
                    )
            pred_xs.append(xp)
            pooled_p = global_add_pool(xp, batch_p)
            pred_vnodes[i] = pred_vnodes[i
                    ] + self.pred_virtualnode_mlp[i](pooled_p)
        xp = self.pred_jk(pred_xs)
        node_pred = self.pred_node_proj(xp)
        mol_seq, mol_mask = to_dense_batch(node_mol, batch)
        pred_seq, pred_mask = to_dense_batch(node_pred, batch_p)
        mol_seq, pred_seq, attn_m2p, attn_p2m = cross_attention_block(
            mol_seq,
            pred_seq,
            self.cross_mol2pred,
            self.cross_pred2mol,
            self.norm_mol,
            self.ff_mol,
            self.norm_pred,
            self.ff_pred,
            mol_mask,
            pred_mask
            )
        mol_emb  = (mol_seq  * mol_mask.unsqueeze(-1).float()
            ).sum(dim=1) / mol_mask.sum(
                dim=1, keepdim=True).clamp_min(1)
        pred_emb = (pred_seq * pred_mask.unsqueeze(-1).float()
            ).sum(dim=1) / pred_mask.sum(
                dim=1, keepdim=True).clamp_min(1)
        h = torch.cat([mol_emb, pred_emb], dim=-1)

        for layer in self.final_mlp:
            h = layer(h)
        penultimate = h.clone()
        out = self.output_layer(h)

        if save_embeddings:
            self.saved_embeddings.append(
                penultimate.detach().cpu())
        if return_attention:
            return (
                out,
                attn_m2p,
                attn_p2m,
                mol_mask,
                pred_mask
            )
        if return_penultimate:
            return penultimate
        
        return out