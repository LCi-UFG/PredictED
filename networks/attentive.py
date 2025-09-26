import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from layers import (
    AttentiveLayer,
    cross_attention_block
    )
from activation import get_activation


class AttentiveMod(nn.Module):
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
        num_timesteps,
        co_heads,
        num_tasks):

        super(AttentiveMod, self).__init__()

        self.edge_pred_dim = edge_pred
        self.num_timesteps = num_timesteps
        dims = [agg_hidden_dims[i] 
            for i in range(num_agg_layers)
            ]
        mol_in = [node_dim] + dims[:-1]
        pred_in = [node_pred] + dims[:-1]
        d_model = dims[-1]

        self.mol_att_layers = nn.ModuleList([
            AttentiveLayer(
                mol_in[i], 
                dims[i], 
                edge_dim, 
                dropout_rate)
            for i in range(num_agg_layers)
            ])
        self.pred_att_layers = nn.ModuleList([
            AttentiveLayer(
                pred_in[i], 
                dims[i], 
                edge_pred, 
                dropout_rate)
            for i in range(num_agg_layers)
            ])
        self.cross_mol2pred = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=co_heads,
            batch_first=True, dropout=dropout_rate
            )
        self.cross_pred2mol = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=co_heads,
            batch_first=True, 
            dropout=dropout_rate
            )
        self.norm_mol = nn.LayerNorm(d_model)
        self.norm_pred = nn.LayerNorm(d_model)
        self.ff_mol = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout_rate)
            )
        self.ff_pred = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout_rate)
            )
        self.readout_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, 1)
            )
        self.mol_gru = nn.GRUCell(d_model, d_model)

        combined_dim = 2 * d_model
        self.final_mlp = nn.ModuleList()
        in_dim = combined_dim
        for i in range(num_lin_layers):
            out_d = lin_hidden_dims[i]
            self.final_mlp.extend([
                nn.Linear(in_dim, out_d),
                nn.LayerNorm(out_d),
                get_activation(activation),
                nn.Dropout(dropout_rate)]
                )
            in_dim = out_d

        self.embedding_layer = nn.Linear(
            in_dim, in_dim
            )
        self.output_layer = nn.Linear(
            in_dim, num_tasks
            )
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
        for layer in self.mol_att_layers:
            x = layer(x, edge_idx, edge_attr, batch)

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
        for layer in self.pred_att_layers:
            xp = layer(xp, edge_idx_p, edge_attr_p, batch_p)
    
        mol_seq, mol_mask = to_dense_batch(x, batch)
        pred_seq, pred_mask = to_dense_batch(xp, batch_p)
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

        # ---- Attentive readout com masked softmax (invariante ao tamanho) ----
        scores = self.readout_mlp(mol_seq).squeeze(-1)
        scores = scores.masked_fill(~mol_mask, float('-inf'))
        alpha  = torch.softmax(scores, dim=1).unsqueeze(-1)
        mol_fp = (alpha * mol_seq).sum(dim=1).relu_()

        for _ in range(self.num_timesteps):
            mol_fp = self.mol_gru(
                mol_fp, mol_fp).relu_()

        scores_p = self.readout_mlp(pred_seq).squeeze(-1)
        scores_p = scores_p.masked_fill(~pred_mask, float('-inf'))
        alpha_p  = torch.softmax(scores_p, dim=1).unsqueeze(-1)
        pred_fp  = (alpha_p * pred_seq).sum(dim=1).relu_()

        for _ in range(self.num_timesteps):
            pred_fp = self.mol_gru(
                pred_fp, pred_fp).relu_()

        h = torch.cat([mol_fp, pred_fp], dim=-1)
        for layer in self.final_mlp:
            h = layer(h)
        penultimate = h.clone()
        out = self.output_layer(
            self.embedding_layer(h)
            )

        if save_embeddings:
            self.saved_embeddings.append(
                penultimate.detach().cpu()
                )
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