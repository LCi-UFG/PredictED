import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn import (
    MessagePassing, 
    GATv2Conv, 
    GraphNorm
    )

class AttentiveLayer(MessagePassing):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        edge_dim, 
        dropout_rate):
        
        super(AttentiveLayer, self).__init__(aggr='add')
        self.node_proj = nn.Linear(input_dim, output_dim
            ) if input_dim != output_dim else nn.Identity()
        self.layer_norm = GraphNorm(output_dim)
        self.edge_encoder = nn.Linear(
            edge_dim, output_dim
            )
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )
        attn_input_dim = 3 * output_dim
        self.attn_mlp = nn.Sequential(
            nn.Linear(attn_input_dim, 1),
            nn.LeakyReLU(0.1)
            )
        self.gru = nn.GRUCell(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def message(
        self, 
        x_i, x_j, 
        edge_attr, 
        index, ptr, 
        size_i):

        attn_input = torch.cat(
            [x_i, x_j, edge_attr], dim=-1
            )
        attn_scores = self.attn_mlp(attn_input)
        attn_weights = softmax(
            attn_scores, index, ptr, size_i
            )
        message = self.msg_mlp(
            torch.cat([x_j, edge_attr], dim=-1)
            )
        
        return message * attn_weights
    
    def forward(
        self, x, 
        edge_index, 
        edge_attr,
        batch):
        
        x_proj = self.node_proj(x)
        edge_enc = self.edge_encoder(edge_attr)
        aggr_out = self.propagate(
            edge_index=edge_index, 
            x=x_proj, 
            edge_attr=edge_enc
            )
        out = self.gru(aggr_out, x_proj)
        out = self.layer_norm(out + x_proj, batch)
        out = self.dropout(out)

        return out                        


class GATLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        heads,
        dropout,
        concat=True):

        super(GATLayer, self).__init__()
        self.conv = GATv2Conv(
            in_channels=input_dim,
            out_channels=(
                output_dim // heads
                if concat else output_dim
                ),
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=concat
            )
        self.res_connection = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
            )
        self.norm = GraphNorm(output_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x,
        edge_index,
        edge_attr,
        batch):

        residual = self.res_connection(x)
        out = self.conv(x, edge_index, edge_attr=edge_attr)

        out = out + residual
        out = self.norm(out, batch)
        out = self.activation(out)
        out = self.dropout(out)

        return out


class MPNNLayer(MessagePassing):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        dropout_rate):

        super(MPNNLayer, self).__init__(aggr='add')
 
        self.message_proj = nn.Sequential(
            nn.Linear(input_dim + edge_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )
        self.node_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
            )
        self.update_net = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim)
            )
        self.norm = GraphNorm(output_dim)

    def message(self, x_j, edge_attr):
        m = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_proj(m)

    def update(self, aggr_out, x, batch):
        h = self.node_proj(x)
        cat = torch.cat([aggr_out, h], dim=-1)
        new_h = self.update_net(cat)
        out = new_h + h
        out = self.norm(out, batch)
        return out

    def forward(self, x, edge_index, edge_attr, batch):
        return self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            batch=batch
            )


def cross_attention_block(
    mol_seq, 
    pred_seq,
    cross_mol2pred, 
    cross_pred2mol,
    norm_mol, 
    ff_mol,
    norm_pred, 
    ff_pred,
    mol_mask, 
    pred_mask):

    pred_ctx, attn_l2p = cross_mol2pred(
        query=pred_seq,
        key=mol_seq,
        value=mol_seq,
        key_padding_mask=~mol_mask,
        need_weights=True,
        average_attn_weights=True
        )
    pred_seq = norm_pred(pred_seq + pred_ctx)
    pred_seq = pred_seq + ff_pred(pred_seq)

    mol_ctx, attn_p2l = cross_pred2mol(
        query=mol_seq,
        key=pred_seq,
        value=pred_seq,
        key_padding_mask=~pred_mask,
        need_weights=True,
        average_attn_weights=True
        )
    mol_seq = norm_mol(mol_seq + mol_ctx)
    mol_seq  = mol_seq + ff_mol(mol_seq)

    return mol_seq, pred_seq, attn_l2p, attn_p2l
