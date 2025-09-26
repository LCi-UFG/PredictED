import torch.nn as nn

from initialization import (
    attentive_weights,
    gat_weights,
    gin_weights,
    mpnn_weights
    )
from utils import set_seed
         

def attentive_resets(model, seed=42):
    set_seed(seed)
    for layer in list(model.mol_att_layers):
        attentive_weights(layer.node_proj)
        attentive_weights(layer.layer_norm)
        attentive_weights(layer.edge_encoder)
        attentive_weights(layer.msg_mlp)
        attentive_weights(layer.attn_mlp)
        if hasattr(layer, 'gru'
            ) and hasattr(layer.gru, 'reset_parameters'):
            set_seed(seed)  
            layer.gru.reset_parameters()

    set_seed(seed + 1337)
    for layer in list(model.pred_att_layers):
        attentive_weights(layer.node_proj)
        attentive_weights(layer.layer_norm)
        attentive_weights(layer.edge_encoder)
        attentive_weights(layer.msg_mlp)
        attentive_weights(layer.attn_mlp)
        if hasattr(layer, 'gru'
            ) and hasattr(layer.gru, 'reset_parameters'):
            set_seed(seed + 1337)
            layer.gru.reset_parameters()

    set_seed(seed + 9001)
    if isinstance(model.cross_mol2pred, nn.MultiheadAttention):
        model.cross_mol2pred._reset_parameters()
    if isinstance(model.cross_pred2mol, nn.MultiheadAttention):
        model.cross_pred2mol._reset_parameters()

    attentive_weights(model.norm_mol)
    attentive_weights(model.ff_mol)
    attentive_weights(model.norm_pred)
    attentive_weights(model.ff_pred)
    attentive_weights(model.readout_mlp)

    if hasattr(model.mol_gru, 'reset_parameters'):
        set_seed(seed + 42)
        model.mol_gru.reset_parameters()

    attentive_weights(model.final_mlp)
    attentive_weights(model.embedding_layer)
    attentive_weights(model.output_layer)


def gat_resets(model, seed=42):
    set_seed(seed)
    for layer in model.mol_agg_layers:
        gat_weights(layer.conv)
        if hasattr(layer, 'conv_norm'):
            gat_weights(layer.conv_norm)
        if hasattr(layer, 'feature_projection'):
            gat_weights(layer.feature_projection)
        if hasattr(layer, 'control_gate'):
            gat_weights(layer.control_gate)
        if hasattr(layer, 'post_mlp'):
            gat_weights(layer.post_mlp)
        if hasattr(layer, 'post_norm'):
            gat_weights(layer.post_norm)
    for mlp in model.mol_virtualnode_mlp:
        gat_weights(mlp)
    for p in model.mol_virtualnode_embedding:
        nn.init.zeros_(p)

    set_seed(seed + 1337)
    for layer in model.pred_agg_layers:
        gat_weights(layer.conv)
        if hasattr(layer, 'conv_norm'):
            gat_weights(layer.conv_norm)
        if hasattr(layer, 'feature_projection'):
            gat_weights(layer.feature_projection)
        if hasattr(layer, 'control_gate'):
            gat_weights(layer.control_gate)
        if hasattr(layer, 'post_mlp'):
            gat_weights(layer.post_mlp)
        if hasattr(layer, 'post_norm'):
            gat_weights(layer.post_norm)
    for mlp in model.pred_virtualnode_mlp:
        gat_weights(mlp)
    for p in model.pred_virtualnode_embedding:
        nn.init.zeros_(p)

    set_seed(seed + 9001)
    if isinstance(model.cross_mol2pred, nn.MultiheadAttention):
        model.cross_mol2pred._reset_parameters()
    if isinstance(model.cross_pred2mol, nn.MultiheadAttention):
        model.cross_pred2mol._reset_parameters()

    gat_weights(model.norm_mol)
    gat_weights(model.ff_mol)
    gat_weights(model.norm_pred)
    gat_weights(model.ff_pred)
    gat_weights(model.mol_node_proj)
    gat_weights(model.pred_node_proj)
    gat_weights(model.final_mlp)
    gat_weights(model.output_layer)


def gin_resets(model, seed=42):
    set_seed(seed)
    for layer in model.mol_agg_layers:
        gin_weights(layer.conv)
        gin_weights(layer.feature_projection)
        gin_weights(layer.control_gate)
        gin_weights(layer.post_mlp)
        gin_weights(layer.post_norm)
    for mlp in model.mol_virtualnode_mlp:
        gin_weights(mlp)
    for p in model.mol_virtualnode_embedding:
        nn.init.zeros_(p)

    set_seed(seed + 1337)
    for layer in model.pred_agg_layers:
        gin_weights(layer.conv)
        gin_weights(layer.feature_projection)
        gin_weights(layer.control_gate)
        gin_weights(layer.post_mlp)
        gin_weights(layer.post_norm)
    for mlp in model.pred_virtualnode_mlp:
        gin_weights(mlp)
    for p in model.pred_virtualnode_embedding:
        nn.init.zeros_(p)

    set_seed(seed + 9001)
    if isinstance(model.cross_mol2pred, nn.MultiheadAttention):
        model.cross_mol2pred._reset_parameters()
    if isinstance(model.cross_pred2mol, nn.MultiheadAttention):
        model.cross_pred2mol._reset_parameters()

    gin_weights(model.norm_mol)
    gin_weights(model.ff_mol)
    gin_weights(model.norm_pred)
    gin_weights(model.ff_pred)
    gin_weights(model.mol_node_proj)
    gin_weights(model.pred_node_proj)
    gin_weights(model.final_mlp)
    gin_weights(model.output_layer)


def mpnn_resets(model, seed=42):
    set_seed(seed)
    for layer in model.mol_agg_layers:
        mpnn_weights(layer.message_proj)
        mpnn_weights(layer.node_proj)
        mpnn_weights(layer.update_net)
        mpnn_weights(layer.norm)
    for mlp in model.mol_virtualnode_mlp:
        mpnn_weights(mlp)
    for p in model.mol_virtualnode_embedding:
        nn.init.zeros_(p)

    set_seed(seed + 1337)
    for layer in model.pred_agg_layers:
        mpnn_weights(layer.message_proj)
        mpnn_weights(layer.node_proj)
        mpnn_weights(layer.update_net)
        mpnn_weights(layer.norm)
    for mlp in model.pred_virtualnode_mlp:
        mpnn_weights(mlp)
    for p in model.pred_virtualnode_embedding:
        nn.init.zeros_(p)

    set_seed(seed + 9001)
    mpnn_weights(model.mol_node_proj)
    mpnn_weights(model.pred_node_proj)
    if isinstance(model.cross_mol2pred, nn.MultiheadAttention):
        model.cross_mol2pred._reset_parameters()
    if isinstance(model.cross_pred2mol, nn.MultiheadAttention):
        model.cross_pred2mol._reset_parameters()

    mpnn_weights(model.norm_mol)
    mpnn_weights(model.ff_mol)
    mpnn_weights(model.norm_pred)
    mpnn_weights(model.ff_pred)
    mpnn_weights(model.final_mlp)
    mpnn_weights(model.output_layer)
