import torch.nn as nn

from utils import set_seed
from torch_geometric.nn import (
    GATv2Conv,
    GraphNorm
    ) 

DEFAULT_SEED = 42

def attentive_weights(m):
    set_seed(DEFAULT_SEED)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.LayerNorm, GraphNorm)):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.reset_running_stats()
        if m.affine:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    elif isinstance(m, (nn.Sequential, nn.ModuleList)):
        for sub in m:
            attentive_weights(sub)
    elif isinstance(m, nn.MultiheadAttention):
        set_seed(DEFAULT_SEED)
        m._reset_parameters()
    elif hasattr(m, "reset_parameters"):
        set_seed(DEFAULT_SEED)
        m.reset_parameters()


def gat_weights(m):
    set_seed(DEFAULT_SEED)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif isinstance(m, GATv2Conv):
        set_seed(DEFAULT_SEED)
        m.reset_parameters()
    elif isinstance(m, nn.MultiheadAttention):
        set_seed(DEFAULT_SEED)
        m._reset_parameters()
    elif isinstance(m, (nn.LayerNorm, GraphNorm)):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.reset_running_stats()
        if m.affine:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    elif isinstance(m, (nn.Sequential, nn.ModuleList)):
        for sub in m:
            gat_weights(sub)
    elif hasattr(m, "reset_parameters"):
        set_seed(DEFAULT_SEED)
        m.reset_parameters()
    elif hasattr(m, "reset_parameters"):
        set_seed(DEFAULT_SEED)

        m.reset_parameters()
