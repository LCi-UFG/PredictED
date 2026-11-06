import torch.nn as nn


def get_activation(activation_choice):
    if activation_choice == 'elu':
        return nn.ELU()
    elif activation_choice == 'gelu':
        return nn.GELU()
    elif activation_choice == 'selu':
        return nn.SELU()
    else:
        raise ValueError(f"Unknown function: {activation_choice}")
