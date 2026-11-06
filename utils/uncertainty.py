"""MC-dropout decomposition adapted from the supplied uncertainty.py."""

import torch
from torch import nn


DROPOUT_TYPES = (
    nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
    nn.AlphaDropout, nn.FeatureAlphaDropout,
)


def binary_entropy(probability):
    eps = torch.finfo(probability.dtype).eps
    probability = probability.clamp(eps, 1.0 - eps)
    return -(
        probability * probability.log()
        + (1.0 - probability) * (1.0 - probability).log()
    )


def uncertainty_parts(probabilities):
    prediction = probabilities.mean(dim=0)
    total = binary_entropy(prediction)
    aleatoric = binary_entropy(probabilities).mean(dim=0)
    epistemic = (total - aleatoric).clamp_min(0.0)
    return {
        "prediction": prediction,
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
    }


def enable_mc_dropout(model):
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    for module in model.modules():
        if isinstance(module, DROPOUT_TYPES) and module.p > 0:
            module.train()
    return states


def restore_states(states):
    for module, training in states:
        module.train(training)


