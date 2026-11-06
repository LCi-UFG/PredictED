"""Load only the three published prediction checkpoints."""

import json
from pathlib import Path

import torch

from attentive import AttentiveMod
from gat import GATMod
from mpnn import MPNNet


def load_params(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _dimensions(params, prefix):
    return [params[f"{prefix}_dim_{index + 1}"]
            for index in range(params["num_agg_layers" if prefix == "agg_hidden"
                                       else "num_lin_layers"])]


def load_tier1_model(model_path, params_path, node_dim=48, edge_dim=12,
                     num_tasks=46, device="cpu"):
    params = load_params(params_path)
    model = MPNNet(
        node_dim, edge_dim,
        _dimensions(params, "agg_hidden"), params["num_agg_layers"],
        _dimensions(params, "lin_hidden"), params["num_lin_layers"],
        params["activation"], params["dropout_rate"], num_tasks,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state.pop("uncertainty.log_vars", None)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_tier2_model(model_path, params_path, architecture,
                     node_dim=48, edge_dim=12, node_pred=1, edge_pred=2,
                     device="cpu"):
    params = load_params(params_path)
    common = dict(
        node_dim=node_dim, edge_dim=edge_dim,
        node_pred=node_pred, edge_pred=edge_pred,
        agg_hidden_dims=_dimensions(params, "agg_hidden"),
        num_agg_layers=params["num_agg_layers"],
        lin_hidden_dims=_dimensions(params, "lin_hidden"),
        num_lin_layers=params["num_lin_layers"],
        activation=params["activation"],
        dropout_rate=params["dropout_rate"],
        co_heads=params["co_heads"], num_tasks=1,
    )
    if architecture == "attentive":
        model = AttentiveMod(**common, num_timesteps=params["num_timesteps"])
    elif architecture == "gat":
        model = GATMod(**common, heads=params["heads"])
    else:
        raise ValueError(f"Unsupported prediction architecture: {architecture}")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
