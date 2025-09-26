
import os
import json
import torch
import optuna
import pandas as pd

from logits import (
    AttentiveNet,
    GATNet,
    GINet,
    MPNNet
    )
from attentive import AttentiveMod
from gat import GATMod
from gin import GINMod
from mpnn import MPNNMod


def load_data(file_path):
    df = pd.read_csv(
        file_path, 
        delimiter=',', 
        low_memory=False
        )
    smiles = df.iloc[:, 1].values
    targets = df.iloc[:, 2:].values
    
    return smiles, targets


def initialize_optuna():

    OUTPUT_DIR = "../output/optimization/"
    DB_FILE = os.path.join(
        OUTPUT_DIR, 'optuna_study.db'
        )
    STORAGE_NAME = f"sqlite:///{DB_FILE}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    study_name = "optimization_study"

    try:
        existing_studies = optuna.study.get_all_study_summaries(
            storage=STORAGE_NAME)
        study_names = [
            s.study_name for s in existing_studies]
        if study_name in study_names:
            print(f"Study '{study_name}' found in the database")
            study = optuna.load_study(
                study_name=study_name,
                storage=STORAGE_NAME)
        else:
            print(f"Creating a new study")
            study = optuna.create_study(
                study_name=study_name,
                direction='minimize',
                storage=STORAGE_NAME)
    except Exception as e:
        print(f"Error occurred while accessing the study: {e}")
        raise

    return study


def load_params(hyperparams_path):
    with open(hyperparams_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters


def load_tier1_model(
    model_path, 
    architecture_type, 
    params, 
    node_dim, 
    edge_dim,
    num_tasks):

    agg_hidden_dims = [
        params[f'agg_hidden_dim_{i+1}'] 
        for i in range(params['num_agg_layers'])
        ]
    lin_hidden_dims = [
        params[f'lin_hidden_dim_{i+1}'] 
        for i in range(params['num_lin_layers'])
        ]

    if architecture_type == 'attentive':
        tier1_model = AttentiveNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            num_timesteps=params['num_timesteps'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'gat':
        tier1_model = GATNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            heads=params['heads'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'gin':
        tier1_model = GINet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'mpnn':
        tier1_model = MPNNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            num_tasks=num_tasks
            )
    else:
        raise ValueError("Invalid architecture type")
    
    tier1_model.load_state_dict(
        torch.load(model_path, weights_only=True),
        strict=False
        )
    
    return tier1_model


def load_tier2_model(
    model_path, 
    architecture_type, 
    params, 
    node_dim, 
    edge_dim, 
    node_pred,
    edge_pred,
    num_tasks):

    agg_hidden_dims = [
        params[f'agg_hidden_dim_{i+1}'] 
        for i in range(params['num_agg_layers'])
        ]
    lin_hidden_dims = [
        params[f'lin_hidden_dim_{i+1}'] 
        for i in range(params['num_lin_layers'])
        ]

    if architecture_type == 'attentive':
        model = AttentiveMod(
            node_dim=node_dim,
            edge_dim=edge_dim,
            node_pred=node_pred,
            edge_pred=edge_pred,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            num_timesteps=params['num_timesteps'],
            co_heads=params['co_heads'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'gat':
        model = GATMod(
            node_dim=node_dim,
            edge_dim=edge_dim,
            node_pred=node_pred,
            edge_pred=edge_pred,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            heads=params['heads'],
            co_heads=params['co_heads'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'gin':
        model = GINMod(
            node_dim=node_dim,
            edge_dim=edge_dim,
            node_pred=node_pred,
            edge_pred=edge_pred,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            co_heads=params['co_heads'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'mpnn':
        model = MPNNMod(
            node_dim=node_dim,
            edge_dim=edge_dim,
            node_pred=node_pred,
            edge_pred=edge_pred,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            co_heads=params['co_heads'],
            num_tasks=num_tasks
            )
    else:
        raise ValueError("Invalid architecture type")
    
    model.load_state_dict(
        torch.load(model_path, weights_only=True),
        strict=False
        )
    
    return model


def load_embeddings(directory_path, epoch):
    target = f"embeddings_epoch_{epoch+1}"
    for filename in os.listdir(directory_path):
        if target in filename:
            data = torch.load(os.path.join(
                directory_path, filename))
            embeddings = data['embeddings']
            labels = data.get('labels', None)
            return embeddings, labels
    raise FileNotFoundError(
        f"No file found for epoch {epoch+1} in {directory_path}"
    )