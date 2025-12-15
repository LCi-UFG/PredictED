import torch.optim as optim

from attentive import AttentiveMod
from gat import GATMod
from gin import GINMod
from mpnn import MPNNMod


def configure_optimizer(trial, model):
    
    optimizer_name = trial.suggest_categorical(
        'optimizer', [
            'Adam', 'AdamW', 'RAdam'])
    weight_decay = trial.suggest_float(
        'weight_decay', 1e-8, 1e-4, log=True)
    lr = 0.005
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
        )
    return optimizer


def configure_attentive(
    trial, 
    node_dim, 
    edge_dim,
    node_pred,
    edge_pred,
    num_tasks):
    
    num_agg_layers = trial.suggest_int('num_agg_layers', 2, 4)
    num_lin_layers = trial.suggest_int('num_lin_layers', 2, 4)

    choices = [12,24,36,48,60,72,84,96,108,120,144,180,192,240,
               300,324,348,372,396,420,444,456,480,516
               ]
    agg_hidden_dims = [
        trial.suggest_categorical(
            f'agg_hidden_dim_{i+1}', choices)
        for i in range(num_agg_layers)
        ]
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i+1}', 10, 500)
        for i in range(num_lin_layers)
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu','leakyrelu','elu','gelu','selu']
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.1, 0.3
        )
    num_timesteps = trial.suggest_int(
        'num_timesteps', 1, 3
        )
    co_heads = trial.suggest_categorical(
        'co_heads', [2,3,4,6]
        )
    
    model = AttentiveMod(
        node_dim, 
        edge_dim,
        node_pred,
        edge_pred,
        agg_hidden_dims, 
        len(agg_hidden_dims), 
        lin_hidden_dims, 
        len(lin_hidden_dims), 
        activation_choice, 
        dropout_rate,
        num_timesteps, 
        co_heads,
        num_tasks
        )
    
    return model


def configure_gat(
    trial,
    node_dim,
    edge_dim,
    node_pred,
    edge_pred,
    num_tasks):

    num_agg_layers = trial.suggest_int('num_agg_layers', 2, 4)
    num_lin_layers = trial.suggest_int('num_lin_layers', 2, 4)

    choices = [12,24,36,48,60,72,84,96,108,120,144,180,192,240,
               300,324,348,372,396,420,444,456,480,516
               ]
    agg_hidden_dims = [
        trial.suggest_categorical(
            f'agg_hidden_dim_{i+1}', choices)
        for i in range(num_agg_layers)
        ]
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i+1}', 10, 500)
        for i in range(num_lin_layers)
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu','leakyrelu','elu','gelu','selu']
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.1, 0.3
        )
    heads = trial.suggest_int('heads', 2, 12)
    co_heads = trial.suggest_categorical(
        'co_heads', [2,3,4,6]
        )

    model = GATMod(
        node_dim,
        edge_dim,
        node_pred,
        edge_pred,
        agg_hidden_dims,
        len(agg_hidden_dims),
        lin_hidden_dims,
        len(lin_hidden_dims),
        activation_choice,
        dropout_rate,
        heads,
        co_heads,
        num_tasks
        )

    return model
