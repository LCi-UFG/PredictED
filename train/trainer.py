import copy
import torch
import optuna
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import (
    device, 
    clip_gradients
    )
from save import save_embeddings


def train_epoch(
    model,
    optimizer,
    data_loader,
    loss_fn,
    max_grad_norm=5.0,
    clip_method='norm'):

    model.train()
    total_loss = 0.0

    for (mol_batch, pred_batch), labels in data_loader:
        optimizer.zero_grad()

        mol_batch = mol_batch.to(device)
        pred_batch = pred_batch.to(device)
        out = model((mol_batch, pred_batch))

        labels = labels.view_as(out).to(device)
        loss = loss_fn(out, labels)
        loss.backward()

        clip_gradients(
            model, max_grad_norm, method=clip_method
        )
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(
    model,
    data_loader,
    loss_fn):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for (mol_batch, pred_batch), labels in data_loader:
            mol_batch = mol_batch.to(device)
            pred_batch = pred_batch.to(device)

            out = model((mol_batch, pred_batch))
            labels = labels.view_as(out).to(device)

            loss = loss_fn(out, labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    num_epochs,
    patience,
    delta,
    window_size,
    best_model=True,
    warm_up_epochs=3,
    eta_min=0.001,
    enable_pruning=True):

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=warm_up_epochs,
        min_lr=eta_min
        )
    best_val_loss = float('inf')
    min_val_loss = float('inf')
    best_model_state = None
    best_epoch = None
    epochs_no_improve = 0
    val_loss_window = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            loss_fn
            )
        train_losses.append(avg_train_loss)
        avg_val_loss = evaluate(model, val_loader, loss_fn)
        val_losses.append(avg_val_loss)
        val_loss_window.append(avg_val_loss)
        if len(val_loss_window) > window_size:
            val_loss_window.pop(0)
        avg_val_loss_window = (sum(val_loss_window)
            / len(val_loss_window)
            )
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_train_loss:.4f} - "
            f"Val: {avg_val_loss:.4f} - "
            f"Win: {avg_val_loss_window:.4f}"
            )
        if (avg_val_loss_window
            < best_val_loss - delta):

            best_val_loss = avg_val_loss_window
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(
                model.state_dict()
                )
            best_epoch = epoch + 1
        if best_model:
            save_embeddings(
                model,
                train_loader,
                epoch,
                "../output/embeddings"
                )
        if (enable_pruning
            and val_losses
            and val_losses[0] < 0.7):
            print(f"Pruned: low initial loss: "
                f"{val_losses[0]:.4f}"
                )
            raise optuna.exceptions.TrialPruned()
        scheduler.step(avg_val_loss)

    if best_model_state is not None:
        print(f"Restoring best model from epoch "
            f"{best_epoch} with val_loss "
            f"{min_val_loss:.4f}"
            )
        model.load_state_dict(best_model_state)

    return (
        best_val_loss,
        min_val_loss,
        train_losses,
        val_losses
        )