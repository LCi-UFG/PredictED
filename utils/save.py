import os
import json
import torch


def save_model(
    model, 
    out_path, 
    filename):

    os.makedirs(out_path, exist_ok=True)
    if not filename.endswith(".pth"):
        filename += ".pth"
    filepath = os.path.join(out_path, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def save_params(
    params, 
    out_path, 
    filename):

    if not filename.endswith(".json"):
        filename += ".json"
    filepath = os.path.join(out_path, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Best parameters saved to {filepath}")


def save_thresholds(
    best_thresholds, 
    out_path='../output/calibration/thresholds.json'):

    try:
        dir_path = os.path.dirname(out_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        absolute_path = os.path.abspath(out_path)
        with open(absolute_path, 'w') as file:
            json.dump(best_thresholds, file, indent=4)
    except Exception as e:
        print(f"Failed to save best thresholds: {e}")


def save_embeddings(
    model,
    data_loader,
    epoch,
    out_path="../output/embeddings"):

    os.makedirs(out_path, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for (mol_batch, pred_batch), labels in data_loader:
            mol_batch  = mol_batch.to(device)
            pred_batch = pred_batch.to(device)

            emb = model(
                (mol_batch, pred_batch),
                save_embeddings=False,
                return_penultimate=True).cpu()
            all_embeddings.append(emb)

            if labels is not None:
                all_labels.append(
                    labels.view(emb.size(0), -1).cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    payload = {'embeddings': embeddings_tensor}

    if all_labels:
        labels_tensor = torch.cat(all_labels, dim=0)
        payload['labels'] = labels_tensor

    filename = os.path.join(
        out_path, f"embeddings_epoch_{epoch+1}.pt")
    torch.save(payload, filename)