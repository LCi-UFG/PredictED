import numpy as np
import torch


def predict_tier1(
    model,
    data_loader,
    device,
    return_embeddings=False):
    
    model.eval()
    all_predictions = []
    all_labels = []
    embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            if return_embeddings:
                emb = model(batch, return_embeddings=True)
                embeddings.extend(emb.cpu().numpy())

            outputs = model(batch)
            #preds = torch.sigmoid(outputs)
            all_predictions.extend(outputs.cpu().numpy())

            if hasattr(batch, 'y') and batch.y is not None:
                all_labels.extend(batch.y.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels) if all_labels else None
    embeddings = np.array(embeddings) if return_embeddings else None

    return all_predictions, all_labels, embeddings

def format_prediction(predictions):
    prob = float(predictions.flatten()[0])
    classification = 1 if prob >= 0.5 else 0
    label = "✅ POSITIVE" if classification == 1 else "❌ NEGATIVE"
    
    print(f"Probability: {prob:.4f} ({prob*100:.2f}%)")
    print(f"Classification: {classification}")
    print(f"Label: {label}")


def predict_tier2(
    model,
    data_loader,
    device,
    return_embeddings=False,
    return_attention=False,
    show_results=False):

    model.eval()
    all_preds, all_labels = [], []
    embeddings = []
    attn_m2p_list, attn_p2m_list = [], []

    with torch.no_grad():
        for (mol_batch, pred_batch), labels in data_loader:
            mol_batch, pred_batch = (
                mol_batch.to(device), pred_batch.to(device)
                )
            labels = labels.to(device)
            if return_embeddings:
                emb = model(
                    (mol_batch, pred_batch),
                    return_penultimate=True
                    )
                embeddings.extend(emb.cpu().numpy())
            if return_attention:
                out, A_m2p, A_p2m = model(
                    (mol_batch, pred_batch),
                    return_attention=True
                    )
                attn_m2p_list.append(A_m2p.cpu())
                attn_p2m_list.append(A_p2m.cpu())
            else:
                out = model((mol_batch, pred_batch))
            preds = torch.sigmoid(out)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(
                labels.view_as(preds).cpu().numpy()
                )
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    embeddings = np.array(embeddings
        ) if return_embeddings else None

    if return_attention:
        A_m2p_all = torch.cat(
            attn_m2p_list, dim=0).numpy()
        A_p2m_all = torch.cat(
            attn_p2m_list, dim=0).numpy()
        
        return (
            all_preds, all_labels, embeddings, 
            A_m2p_all, A_p2m_all
            )
    # Show formatted results
    if show_results and len(all_preds) == 1:
        print("\n--- TIER-2 PREDICTION ---")
        print("=" * 35)
        format_prediction(all_preds)
        print("=" * 35)

    return all_preds, all_labels, embeddings