import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP

from params import load_embeddings


def filter_nan(labels, embeddings):
    labels_np = np.array(labels).ravel()
    valid_indices = ~np.isnan(labels_np)
    filtered_labels = labels_np[valid_indices].tolist()
    filtered_embeddings = embeddings[valid_indices]
    return filtered_labels, filtered_embeddings


def normalize_embeddings(embeddings):
    scaler = StandardScaler()
    return scaler.fit_transform(embeddings)


def embeddings2tSNE(
    embeddings, 
    perplexity, 
    learning_rate, 
    seed):

    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity,
        learning_rate=learning_rate, 
        max_iter=1000,
        random_state=seed, init='pca'
    )
    return tsne.fit_transform(embeddings)


def embeddings2uMAP(
    embeddings, 
    n_neighbors, 
    min_dist, seed):

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed
    )
    return reducer.fit_transform(embeddings)


def plot_embeddings(
    reduced_embeddings,
    labels,
    file_path,
    title,
    x_label,
    y_label):

    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Define Arial como fonte padrão
    plt.rcParams['font.family'] = 'Arial'

    plt.figure(figsize=(4, 4))
    has_labels = labels is not None
    if has_labels:
        lab = (
            labels.numpy()
            if torch.is_tensor(labels)
            else labels
        )
        colors = [
            "silver" if l == 0 else "tomato"
            for l in lab
        ]
        plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=colors, s=11, alpha=0.8
        )
    else:
        plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c="gray", s=9, alpha=0.8
        )

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # Aqui ajustamos os números dos eixos (ticks)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig(
        file_path, dpi=500,
        bbox_inches="tight"
    )
    plt.show()


def visualize_embeddings(
    in_path, 
    epoch, 
    method, 
    task_index=None,
    out_path=None,
    perplexity=30, 
    learning_rate=200,
    n_neighbors=15, 
    min_dist=0.1,
    seed=42):

    embeddings, labels = load_embeddings(in_path, epoch)
    if isinstance(embeddings, list):
        embeddings = torch.cat(embeddings, dim=0)
        lbl = labels
        if torch.is_tensor(labels) and labels.dim() > 1:
            if task_index is None:
                raise ValueError(
                    "Please specify task_index"
                    )
            lbl = labels[:, task_index]
        labels = lbl.tolist() if torch.is_tensor(lbl) else lbl
    emb_array = embeddings.numpy()
    emb_norm = normalize_embeddings(emb_array)
    if method == 'tSNE':
        reduced = embeddings2tSNE(
            emb_norm, perplexity,
            learning_rate, seed
            )
    elif method == 'uMAP':
        reduced = embeddings2uMAP(
            emb_norm, n_neighbors,
            min_dist, seed
            )
    else:
        raise ValueError("Method must be 'tSNE' or 'uMAP'")
    plot_embeddings(
        reduced, labels,
        out_path or f"{method}_epoch_{epoch+1}.svg",
        title=f"{method} Embedding (Epoch {epoch+1})",
        x_label=f"{method} Dim 1",
        y_label=f"{method} Dim 2"
        )


def visualize_kde_dim(
    in_path,
    epoch,
    method,
    task_index=None,
    dim=0,
    out_path=None,
    perplexity=30,
    learning_rate=200,
    n_neighbors=15,
    min_dist=0.1,
    seed=42):

    embeddings, labels = load_embeddings(in_path, epoch)
    if isinstance(embeddings, list):
        embeddings = torch.cat(embeddings, dim=0)
        if torch.is_tensor(labels) and labels.dim() > 1:
            if task_index is None:
                raise ValueError(
                    "Para dados multitask, especifique "
                    "task_index"
                )
            labels = labels[:, task_index]
        labels = (
            labels.tolist()
            if torch.is_tensor(labels)
            else labels
        )

    labels, embeddings = filter_nan(labels, embeddings)
    X = (
        embeddings.numpy()
        if hasattr(embeddings, "numpy")
        else embeddings
    )
    X = normalize_embeddings(X)

    if method == "tSNE":
        reduced = embeddings2tSNE(X, perplexity, learning_rate, seed)
    elif method == "uMAP":
        reduced = embeddings2uMAP(X, n_neighbors, min_dist, seed)
    else:
        raise ValueError("method deve ser 'tSNE' ou 'uMAP'")

    arr = np.array(labels)
    vals0 = reduced[arr == 0, dim]
    vals1 = reduced[arr == 1, dim]

    from scipy.stats import gaussian_kde
    kde0 = gaussian_kde(vals0)
    kde1 = gaussian_kde(vals1)

    vmin = min(vals0.min(), vals1.min())
    vmax = max(vals0.max(), vals1.max())
    pad = (vmax - vmin) * 0.2
    xs = np.linspace(vmin - pad, vmax + pad, 300)
    y0 = kde0(xs)
    y1 = kde1(xs)
    overlap = np.trapz(np.minimum(y0, y1), xs)

    plt.figure(figsize=(4, 1.2))

    # >>> Ajuste da fonte e tamanho dos números dos eixos
    plt.rcParams['font.family'] = 'Arial'
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.fill_between(xs, y0, alpha=0.5, color="silver")
    plt.fill_between(xs, y1, alpha=0.5, color="tomato")
    plt.xlim(xs[0], xs[-1])

    plt.xlabel(f"Dim{dim+1}", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(f"Epoch {epoch+1} Dim{dim+1} KDE (overlap={overlap:.3f})", fontsize=16)

    if out_path is None:
        out_path = f"{method}_epoch_{epoch+1}_dim{dim+1}_kde.svg"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()