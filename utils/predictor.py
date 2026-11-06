"""One-substrate inference for the three frozen PredictED checkpoints."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

import path
from domain import DomainManifestError, file_sha256, load_domain
from graphs import pathway_edges, prediction_graph, smiles2graph
from params import load_tier1_model, load_tier2_model
from uncertainty import enable_mc_dropout, restore_states, uncertainty_parts
from common import device as default_device


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "tier1": ROOT / "output/models/tier1.pth",
    "hershberger": ROOT / "output/models/hershberger_gat.pth",
    "uterotrophic": ROOT / "output/models/uterotrophic_attentive.pth",
}
PARAM_PATHS = {
    "tier1": ROOT / "output/params/tier1.json",
    "hershberger": ROOT / "output/params/hershberger_gat.json",
    "uterotrophic": ROOT / "output/params/uterotrophic_attentive.json",
}
DOMAIN_PATHS = {
    name: ROOT / "output/models" / f"{name}_domain.joblib"
    for name in MODEL_PATHS
}
MC_SAMPLES = 50
MC_SEED = 42


def load_runtime(model_paths=None, params_paths=None, domain_paths=None,
                 device=default_device, require_domains=True):
    """Load weights and AD artifacts once, suitable for a FastAPI lifespan."""
    model_paths = {**MODEL_PATHS, **(model_paths or {})}
    params_paths = {**PARAM_PATHS, **(params_paths or {})}
    domain_paths = {**DOMAIN_PATHS, **(domain_paths or {})}
    assays = tuple(pd.read_csv(ROOT / "data/dataset.csv", nrows=0).columns[2:])
    edge_index, edge_attr = pathway_edges(ROOT / "data/info.csv", assays)
    models = {
        "tier1": load_tier1_model(model_paths["tier1"], params_paths["tier1"],
                                  num_tasks=len(assays), device=device),
        "hershberger": load_tier2_model(
            model_paths["hershberger"], params_paths["hershberger"],
            "gat", device=device),
        "uterotrophic": load_tier2_model(
            model_paths["uterotrophic"], params_paths["uterotrophic"],
            "attentive", device=device),
    }
    domains = {}
    for name, domain_path in domain_paths.items():
        if Path(domain_path).is_file():
            domain = load_domain(domain_path)
            domain.verify_trial(model_paths[name], params_paths[name])
            expected_tasks = assays if name == "tier1" else (name,)
            if tuple(domain.task_names) != expected_tasks:
                raise DomainManifestError(f"AD tasks do not match {name}")
            if name != "tier1" and domain.manifest.get("tier1_model") != \
                    file_sha256(model_paths["tier1"]):
                raise DomainManifestError(
                    f"{name} AD was built with different Tier-1 weights")
            if name != "tier1" and domain.manifest.get("pathway_info") != \
                    file_sha256(ROOT / "data/info.csv"):
                raise DomainManifestError(
                    f"{name} AD was built with different pathway metadata")
            domains[name] = domain
        elif require_domains:
            raise FileNotFoundError(
                f"Required prediction AD artifact missing: {domain_path}. "
                "Restore the fitted domain file before loading the runtime."
            )
    return SimpleNamespace(
        models=models, domains=domains, assays=assays,
        edge_index=edge_index.to(device), edge_attr=edge_attr.to(device),
        device=torch.device(device), model_paths=model_paths,
    )


def _molecule_batch(smiles, device):
    return Batch.from_data_list([smiles2graph(smiles)]).to(device)


def _pathway_batch(logits, runtime):
    graph = prediction_graph(logits.squeeze(0), runtime.edge_index,
                             runtime.edge_attr)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long,
                              device=runtime.device)
    return graph


def deterministic_embeddings(runtime, smiles):
    """Penultimate embeddings and original eval-mode logits for all models."""
    molecule = _molecule_batch(smiles, runtime.device)
    with torch.inference_mode():
        tier1_embedding = runtime.models["tier1"](
            molecule, return_penultimate=True)
        logits = runtime.models["tier1"].output_layer(tier1_embedding)
        pathway = _pathway_batch(logits, runtime)
        tier2_embeddings = {}
        tier2_logits = {}
        for name in ("hershberger", "uterotrophic"):
            model = runtime.models[name]
            embedding = model((molecule, pathway), return_penultimate=True)
            tier2_embeddings[name] = embedding
            head_input = (model.embedding_layer(embedding)
                          if name == "uterotrophic" else embedding)
            tier2_logits[name] = model.output_layer(head_input)
    return {
        "tier1": tier1_embedding,
        **tier2_embeddings,
    }, {"tier1": logits, **tier2_logits}


def _domain_rows(runtime, embeddings):
    result = {}
    for name, embedding in embeddings.items():
        domain = runtime.domains.get(name)
        if domain is None:
            result[name] = None
            continue
        tasks = runtime.assays if name == "tier1" else (name,)
        values = embedding.detach().cpu().numpy()
        values = values[:, None, :].repeat(len(tasks), axis=1)
        result[name] = domain.query(values, tasks, tasks=tasks)[0]
    return result


def predict(smiles, runtime, n_samples=MC_SAMPLES, seed=MC_SEED):
    """Return 46 Tier-1 assays and both Tier-2 endpoints with AD and UQ."""
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer greater than one")
    embeddings, point_logits = deterministic_embeddings(runtime, smiles)
    point_scores = {
        name: (logits.detach().cpu().numpy(),
               logits.sigmoid().detach().cpu().numpy())
        for name, logits in point_logits.items()
    }
    domain_rows = _domain_rows(runtime, embeddings)
    molecule = _molecule_batch(smiles, runtime.device)
    devices = ([runtime.device.index if runtime.device.index is not None
                else torch.cuda.current_device()]
               if runtime.device.type == "cuda" else [])
    states = [enable_mc_dropout(model) for model in runtime.models.values()]
    samples = {name: [] for name in runtime.models}
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if runtime.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                for _ in range(n_samples):
                    tier1_logits = runtime.models["tier1"](molecule)
                    samples["tier1"].append(tier1_logits.sigmoid())
                    pathway = _pathway_batch(tier1_logits, runtime)
                    for name in ("hershberger", "uterotrophic"):
                        output = runtime.models[name]((molecule, pathway))
                        samples[name].append(output.sigmoid())
    finally:
        for state in states:
            restore_states(state)
    decomposed = {
        name: {part: values.detach().cpu().numpy()
               for part, values in uncertainty_parts(torch.stack(draws)).items()
               if part != "prediction"}
        for name, draws in samples.items()
    }

    def row(name, index, task):
        parts = decomposed[name]
        logits, probabilities = point_scores[name]
        result = {
            "task": task,
            "logit": float(logits[0, index]),
            "probability": float(probabilities[0, index]),
            "total": float(parts["total"][0, index]),
            "aleatoric": float(parts["aleatoric"][0, index]),
            "epistemic": float(parts["epistemic"][0, index]),
            "domain": (domain_rows[name] or {}).get(task),
        }
        return result

    return {
        "smiles": smiles,
        "tier1": [row("tier1", index, task)
                  for index, task in enumerate(runtime.assays)],
        "hershberger": row("hershberger", 0, "hershberger"),
        "uterotrophic": row("uterotrophic", 0, "uterotrophic"),
        "mc_samples": n_samples,
    }


def predict_tier2(model, data_loader, device, return_embeddings=False,
                  return_attention=False):
    model.eval()
    all_preds, all_labels = [], []
    embeddings = []
    attn_m2p_list, attn_p2m_list = [], []
    with torch.no_grad():
        for (mol_batch, pred_batch), labels in data_loader:
            mol_batch, pred_batch = mol_batch.to(device), pred_batch.to(device)
            labels = labels.to(device)
            if return_embeddings:
                emb = model((mol_batch, pred_batch), return_penultimate=True)
                embeddings.extend(emb.cpu().numpy())
            if return_attention:
                out, attn_m2p, attn_p2m, _, _ = model(
                    (mol_batch, pred_batch), return_attention=True
                )
                attn_m2p_list.append(attn_m2p.cpu())
                attn_p2m_list.append(attn_p2m.cpu())
            else:
                out = model((mol_batch, pred_batch))
            preds = torch.sigmoid(out)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.view_as(preds).cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    embeddings = np.array(embeddings) if return_embeddings else None
    if return_attention:
        attn_m2p_all = torch.cat(attn_m2p_list, dim=0).numpy()
        attn_p2m_all = torch.cat(attn_p2m_list, dim=0).numpy()
        return all_preds, all_labels, embeddings, attn_m2p_all, attn_p2m_all
    return all_preds, all_labels, embeddings
