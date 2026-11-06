"""Embedding applicability domain adapted from the supplied domain.py."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
from scipy.special import ndtr
from scipy.spatial.distance import cdist


ARTIFACT_VERSION = "task-relative-embedding-domain-v3"
_EPSILON = np.finfo(np.float64).eps


class DomainError(ValueError):
    pass


class DomainManifestError(DomainError):
    pass


def file_sha256(path):
    filepath = Path(path)
    if not filepath.is_file():
        raise DomainManifestError(f"required file does not exist: {filepath}")
    digest = hashlib.sha256()
    with filepath.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_space(embeddings, scaler, pca, covariance, inverse_root):
    reduced = pca.transform(scaler.transform(embeddings))
    return (reduced - covariance.location_) @ inverse_root


def query_distance(reference, query, neighbors, audit_neighbors):
    count = max(neighbors, min(max(1, audit_neighbors), reference.shape[0]))
    distances = cdist(query, reference, metric="euclidean")
    indices = np.argsort(distances, axis=1, kind="stable")[:, :count]
    distances = np.take_along_axis(distances, indices, axis=1)
    kept = min(max(1, audit_neighbors), reference.shape[0])
    return distances[:, neighbors - 1], distances[:, :kept], indices[:, :kept]


def tail_probability(reference, query, bandwidth):
    if reference.size < 1:
        raise DomainError("KDE requires reference values")
    result = np.empty(query.size, dtype=float)
    for start in range(0, query.size, 2048):
        stop = min(start + 2048, query.size)
        cumulative = ndtr(
            (query[start:stop, None] - reference[None, :]) / bandwidth
        ).sum(axis=1)
        result[start:stop] = 1.0 - cumulative / reference.size
    return np.clip(result, _EPSILON, 1.0)


def domain_status(anomaly, thresholds):
    if anomaly <= thresholds["inside_max"]:
        return "inside"
    if anomaly <= thresholds["borderline_max"]:
        return "borderline"
    return "outside"


class TaskEmbeddingDomain:
    def verify_trial(self, model_path, params_path=None):
        if file_sha256(model_path) != self.manifest["model"]["sha256"]:
            raise DomainManifestError(
                "the domain artifact belongs to different model weights")
        if params_path is not None and file_sha256(params_path) != \
                self.manifest["params"]["sha256"]:
            raise DomainManifestError(
                "the domain artifact belongs to different model parameters")

    def query_task(self, embeddings, task, audit_neighbors):
        if task not in self.task_domains:
            return [{"task": task, "status": "unavailable"}
                    for _ in embeddings]
        model = self.task_domains[task]
        transformed = transform_space(
            embeddings, model["scaler"], model["pca"],
            model["covariance"], model["inverse_root"],
        )
        kth, distances, indices = query_distance(
            model["reference"], transformed,
            model["configuration"]["neighbors"], audit_neighbors,
        )
        query_log = np.log(np.maximum(np.asarray(kth, dtype=float), 1e-12))
        tail = tail_probability(
            model["loo_log_distance"], query_log,
            model["configuration"]["bandwidth"],
        )
        anomaly = -np.log(tail)
        density = model["kde"].score_samples(query_log.reshape(-1, 1))
        rows = []
        for row in range(transformed.shape[0]):
            nearest = [
                {
                    "sample_id": model["reference_ids"][index],
                    "smiles": model["reference_smiles"][index],
                    "distance": float(distance),
                }
                for distance, index in zip(distances[row], indices[row])
            ]
            rows.append({
                "task": task,
                "status": domain_status(float(anomaly[row]),
                                        model["thresholds"]),
                "anomaly_score": float(anomaly[row]),
                "distance_percentile": float(100.0 * (1.0 - tail[row])),
                "train_relative_distance_tail": float(tail[row]),
                "kth_neighbor_distance": float(kth[row]),
                "log_kth_neighbor_distance": float(query_log[row]),
                "log_distance_density": float(density[row]),
                "reference_count": len(model["reference_ids"]),
                "configuration": dict(model["configuration"]),
                "distance_quantiles": dict(model["distance_quantiles"]),
                "thresholds": dict(model["thresholds"]),
                "nearest_neighbors": nearest,
            })
        return rows

    def query(self, embeddings, embedding_tasks, tasks=None, audit_neighbors=5):
        values = np.asarray(embeddings, dtype=float)
        if values.ndim != 3:
            raise DomainError("query embeddings must be 3-dimensional")
        source_tasks = tuple(map(str, embedding_tasks))
        if values.shape[1] != len(source_tasks):
            raise DomainError("query task names do not match the embedding columns")
        selected_tasks = self.task_names if tasks is None else tuple(map(str, tasks))
        indices = {task: index for index, task in enumerate(source_tasks)}
        missing = [task for task in selected_tasks if task not in indices]
        if missing:
            raise DomainError(f"query embeddings are missing tasks: {', '.join(missing)}")
        task_rows = {
            task: self.query_task(values[:, indices[task]], task,
                                  audit_neighbors)
            for task in selected_tasks
        }
        return [
            {task: task_rows[task][row] for task in selected_tasks}
            for row in range(values.shape[0])
        ]

    @classmethod
    def load(cls, path):
        filepath = Path(path)
        if not filepath.is_file():
            raise DomainError(f"domain artifact does not exist: {filepath}")
        payload = joblib.load(filepath)
        if not isinstance(payload, Mapping):
            raise DomainError("invalid domain artifact")
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise DomainError("unsupported domain artifact version")
        state = payload.get("state")
        if not isinstance(state, Mapping) or not {
            "task_domains", "task_names", "bom_names"
        }.issubset(state):
            raise DomainError("invalid task-domain state")
        domain = cls()
        domain.__dict__.update(state)
        return domain


def load_domain(path):
    return TaskEmbeddingDomain.load(path)
