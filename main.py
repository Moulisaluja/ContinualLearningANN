import torch
import numpy as np
from collections import defaultdict
from typing import Tuple


class PrivacyPreservingReplayBuffer:
    """Privacy-preserving replay buffer storing only embeddings (not raw images)."""

    def __init__(self, config: dict, embedding_dim: int, device: torch.device):
        self.config = config
        self.buffer_config = config['replay_buffer']
        self.device = device
        self.embedding_dim = embedding_dim

        self.buffer_size = self.buffer_config['buffer_size']
        self.alpha = self.buffer_config['priority_weights']['uncertainty']
        self.beta = self.buffer_config['priority_weights']['recency']
        self.gamma = self.buffer_config['priority_weights']['diversity']

        # Storage
        self.embeddings = []
        self.labels = []
        self.uncertainties = []
        self.timestamps = []
        self.task_ids = []

        self.current_size = 0
        self.total_added = 0
        self.current_timestamp = 0

        self.selection_counts = defaultdict(int)

    # -------------------------
    # SAFE NORMALIZATION
    # -------------------------
    def _normalize(self, arr):
        if len(arr) == 0:
            return arr
        min_v, max_v = arr.min(), arr.max()
        if max_v - min_v < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_v) / (max_v - min_v)

    # -------------------------
    # ADD SAMPLES
    # -------------------------
    def add_samples(self, embeddings, labels, uncertainties, task_id):
        batch_size = embeddings.size(0)

        for i in range(batch_size):
            embedding = embeddings[i].detach().cpu().clone()
            label = labels[i].item()
            uncertainty = uncertainties[i].item()

            if self.current_size < self.buffer_size:
                self.embeddings.append(embedding)
                self.labels.append(label)
                self.uncertainties.append(uncertainty)
                self.timestamps.append(self.current_timestamp)
                self.task_ids.append(task_id)
                self.current_size += 1
            else:
                idx = self._get_replacement_index()
                self.embeddings[idx] = embedding
                self.labels[idx] = label
                self.uncertainties[idx] = uncertainty
                self.timestamps[idx] = self.current_timestamp
                self.task_ids[idx] = task_id

            self.total_added += 1
            self.current_timestamp += 1

    # -------------------------
    # PRIORITY COMPUTATION
    # -------------------------
    def _get_replacement_index(self):
        priorities = self._compute_priorities()
        return np.argmin(priorities)

    def _compute_priorities(self):
        u_scores = self._normalize(np.array(self.uncertainties))
        r_scores = self._normalize(np.array(self.timestamps))
        d_scores = self._normalize(self._compute_diversity_scores())

        return (
            self.alpha * u_scores +
            self.beta * r_scores +
            self.gamma * d_scores
        )

    # -------------------------
    # DIVERSITY (FIXED)
    # -------------------------
    def _compute_diversity_scores(self):
        if self.current_size < 2:
            return np.zeros(self.current_size)

        embeddings_array = torch.stack(
            [e.to(self.device) for e in self.embeddings[:self.current_size]]
        )

        distances = torch.cdist(embeddings_array, embeddings_array)

        k = min(5, self.current_size - 1)
        diversity = torch.topk(distances, k=k + 1)[0][:, 1:].mean(dim=1)

        return diversity.detach().cpu().numpy()

    # -------------------------
    # SAMPLING (IMPROVED)
    # -------------------------
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current_size == 0:
            return None, None

        sample_size = min(batch_size, self.current_size)
        unique_tasks = list(set(self.task_ids))

        samples_per_task = sample_size // len(unique_tasks)

        sampled_indices = []

        for task_id in unique_tasks:
            task_indices = [i for i, t in enumerate(self.task_ids) if t == task_id]
            if task_indices:
                selected = np.random.choice(
                    task_indices,
                    size=min(samples_per_task, len(task_indices)),
                    replace=False
                )
                sampled_indices.extend(selected)

                for idx in selected:
                    self.selection_counts[idx] += 1

        # Fill remaining samples
        if len(sampled_indices) < sample_size:
            remaining = sample_size - len(sampled_indices)
            all_indices = list(range(self.current_size))
            additional = np.random.choice(all_indices, size=remaining, replace=True)
            sampled_indices.extend(additional)

        sampled_indices = sampled_indices[:sample_size]

        embeddings = torch.stack(
            [self.embeddings[i].to(self.device) for i in sampled_indices]
        )
        labels = torch.tensor(
            [self.labels[i] for i in sampled_indices],
            dtype=torch.long,
            device=self.device
        )

        return embeddings, labels

    # -------------------------
    # STATS
    # -------------------------
    def get_statistics(self):
        if self.current_size == 0:
            return {'buffer_size': 0, 'utilization': 0.0}

        task_distribution = defaultdict(int)
        for task_id in self.task_ids[:self.current_size]:
            task_distribution[task_id] += 1

        return {
            'current_size': self.current_size,
            'buffer_capacity': self.buffer_size,
            'utilization': self.current_size / self.buffer_size,
            'total_added': self.total_added,
            'task_distribution': dict(task_distribution),
            'mean_uncertainty': float(np.mean(self.uncertainties[:self.current_size])),
            'mean_age': float(
                np.mean(self.current_timestamp - np.array(self.timestamps[:self.current_size]))
            ),
        }

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        self.embeddings = []
        self.labels = []
        self.uncertainties = []
        self.timestamps = []
        self.task_ids = []
        self.current_size = 0
        self.current_timestamp = 0
        self.selection_counts.clear()
