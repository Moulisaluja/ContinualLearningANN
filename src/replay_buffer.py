import torch
import torch.nn as nn
import numpy as np
import logging
from collections import defaultdict
from typing import List, Tuple


logger = logging.getLogger(__name__)


class PrivacyPreservingReplayBuffer:
    """Privacy-preserving replay buffer storing only embeddings (not raw images)."""
    
    def __init__(self, config: dict, embedding_dim: int, device: torch.device):
        self.config = config
        self.buffer_config = config['replay_buffer']
        self.device = device
        self.embedding_dim = embedding_dim
        
        self.buffer_size = self.buffer_config['buffer_size']
        self.alpha = self.buffer_config['priority_weights']['uncertainty']  # Uncertainty weight
        self.beta = self.buffer_config['priority_weights']['recency']       # Recency weight
        self.gamma = self.buffer_config['priority_weights']['diversity']    # Diversity weight
        
        # Storage
        self.embeddings = []  # List of embedding tensors
        self.labels = []      # List of labels
        self.uncertainties = []  # Uncertainty scores (MC Dropout variance)
        self.timestamps = []  # When sample was added
        self.task_ids = []    # Which task sample belongs to
        
        self.current_size = 0
        self.total_added = 0
        self.current_timestamp = 0
        
        # Statistics
        self.selection_counts = defaultdict(int)
    
    def add_samples(self, embeddings: torch.Tensor, labels: torch.Tensor,
                   uncertainties: torch.Tensor, task_id: int):
        """
        Add samples to buffer with uncertainty-guided priority.
        
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size]
            uncertainties: [batch_size] - MC Dropout variance
            task_id: Current task ID
        """
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            embedding = embeddings[i].detach().cpu()
            label = labels[i].item()
            uncertainty = uncertainties[i].item()
            
            # If buffer not full, just add
            if self.current_size < self.buffer_size:
                self.embeddings.append(embedding)
                self.labels.append(label)
                self.uncertainties.append(uncertainty)
                self.timestamps.append(self.current_timestamp)
                self.task_ids.append(task_id)
                self.current_size += 1
            else:
                # Buffer full - replace with priority
                replacement_idx = self._get_replacement_index()
                self.embeddings[replacement_idx] = embedding
                self.labels[replacement_idx] = label
                self.uncertainties[replacement_idx] = uncertainty
                self.timestamps[replacement_idx] = self.current_timestamp
                self.task_ids[replacement_idx] = task_id
            
            self.total_added += 1
            self.current_timestamp += 1
    
    def _get_replacement_index(self) -> int:
        """Get index of sample to replace (lowest priority)."""
        priorities = self._compute_priorities()
        return np.argmin(priorities)
    
    def _compute_priorities(self) -> np.ndarray:
        """Compute priority scores for all samples."""
        n_samples = len(self.embeddings)
        
        # Uncertainty score (higher is higher priority)
        u_scores = np.array(self.uncertainties)
        u_scores = (u_scores - u_scores.min()) / (u_scores.max() - u_scores.min() + 1e-8)
        
        # Recency score (more recent is higher priority)
        r_scores = np.array(self.timestamps)
        r_scores = (r_scores - r_scores.min()) / (r_scores.max() - r_scores.min() + 1e-8)
        
        # Diversity score (how different from others)
        d_scores = self._compute_diversity_scores()
        d_scores = (d_scores - d_scores.min()) / (d_scores.max() - d_scores.min() + 1e-8)
        
        # Combined priority
        priorities = (self.alpha * u_scores +
                     self.beta * r_scores +
                     self.gamma * d_scores)
        
        return priorities
    
    def _compute_diversity_scores(self) -> np.ndarray:
        """Compute diversity of each sample in buffer."""
        if self.current_size == 0:
            return np.array([])
        
        embeddings_array = torch.stack(
            [e.to(self.device) for e in self.embeddings[:self.current_size]]
        )
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings_array, embeddings_array)
        
        # Diversity = average distance to k nearest neighbors
        k = min(5, self.current_size - 1)
        diversity = torch.topk(distances, k=k+1)[0][:, 1:].mean(dim=1)  # Exclude self
        
        return diversity.detach().cpu().numpy()
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample batch from buffer.
        
        Args:
            batch_size: Number of samples to draw
        
        Returns:
            (embeddings, labels)
        """
        if self.current_size == 0:
            return None, None
        
        sample_size = min(batch_size, self.current_size)
        
        # Stratified sampling: sample from each task
        unique_tasks = set(self.task_ids)
        samples_per_task = max(1, sample_size // len(unique_tasks))
        
        sampled_indices = []
        for task_id in unique_tasks:
            task_indices = [i for i, t in enumerate(self.task_ids) if t == task_id]
            if task_indices:
                selected = np.random.choice(task_indices, 
                                          size=min(samples_per_task, len(task_indices)),
                                          replace=False)
                sampled_indices.extend(selected)
                for idx in selected:
                    self.selection_counts[idx] += 1
        
        # If we don't have enough, random sample
        if len(sampled_indices) < sample_size:
            remaining = sample_size - len(sampled_indices)
            all_indices = list(range(self.current_size))
            additional = np.random.choice(all_indices, size=remaining, replace=True)
            sampled_indices.extend(additional)
        
        sampled_indices = sampled_indices[:sample_size]
        
        # Collect embeddings and labels
        embeddings = torch.stack([self.embeddings[i].to(self.device) for i in sampled_indices])
        labels = torch.tensor([self.labels[i] for i in sampled_indices],
                             dtype=torch.long, device=self.device)
        
        return embeddings, labels
    
    def get_statistics(self) -> dict:
        """Get buffer statistics."""
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
            'mean_age': float(np.mean(self.current_timestamp - np.array(self.timestamps[:self.current_size]))),
        }
    
    def reset(self):
        """Reset buffer."""
        self.embeddings = []
        self.labels = []
        self.uncertainties = []
        self.timestamps = []
        self.task_ids = []
        self.current_size = 0
        self.current_timestamp = 0
        self.selection_counts.clear()
