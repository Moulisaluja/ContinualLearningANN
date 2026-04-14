import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src import (
    create_model, TaskAwareEWC, DriftDetector,
    PrivacyPreservingReplayBuffer, ContinualLearningMetrics
)
from src.metrics import compute_mc_dropout_uncertainty


logger = logging.getLogger(__name__)


class ContinualLearningTrainer:
    """Trainer for continual learning with DEAR framework."""

    def __init__(self, config: dict, device: torch.device, logger):
        self.config = config
        self.device = device
        self.logger = logger

        # Model
        self.model = create_model(config, device)

        # Components
        self.ewc = TaskAwareEWC(self.model, config, device)
        self.drift_detector = DriftDetector(config, device)

        self.replay_buffer = PrivacyPreservingReplayBuffer(
            config,
            config['model']['embedding_dim'],
            device
        )

        self.metrics = ContinualLearningMetrics()

        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()

        self.current_task = 0

    def _create_optimizer(self):
        if self.config['training']['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        return optim.SGD(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            momentum=0.9
        )

    # =========================
    # TRAIN TASK
    # =========================
    def train_task(self, task_id: int, train_loader, val_loader, task_classes: list):

        self.logger.info(f"\nTraining Task {task_id}: {task_classes}")

        self.current_task = task_id
        best_acc = 0.0

        for epoch in range(self.config['training']['epochs_per_task']):

            train_loss = self._train_epoch(train_loader)

            val_acc, val_loss = self._evaluate(val_loader)

            self.logger.info(
                f"Task {task_id} Epoch {epoch+1} | "
                f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Drift check
            if self._check_drift(val_loader):
                self.logger.info("Drift detected → replay triggered")
                self._replay_and_retrain()

            if val_acc > best_acc:
                best_acc = val_acc
                self._save_checkpoint(task_id, epoch, val_acc)

        # Fisher matrix after task
        if self.config['ewc']['use_ewc']:
            self.ewc.compute_fisher_information_matrix(train_loader, task_id)

        # Fill replay buffer
        self._populate_replay_buffer(train_loader, task_id)

        return best_acc

    # =========================
    # TRAIN EPOCH
    # =========================
    def _train_epoch(self, train_loader):

        self.model.train()
        total_loss = 0

        for images, labels, _ in tqdm(train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits, embeddings = self.model(images, return_embedding=True)

            ce_loss = self.criterion(logits, labels)

            ewc_loss = self.ewc.ewc_loss(self.model) if self.config['ewc']['use_ewc'] else 0.0

            loss = ce_loss + ewc_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    # =========================
    # EVALUATION
    # =========================
    def _evaluate(self, data_loader):

        self.model.eval()

        correct = 0
        total = 0
        loss_sum = 0

        with torch.no_grad():
            for images, labels, _ in data_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)

                loss = self.criterion(logits, labels)

                loss_sum += loss.item()

                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total, loss_sum / len(data_loader)

    # =========================
    # DRIFT DETECTION
    # =========================
    def _check_drift(self, val_loader):

        self.model.eval()

        drift_flag = False

        with torch.no_grad():
            for i, (images, _, _) in enumerate(val_loader):

                images = images.to(self.device)

                embeddings = self.model.extract_embeddings(images)

                drift, _ = self.drift_detector.detect_drift(embeddings, i)

                if drift:
                    drift_flag = True

        return drift_flag

    # =========================
    # REPLAY BUFFER POPULATION
    # =========================
    def _populate_replay_buffer(self, train_loader, task_id):

        self.model.eval()

        self.logger.info("Populating replay buffer...")

        with torch.no_grad():
            for images, labels, _ in train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                logits, embeddings = self.model(images, return_embedding=True)

                # =========================
                # FIXED MC DROPOUT UNCERTAINTY
                # =========================
                mc_outputs = []

                for _ in range(self.config['model']['mc_dropout_samples']):
                    logits_mc, _ = self.model(images, return_embedding=True)
                    probs_mc = torch.softmax(logits_mc, dim=1)
                    mc_outputs.append(probs_mc)

                uncertainty = compute_mc_dropout_uncertainty(mc_outputs)

                # IMPORTANT FIX: detach embeddings
                self.replay_buffer.add_samples(
                    embeddings.detach(),
                    labels,
                    uncertainty,
                    task_id
                )

        self.logger.info(f"Replay buffer stats: {self.replay_buffer.get_statistics()}")

    # =========================
    # REPLAY TRAINING
    # =========================
    def _replay_and_retrain(self):

        self.logger.info("Replay training started...")

        for _ in range(5):

            embeddings, labels = self.replay_buffer.sample_batch(
                self.config['replay_buffer']['buffer_size'] // 10
            )

            if embeddings is None:
                return

            logits = self.model.classifier(embeddings)

            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # =========================
    # CHECKPOINT
    # =========================
    def _save_checkpoint(self, task_id, epoch, acc):

        path = Path(self.config['logging']['checkpoint_dir'])
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "task_id": task_id,
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "accuracy": acc
        }, path / f"task_{task_id}_acc_{acc:.4f}.pt")
