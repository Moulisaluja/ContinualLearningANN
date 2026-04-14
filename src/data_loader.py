import os
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


# =========================
# Dataset Class
# =========================
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_to_idx = label_to_idx

        self.encoded_labels = [self.label_to_idx[l] for l in labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        label = self.encoded_labels[idx]
        return image, label, path


# =========================
# Data Pipeline
# =========================
class DataPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_cfg = config["dataset"]
        self.task_cfg = config["tasks"]
        self.hw_cfg = config["hardware"]

        # =========================
        # FIX: PROJECT ROOT HANDLING
        # =========================
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_root = self.project_root / self.dataset_cfg["download_path"]

        logger.info(f"Data root resolved to: {self.data_root}")

        self.label_to_idx = {
            c: i for i, c in enumerate(self.dataset_cfg["class_names"])
        }

        self.train_tfms = self._train_transforms()
        self.eval_tfms = self._eval_transforms()

    # =========================
    # Transforms
    # =========================
    def _train_transforms(self):
        return transforms.Compose([
            transforms.Resize(
                (self.dataset_cfg["image_size"], self.dataset_cfg["image_size"])
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.dataset_cfg["normalize_mean"],
                std=self.dataset_cfg["normalize_std"]
            )
        ])

    def _eval_transforms(self):
        return transforms.Compose([
            transforms.Resize(
                (self.dataset_cfg["image_size"], self.dataset_cfg["image_size"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.dataset_cfg["normalize_mean"],
                std=self.dataset_cfg["normalize_std"]
            )
        ])

    # =========================
    # Dataset Scanning
    # =========================
    def _collect_images(self):
        image_paths = []
        labels = []

        valid_ext = (".png", ".jpg", ".jpeg")

        # walk entire dataset folder
        for root, _, files in os.walk(self.data_root):
            for f in files:
                if f.lower().endswith(valid_ext):
                    full_path = os.path.join(root, f)

                    # infer label from folder name
                    folder = Path(full_path).parent.name.lower()

                    for cls in self.dataset_cfg["class_names"]:
                        if cls.lower() in folder:
                            image_paths.append(full_path)
                            labels.append(cls)
                            break

        if len(image_paths) == 0:
            raise RuntimeError(f"No images found in {self.data_root}")

        return image_paths, labels

    # =========================
    # Task Creation
    # =========================
    def prepare_tasks(self):
        image_paths, labels = self._collect_images()

        logger.info(f"Total images found: {len(image_paths)}")

        tasks_data = {}

        for task in self.task_cfg["task_definitions"]:
            task_name = task["name"]
            task_classes = task["classes"]

            task_paths = []
            task_labels = []

            for p, l in zip(image_paths, labels):
                if l in task_classes:
                    task_paths.append(p)
                    task_labels.append(l)

            if len(task_paths) == 0:
                logger.warning(f"{task_name}: No data found")
                continue

            logger.info(f"{task_name}: {len(task_paths)} samples")

            train_val_x, test_x, train_val_y, test_y = train_test_split(
                task_paths,
                task_labels,
                test_size=self.dataset_cfg["test_split"],
                stratify=task_labels,
                random_state=self.hw_cfg["seed"]
            )

            val_ratio = self.dataset_cfg["val_split"] / (
                self.dataset_cfg["train_split"] + self.dataset_cfg["val_split"]
            )

            train_x, val_x, train_y, val_y = train_test_split(
                train_val_x,
                train_val_y,
                test_size=val_ratio,
                stratify=train_val_y,
                random_state=self.hw_cfg["seed"]
            )

            tasks_data[task_name] = {
                "train": (train_x, train_y),
                "val": (val_x, val_y),
                "test": (test_x, test_y),
                "classes": task_classes
            }

        self._log(tasks_data)
        return tasks_data

    # =========================
    # Logging
    # =========================
    def _log(self, tasks_data):
        logger.info("\n" + "=" * 60)
        logger.info("CONTINUAL LEARNING TASKS")
        logger.info("=" * 60)

        for t, d in tasks_data.items():
            logger.info(f"\n{t}")
            logger.info(f"Classes: {d['classes']}")
            logger.info(f"Train: {len(d['train'][0])}")
            logger.info(f"Val:   {len(d['val'][0])}")
            logger.info(f"Test:  {len(d['test'][0])}")

    # =========================
    # DataLoaders
    # =========================
    def get_dataloaders(self, task_data, batch_size=None):
        if batch_size is None:
            batch_size = self.config["training"]["batch_size"]

        train_x, train_y = task_data["train"]
        val_x, val_y = task_data["val"]
        test_x, test_y = task_data["test"]

        train_ds = ChestXRayDataset(train_x, train_y, self.label_to_idx, self.train_tfms)
        val_ds = ChestXRayDataset(val_x, val_y, self.label_to_idx, self.eval_tfms)
        test_ds = ChestXRayDataset(test_x, test_y, self.label_to_idx, self.eval_tfms)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.hw_cfg["num_workers"],
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hw_cfg["num_workers"],
            pin_memory=True
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hw_cfg["num_workers"],
            pin_memory=True
        )

        return train_loader, val_loader, test_loader


# =========================
# PUBLIC API
# =========================
def create_dataloaders(config: dict):
    pipeline = DataPipeline(config)
    tasks = pipeline.prepare_tasks()
    return pipeline, tasks
