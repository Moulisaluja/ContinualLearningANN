import yaml
import logging
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


# =========================
# Config Handling
# =========================
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, save_path: str):
    """Save configuration to YAML file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# =========================
# Logging
# =========================
def setup_logging(log_dir: str, exp_name: str = None):
    """Setup logging (file + console)."""
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(log_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # FIX: prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_dir / "run.log")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_dir


# =========================
# Reproducibility
# =========================
def set_seed(seed: int, deterministic: bool = True):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# Directories
# =========================
def create_directories(config: dict):
    """Create required directories from config."""
    dirs = [
        config['logging']['log_dir'],
        config['logging']['checkpoint_dir'],
        config['logging']['metrics_dir'],
        config['logging']['plot_dir'],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# =========================
# Device Handling
# =========================
def get_device(device_str: str) -> torch.device:
    """Return torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    """Move tensors to device safely."""
    if isinstance(data, (list, tuple)):
        return type(data)(x.to(device) for x in data)
    return data.to(device)


# =========================
# Training Utilities
# =========================
class AverageMeter:
    """Tracks average values (loss, accuracy, etc.)."""

    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.val:{self.fmt}} (avg: {self.avg:{self.fmt}})"


class ProgressMeter:
    """Displays batch progress during training."""

    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + str(num_batches) + "]"

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(" | ".join(entries))


# =========================
# Checkpoint Utilities
# =========================
def save_checkpoint(state: dict, save_path: str):
    """Save model checkpoint."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_path)


def load_checkpoint(path: str, device=None):
    """Load model checkpoint."""
    if device is None:
        device = torch.device("cpu")
    return torch.load(path, map_location=device)


# =========================
# Misc
# =========================
def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_class_weights(class_counts: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalance."""
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)
