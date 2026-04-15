# Catastrophic Forgetting in Healthcare AI - DEAR Framework

A research hackathon prototype implementing the **DEAR framework** (Drift-Aware EWC with Adaptive Replay) for continual learning on Chest X-ray datasets.

## Overview

This project addresses catastrophic forgetting in medical image classification through:

- **Task-Aware Elastic Weight Consolidation (TA-EWC)**: Protects clinically critical parameters while allowing task-specific adaptation
- **Drift Detection**: Page-Hinkley test on embedding streams to trigger adaptive replay
- **Privacy-Preserving Replay Buffer**: Stores only embeddings (not raw images) for HIPAA compliance

## Project Structure

```
catastrophic-forgetting-healthcare-ai/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── config/
│   └── config.yaml                    # Hyperparameters and settings
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── utils.py                       # Utility functions
│   ├── data_loader.py                 # Data pipeline (Kaggle download + preprocessing)
│   ├── models.py                      # ResNet/DenseNet with MC Dropout
│   ├── ta_ewc.py                      # Task-Aware EWC regularization
│   ├── drift_detection.py             # Page-Hinkley drift detector
│   ├── replay_buffer.py               # Privacy-preserving embedding buffer
│   ├── metrics.py                     # Evaluation metrics
│   └── utils.py                       # Helper functions
├── main.py                            # Entry point
├── train.py                           # Training loop
├── evaluate.py                        # Evaluation and visualization
├── notebooks/
   └── exploratory_analysis.ipynb     # Dataset exploration

```

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- Kaggle API credentials

### Setup

1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd catastrophic-forgetting-healthcare-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Kaggle API**
   - Download API token from https://www.kaggle.com/settings/account
   - Place `kaggle.json` in `~/.kaggle/`
   - Run `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

## Dataset

**Chest X-ray COVID-19 Pneumonia Dataset**
- Source: [Kaggle - Chest X-ray COVID19 Pneumonia](https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset)
- Classes: NORMAL, PNEUMONIA, COVID-19
- Task Definition: Sequential arrival of new disease classes (simulating continual learning scenario)
  - **Task 1**: Normal vs. Pneumonia
  - **Task 2**: Add COVID-19

### Download & Prepare

The dataset is automatically downloaded via Kaggle API when running `main.py`. Ensure Kaggle credentials are configured first.

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model
model.backbone: "resnet18"        # resnet18, resnet50, densenet121
model.embedding_dim: 512          # Embedding dimension
model.mc_dropout_samples: 20      # MC Dropout samples for uncertainty

# Training
training.epochs_per_task: 50      # Epochs per task
training.batch_size: 32           # Batch size
training.learning_rate: 0.001     # Learning rate

# EWC
ewc.lambda_cc: 1000.0            # Clinically Critical penalty
ewc.lambda_sr: 500.0             # Shared Representational penalty
ewc.lambda_tp: 100.0             # Task Peripheral penalty

# Drift Detection
drift_detection.drift_threshold: 3.0   # Page-Hinkley threshold
drift_detection.window_size: 50        # Window for drift detection

# Replay Buffer
replay_buffer.buffer_size: 1000        # Max embeddings stored
replay_buffer.priority_weights:
  uncertainty: 0.5                     # Uncertainty score weight
  recency: 0.3                         # Recency score weight
  diversity: 0.2                       # Diversity score weight
```

## Training

### Run Full Pipeline

```bash
python main.py
```

This will:
1. Download Chest X-ray dataset from Kaggle
2. Preprocess images (resize, normalize, augment)
3. Split into sequential tasks
4. Train with TA-EWC regularization
5. Detect drift and trigger replay
6. Evaluate all tasks
7. Generate plots and metrics

### Custom Training

```python
from train import ContinualLearningTrainer
from src import load_config, get_device

config = load_config("config/config.yaml")
device = get_device(config['hardware']['device'])
trainer = ContinualLearningTrainer(config, device, logger)

# Train task
trainer.train_task(task_id=0, train_loader, val_loader, classes)
```

## Results
![alt text](image.png)

### Key Metrics

1. **Forward Transfer (FT)**: `A_t^t - A_t^0` 
   - Performance on new task vs. random baseline
   - Higher is better

2. **Backward Transfer (BT)**: `A_i^t - A_i^{t-1}`
   - Change in accuracy on old tasks after learning new task
   - Less negative is better (less forgetting)

3. **Forgetting**: `max_accuracy - final_accuracy`
   - Performance drop on old tasks
   - Lower is better

4. **Average Accuracy**: Mean accuracy across all tasks
   - Measures overall model performance

## Architecture Details

### Model

```
Input Image (3, 224, 224)
     ↓
Backbone (ResNet-18)
     ↓
Global Average Pooling
     ↓
Embedding Layer (512-dim)
     ↓
MC Dropout (p=0.3)
     ↓
Classification Head (num_classes)
     ↓
Logits & Embeddings
```

### Training Loop

```
For each task:
  1. Train with classification loss
  2. Add TA-EWC regularization loss
  3. Extract embeddings for drift detection
  4. Compute MC Dropout uncertainty
  5. Check for drift (Page-Hinkley test)
  6. If drift detected:
     - Sample from replay buffer
     - Retrain on replayed embeddings
  7. Compute Fisher Information Matrix
  8. Populate replay buffer with current task samples
  9. Evaluate on all seen tasks
```

### Parameter Importance Classification

```
For each parameter:
  Fisher Score = mean of Fisher Information
  Variance = variance across tasks
  
  If Fisher_high AND Variance_high:
    → Clinically Critical (CC)        λ = 1000.0
  Elif Fisher_moderate OR Variance_low:
    → Shared Representational (SR)    λ = 500.0
  Else:
    → Task Peripheral (TP)            λ = 100.0
```

### Replay Buffer Priority

```
Priority = α·U_i + β·R_i + γ·D_i

U_i = Predictive Uncertainty
      (MC Dropout variance)

R_i = Recency Score
      (inverse of age)

D_i = Diversity Score
      (average distance to nearest neighbors)
```

## Privacy Considerations

- **Embedding-Only Storage**: Replay buffer stores only 512-dim embeddings, not raw images
  - Reduces privacy risk (images cannot be reconstructed)
  - Complies with HIPAA guidelines
  - Reduces memory footprint (100x reduction)

- **No Patient ID Storage**: Clinical identifiers removed during preprocessing

## Evaluation Results
<img width="1600" height="557" alt="image" src="https://github.com/user-attachments/assets/f46665ea-5345-4ef8-b5eb-c79c5d8d62f1" />

## Troubleshooting

### Kaggle Download Fails
```bash
# Verify credentials
ls ~/.kaggle/kaggle.json

# Test API
kaggle datasets list

# Manual download: https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset
# Extract to ./data/raw/
```

### Out of Memory
- Reduce `batch_size` in config
- Reduce `buffer_size` in replay buffer config
- Use smaller model (`resnet18` instead of `resnet50`)

### Slow Training
- Enable CUDA: Set `hardware.device: cuda`
- Increase `num_workers` in config
- Use mixed precision (add `--amp` flag)

## Citation

If using this code, please cite:

```bibtex
@inproceedings{dear2024,
  title={DEAR: Drift-Aware EWC with Adaptive Replay for Continual Learning in Healthcare},
  author={Your Name},
  booktitle={ML Hackathon 2024},
  year={2024}
}
```

## References

- **Elastic Weight Consolidation**: Kirkpatrick et al. (2017)
- **Page-Hinkley Drift Detector**: Page & Hinkley (1961)
- **MC Dropout**: Gal & Ghahramani (2016)

## License

MIT License - See LICENSE file for details

