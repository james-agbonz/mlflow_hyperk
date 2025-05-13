# XAI Pipeline with CutMix Augmentation

This project implements a comprehensive explainable AI (XAI) pipeline for image classification with CutMix augmentation. The pipeline includes model training, evaluation, explanation generation, and visualization through a dashboard, all tracked with MLflow.

## Project Overview

The XAI Pipeline is designed to:

1. **Train** models using CutMix data augmentation for improved performance
2. **Evaluate** trained models with comprehensive metrics
3. **Explain** model predictions using various visualization techniques
4. **Visualize** all results in an interactive dashboard
5. **Track** experiments with MLflow

## Project Structure

```
xai_pipeline/
│
├── data/                       # Contains 'train/', 'test/' with GOOD/BAD subfolders
│
├── augmenter/
│   └── cutmix.py              # Contains CutMix augmentation logic
│
├── trainer/
│   └── train.py               # Trains model with CutMix
│
├── evaluator/
│   └── evaluate.py            # Evaluates the trained model
│
├── explainer/
│   └── explain.py             # XAI explanations (GradCAM, SHAP, etc.)
│
├── dashboard/
│   └── dashboard.py           # Streamlit dashboard for visualization
│
├── Dockerfiles/
│   ├── Dockerfile.train       # Dockerfile for training container
│   ├── Dockerfile.eval        # Dockerfile for evaluation container
│   ├── Dockerfile.explain     # Dockerfile for explanation container
│   ├── Dockerfile.dashboard   # Dockerfile for dashboard container
│   └── Dockerfile.mlflow      # Dockerfile for MLflow tracking server
│
├── docker-compose.yml         # Compose file to run services together
│
├── kubernetes/
│   ├── trainer.yaml           # Kubernetes deployment for trainer
│   ├── evaluator.yaml         # Kubernetes deployment for evaluator
│   ├── explainer.yaml         # Kubernetes deployment for explainer
│   ├── dashboard.yaml         # Kubernetes deployment for dashboard
│   └── mlflow.yaml            # Kubernetes deployment for MLflow
│
├── .github/
│   └── workflows/
│       ├── docker-build.yml   # GitHub workflow for Docker builds
│       └── k8s-deploy.yml     # GitHub workflow for Kubernetes deployment
│
├── run-pipeline.bat           # Windows batch script to run the full pipeline
│
├── requirements.txt           # Common dependencies
│
└── README.md                  # This file
```

## Components

### CutMix Augmentation

[CutMix](https://arxiv.org/abs/1905.04899) is a data augmentation technique that creates new training samples by cutting and pasting patches between training images while mixing the target labels proportionally to the area of patches.

Our implementation in `augmenter/cutmix.py` applies this technique to improve model robustness and performance.

### Training Module

The training module (`trainer/train.py`) implements:
- Model training with CutMix augmentation
- Configurable model architectures (ResNet18, ResNet34, ResNet50, MobileNetV2)
- Learning rate and batch size configuration
- MLflow tracking for parameters, metrics, and model artifacts
- Docker containerization for reproducibility

### Evaluation Module

The evaluation module (`evaluator/evaluate.py`) provides:
- Comprehensive model evaluation metrics
- Confusion matrix generation
- ROC and precision-recall curves
- Performance analysis
- MLflow logging of evaluation results

### Explainer Module

The explainer module (`explainer/explain.py`) generates:
- Multiple explanation types (GradientSHAP, Integrated Gradients, GradCAM, Occlusion)
- Visual explanations for model predictions
- Class activation map summaries
- MLflow logging of explanations

### Dashboard Module

The dashboard module (`dashboard/dashboard.py`) offers:
- Interactive visualization of training, evaluation, and explanation results
- Integration with MLflow for experiment tracking
- Class-wise metrics and explanation visualization
- Complete pipeline execution from the UI

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (if running locally)
- CUDA-capable GPU (recommended for training)
- Kubernetes cluster (optional, for deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xai_pipeline.git
cd xai_pipeline
```

2. Prepare your dataset with the following structure:
```
data/
├── train/
│   ├── GOOD/
│   └── BAD/
└── test/
    ├── GOOD/
    └── BAD/
```

3. Build the Docker images:
```bash
docker-compose build
```

## Usage

### Using Docker Compose

Start the MLflow tracking server:

```bash
docker-compose up -d mlflow
```

Run the full pipeline:

```bash
docker-compose run --rm trainer
docker-compose run --rm evaluator
docker-compose run --rm explainer
```

Alternatively, use the provided batch script on Windows:

```bash
.\run-pipeline.bat
```

Start the dashboard:

```bash
docker-compose up dashboard
```

Access the dashboard at http://localhost:8501

### Using Kubernetes

Apply the Kubernetes manifests:

```bash
kubectl apply -f kubernetes/
```

### Environment Variables

The following environment variables can be used to configure the pipeline:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Directory containing the dataset | `/app/data` |
| `BATCH_SIZE` | Batch size for training | `32` |
| `EPOCHS` | Number of epochs to train | `20` |
| `LEARNING_RATE` | Learning rate | `0.001` |
| `CUTMIX_PROB` | Probability of applying CutMix | `0.5` |
| `CUTMIX_BETA` | Beta parameter for CutMix | `1.0` |
| `MODEL_NAME` | Model architecture | `resnet18` |
| `SAVE_DIR` | Directory to save models | `/app/models` |
| `RESULTS_DIR` | Directory to save results | `/app/results` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `http://mlflow:5000` |
| `EXPERIMENT_NAME` | MLflow experiment name | `xai_pipeline` |
| `DEVICE` | Device to use (cuda/cpu) | `cuda` if available |

## Shared Run ID Support

The pipeline supports a shared MLflow run ID across all components, allowing unified tracking of the entire process. The run ID is generated during training and passed to evaluation and explanation components.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [CutMix Paper](https://arxiv.org/abs/1905.04899): "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Captum Library for XAI](https://captum.ai/)
- [Streamlit for Interactive Dashboards](https://streamlit.io/)
