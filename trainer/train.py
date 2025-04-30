import sys
import os
import logging
from pathlib import Path
os.environ["MLFLOW_TEMP_DIR"] = "/app/tmp"
sys.path.append("/app")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from augmenter.cutmix import cutmix
from torchvision.datasets import ImageFolder
from PIL import UnidentifiedImageError
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/artifacts/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_mlflow_connection():
    """Ensure MLflow server is reachable"""
    try:
        mlflow.search_experiments()
        logger.info("✅ Successfully connected to MLflow server")
        return True
    except Exception as e:
        logger.error(f"❌ MLflow connection failed: {str(e)}")
        return False

def setup_artifacts_dir():
    """Create and verify artifacts directory"""
    artifacts_dir = Path("/app/artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Test write permission
    test_file = artifacts_dir / "permission_test.txt"
    try:
        test_file.write_text("test")
        test_file.unlink()
        logger.info(f"✅ Artifacts directory is writable: {artifacts_dir}")
        return True
    except Exception as e:
        logger.error(f"❌ Artifacts directory error: {str(e)}")
        return False

# Configuration
config = {
    "batch_size": 8,
    "epochs": 3,
    "learning_rate": 0.001,
    "augmentation": "cutmix",
    "model_architecture": "resnet18",
    "optimizer": "adam",
    "mlflow_uri": "http://mlflow:5000"
}

def train_model():
    # Verify system readiness
    if not all([verify_mlflow_connection(), setup_artifacts_dir()]):
        raise RuntimeError("System verification failed")

    # Data pipeline
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    class SafeImageFolder(ImageFolder):
        def __getitem__(self, index):
            try:
                return super().__getitem__(index)
            except (UnidentifiedImageError, OSError) as e:
                logger.warning(f"Skipping bad image at index {index}: {e}")
                return self.__getitem__((index + 1) % len(self))

    # Initialize MLflow
    mlflow.set_tracking_uri(config["mlflow_uri"])
    mlflow.set_experiment("CutMixExperiment")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started training run: {run_id}")
        
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("device", str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        # Model setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        # Data loading
        train_dataset = SafeImageFolder(root="/app/data/train", transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

        # Training loop
        accuracies = []
        losses = []
        
        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0.0
            correct = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                images, y_a, y_b, lam = cutmix(images, labels)

                optimizer.zero_grad()
                outputs = model(images)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (lam * predicted.eq(y_a.data).cpu().sum().float() +
                          (1 - lam) * predicted.eq(y_b.data).cpu().sum().float())

            # Log epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / len(train_dataset)
            losses.append(epoch_loss)
            accuracies.append(epoch_acc)
            
            mlflow.log_metric("loss", epoch_loss, step=epoch)
            mlflow.log_metric("accuracy", epoch_acc, step=epoch)
            logger.info(f"Epoch {epoch+1}/{config['epochs']} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Save artifacts
        # Training curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, config["epochs"] + 1), accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, config["epochs"] + 1), losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plot_path = "/app/artifacts/training_plot.png"
        plt.savefig(plot_path)
        plt.close()

        # Model summary
        summary_path = "/app/artifacts/model_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(str(model))

        # Sample images
        plt.figure(figsize=(10, 10))
        sample_images, _ = next(iter(train_loader))
        for i, img in enumerate(sample_images[:4]):
            plt.subplot(2, 2, i+1)
            plt.imshow(img.permute(1, 2, 0))
            plt.axis('off')
        samples_path = "/app/artifacts/sample_images.png"
        plt.savefig(samples_path)
        plt.close()

        # Model registration
        input_example = torch.randn(1, 3, 64, 64)
        signature = infer_signature(
            input_example.numpy(),
            model(input_example.to(device)).cpu().detach().numpy()
        )

        mlflow.pytorch.log_model(
            model,
            "cutmix-model",
            signature=signature,
            input_example=input_example.numpy(),
            registered_model_name="CutMix_Model"
        )

        logger.info(f"Training completed successfully! Run ID: {run_id}")

        # NEW: Log all artifacts at once
        artifacts_dir = "/app/artifacts"
        # if os.path.exists(artifacts_dir):
        mlflow.log_artifacts(artifacts_dir)
        #     logger.info(f"✅ Logged all artifacts from {artifacts_dir}")
        # else:
        #     logger.warning(f"❌ Artifacts directory {artifacts_dir} does not exist")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise



        # # Save the model
        # model_path = "artifacts/simple_model.pth"
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # torch.save(model.state_dict(), model_path)

        # # Log the model artifact
        # mlflow.log_artifact(model_path)

        # # Save run ID for other services
        # with open("/app/run_id.txt", "w") as f:
        #     f.write(run.info.run_id)

