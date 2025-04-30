import mlflow
from mlflow.tracking import MlflowClient
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/artifacts/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_latest_run_id():
    """Get the latest run ID from MLflow"""
    try:
        client = MlflowClient(tracking_uri="http://mlflow:5000")
        experiments = client.search_experiments()
        if not experiments:
            logger.error("No experiments found in MLflow")
            return None
            
        # Search in all experiments
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs:
                logger.info(f"Found latest run: {runs[0].info.run_id} in experiment {exp.name}")
                return runs[0].info.run_id
                
        logger.error("No runs found in any experiments")
        return None
    except Exception as e:
        logger.error(f"Error getting latest run ID: {str(e)}")
        return None

def evaluate_model(run_id):
    """Evaluate the model and return metrics"""
    logger.info(f"Evaluating model for run ID: {run_id}")
    
    try:
        # Try different model paths
        model = None
        for model_path in ["cutmix-model", "model"]:
            try:
                model = mlflow.pytorch.load_model(f"runs:/{run_id}/{model_path}")
                logger.info(f"Successfully loaded model from path: {model_path}")
                break
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {str(e)}")
        
        if model is None:
            # Try searching for models in artifacts
            client = MlflowClient(tracking_uri="http://mlflow:5000")
            artifacts = client.list_artifacts(run_id)
            model_paths = [a.path for a in artifacts if 'model' in a.path.lower()]
            if model_paths:
                model = mlflow.pytorch.load_model(f"runs:/{run_id}/{model_paths[0]}")
                logger.info(f"Loaded model from discovered path: {model_paths[0]}")
            else:
                raise ValueError(f"No model artifacts found in run {run_id}")
        
        model.eval()
        
        # Set up transformation pipeline
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        # Load test data
        try:
            test_data = datasets.ImageFolder("/app/data/test", transform=transform)
            loader = DataLoader(test_data, batch_size=32, shuffle=False)
            logger.info(f"Loaded test data with {len(test_data)} images")
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
        
        # Predictions
        y_true, y_pred, y_scores = [], [], []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate confusion matrix visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_path = "/app/artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        # Log metrics to MLflow
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1", f1)
                mlflow.log_artifact(cm_path)
            logger.info(f"Successfully logged evaluation metrics to MLflow")
        except Exception as e:
            logger.warning(f"Could not log metrics to MLflow: {str(e)}")
        
        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix_path": cm_path
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    run_id = get_latest_run_id()
    if run_id:
        try:
            metrics = evaluate_model(run_id)
            print(f"Evaluation complete: {metrics}")
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
    else:
        print("No runs found to evaluate")
