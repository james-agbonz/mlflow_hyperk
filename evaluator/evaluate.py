"""
Evaluation module for the XAI pipeline.
Provides model evaluation with metrics and visualizations.
"""

import os
import sys
import argparse
import logging
import json
import time
import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import mlflow
import mlflow.pytorch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if run ID is provided for MLflow
run_id = os.environ.get('MLFLOW_RUN_ID')
if run_id:
    logger.info(f"Will use existing MLflow run ID: {run_id}")
else:
    # Try to read the run ID from the file if not provided in environment
    run_id_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "latest_run_id.txt")
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
        logger.info(f"Read MLflow run ID from file: {run_id}")
    else:
        logger.info("No MLflow run ID provided or found - will create new run")

# Custom dataset class
class CustomDataset(Dataset):
    """Custom dataset for image classification with 'GOOD' and 'BAD' classes"""
    
    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = ['GOOD', 'BAD']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                logger.warning(f"Directory not found: {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path  # Return path as well for identification

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('DATA_DIR', 'data'))
    parser.add_argument('--model_path', type=str, default=os.environ.get('MODEL_PATH', None))
    parser.add_argument('--model_name', type=str, default=os.environ.get('MODEL_NAME', 'resnet18'))
    parser.add_argument('--batch_size', type=int, default=int(os.environ.get('BATCH_SIZE', '32')))
    parser.add_argument('--num_workers', type=int, default=int(os.environ.get('NUM_WORKERS', '4')))
    parser.add_argument('--device', type=str, default=os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--results_dir', type=str, default=os.environ.get('RESULTS_DIR', 'results/evaluation'))
    parser.add_argument('--split', type=str, default=os.environ.get('SPLIT', 'test'))
    parser.add_argument('--experiment_name', type=str, default=os.environ.get('EXPERIMENT_NAME', 'xai_pipeline'))
    
    args = parser.parse_args()
    
    if not args.model_path:
        raise ValueError("MODEL_PATH is not set! Pass it via environment variable or --model_path")
    
    return args

def get_model(model_name, num_classes, model_path, device):
    """Load a model with the specified architecture and weights."""
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # If model_path is a directory (from MLflow), look for a .pth file
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model weights from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model weights: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return model

def evaluate_model(args):
    """Evaluate a trained model and generate comprehensive metrics."""
    start_time = time.time()
    
    # Set up device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Set up transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    test_dataset = CustomDataset(
        root_dir=args.data_dir, 
        split=args.split, 
        transform=test_transform
    )
    
    logger.info(f"Dataset size ({args.split} split): {len(test_dataset)}")
    
    # Class names and number of classes
    classes = test_dataset.classes
    num_classes = len(classes)
    logger.info(f"Classes: {classes}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load model
    model = get_model(args.model_name, num_classes, args.model_path, device)
    
    # Prepare for evaluation
    criterion = nn.CrossEntropyLoss()
    all_predictions = []
    all_targets = []
    all_probs = []
    all_image_paths = []
    incorrect_predictions = []
    test_loss = 0.0
    
    # Get model information
    model_info = {
        'model_name': args.model_name,
        'model_path': args.model_path,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    # Evaluation loop
    with torch.no_grad():
        for images, targets, img_paths in tqdm(test_loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Get predictions and probabilities
            # Get predictions and probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_image_paths.extend(img_paths)
            
            # Store incorrect predictions
            incorrect_mask = (predicted != targets).cpu().numpy()
            for i, is_incorrect in enumerate(incorrect_mask):
                if is_incorrect:
                    incorrect_predictions.append({
                        'image_path': img_paths[i],
                        'true_label': classes[targets[i].item()],
                        'predicted_label': classes[predicted[i].item()],
                        'confidence': probs[i][predicted[i]].item()
                    })
    
    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Basic metrics
    accuracy = np.mean(all_predictions == all_targets) * 100
    avg_test_loss = test_loss / len(test_loader)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Classification report
    cr = classification_report(all_targets, all_predictions, 
                              target_names=classes, 
                              output_dict=True)
    
    # ROC curve and AUC (for binary classification)
    roc_auc = None
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(all_targets, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve (for each class)
    pr_curves = {}
    for i in range(num_classes):
        # Convert to binary classification problem (one-vs-rest)
        binary_targets = (all_targets == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_targets, all_probs[:, i])
        pr_auc = auc(recall, precision)
        pr_curves[classes[i]] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'auc': pr_auc
        }
    
    # Performance metrics
    evaluation_time = time.time() - start_time
    inference_time = evaluation_time / len(test_dataset)
    
    # Create metadata
    metadata = {
        'evaluation_datetime': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': {
            'path': args.data_dir,
            'split': args.split,
            'size': len(test_dataset),
            'classes': classes,
        },
        'model': model_info,
        'environment': {
            'device': str(device),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        },
        'performance': {
            'evaluation_time_seconds': evaluation_time,
            'inference_time_seconds_per_image': inference_time,
            'batch_size': args.batch_size,
        },
        'metrics': {
            'accuracy': accuracy,
            'avg_loss': avg_test_loss,
            'classification_report': cr,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'pr_curves': pr_curves,
        },
        'incorrect_predictions': incorrect_predictions[:20],  # Limit to first 20
    }
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save metadata to JSON
    metadata_path = os.path.join(args.results_dir, 'evaluation_results.json')
    with open(metadata_path, 'w') as f:
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    
    # Generate and save visualizations
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(args.results_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC curve (for binary classification)
    if num_classes == 2:
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.results_dir, 'roc_curve.png'))
        plt.close()
    
    # Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        pr_data = pr_curves[class_name]
        plt.plot(pr_data['recall'], pr_data['precision'], lw=2,
                label=f'{class_name} (area = {pr_data["auc"]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.savefig(os.path.join(args.results_dir, 'precision_recall_curves.png'))
    plt.close()
    
    # Log summary results
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Average Loss: {avg_test_loss:.4f}")
    logger.info(f"Results saved to {args.results_dir}")

    # Print class-wise metrics
    logger.info("Class-wise metrics:")
    for cls_name in classes:
        logger.info(f"  {cls_name}:")
        logger.info(f"    Precision: {cr[cls_name]['precision']:.4f}")
        logger.info(f"    Recall: {cr[cls_name]['recall']:.4f}")
        logger.info(f"    F1-score: {cr[cls_name]['f1-score']:.4f}")

    # MLflow logging
    try:
        # Print the tracking URI for debugging
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        logger.info(f"MLflow Tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set the experiment
        experiment_name = args.experiment_name
        logger.info(f"Setting experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

        # Start a run - use existing run ID if provided
        global run_id
        if run_id:
            logger.info(f"Using existing MLflow run ID: {run_id}")
            with mlflow.start_run(run_id=run_id):
                # Log parameters
                logger.info("Logging parameters...")
                mlflow.log_params({
                    "model_name": args.model_name,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "device": args.device,
                    "split": args.split,
                    "dataset_size": len(test_dataset),
                    "model_path": args.model_path
                })

                # Log metrics
                logger.info("Logging metrics...")
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "avg_test_loss": avg_test_loss
                })

                if roc_auc is not None:
                    mlflow.log_metric("roc_auc", roc_auc)

                # Log artifacts
                logger.info("Logging artifacts...")
                mlflow.log_artifact(metadata_path)
                mlflow.log_artifact(os.path.join(args.results_dir, 'confusion_matrix.png'))
                if num_classes == 2:
                    mlflow.log_artifact(os.path.join(args.results_dir, 'roc_curve.png'))
                mlflow.log_artifact(os.path.join(args.results_dir, 'precision_recall_curves.png'))

                logger.info("Logged evaluation artifacts to MLflow")
        else:
            # Create a new run if no run ID provided
            logger.info("Starting new MLflow run...")
            with mlflow.start_run(run_name=f"evaluation_{args.model_name}"):
                # Log parameters
                logger.info("Logging parameters...")
                mlflow.log_params({
                    "model_name": args.model_name,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "device": args.device,
                    "split": args.split,
                    "dataset_size": len(test_dataset),
                    "model_path": args.model_path
                })

                # Log metrics
                logger.info("Logging metrics...")
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "avg_test_loss": avg_test_loss
                })

                if roc_auc is not None:
                    mlflow.log_metric("roc_auc", roc_auc)

                # Log artifacts
                logger.info("Logging artifacts...")
                mlflow.log_artifact(metadata_path)
                mlflow.log_artifact(os.path.join(args.results_dir, 'confusion_matrix.png'))
                if num_classes == 2:
                    mlflow.log_artifact(os.path.join(args.results_dir, 'roc_curve.png'))
                mlflow.log_artifact(os.path.join(args.results_dir, 'precision_recall_curves.png'))

                logger.info("Logged evaluation artifacts to MLflow")

    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")
        logger.error("Continuing without MLflow logging")

    return metadata

if __name__ == "__main__":
    args = get_args()
    metadata = evaluate_model(args)
