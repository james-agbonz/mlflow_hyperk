"""
Explainer module for the XAI pipeline.
Provides visual explanations for model predictions using Captum.
"""

import os
import sys
import argparse
import logging
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from captum.attr import GradientShap, IntegratedGradients, Occlusion, LayerGradCam
from skimage.transform import resize
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
    
    def __init__(self, root_dir, split='test', transform=None, limit=None):
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
        
        # Limit the number of samples if specified
        if limit is not None:
            self.samples = self.samples[:limit]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        with Image.open(img_path) as image:
            image = image.convert('RGB')
            original_image = np.array(image)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
                
            return image, label, img_path, original_image

def get_args():
    parser = argparse.ArgumentParser(description='Generate explanations for a trained model')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('DATA_DIR', 'data'))
    parser.add_argument('--model_path', type=str, default=os.environ.get('MODEL_PATH', None))
    parser.add_argument('--model_name', type=str, default=os.environ.get('MODEL_NAME', 'resnet18'))
    parser.add_argument('--device', type=str, default=os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--split', type=str, default=os.environ.get('SPLIT', 'test'))
    parser.add_argument('--results_dir', type=str, default=os.environ.get('RESULTS_DIR', 'results/explanations'))
    parser.add_argument('--num_samples', type=int, default=int(os.environ.get('NUM_SAMPLES', '10')))
    parser.add_argument('--methods', type=str, default=os.environ.get('METHODS', 'all'))
    parser.add_argument('--image_path', type=str, default=None, help='Optional single image for explanation')
    parser.add_argument('--summary_class', type=str, default='all', help='Class to generate CAM summary for (e.g., all, good, bad)')
    parser.add_argument('--mlflow_tracking_uri', type=str, default=os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    parser.add_argument('--experiment_name', type=str, default=os.environ.get('EXPERIMENT_NAME', 'xai_pipeline'))
    
    args = parser.parse_args()
    if not args.model_path:
        raise ValueError("MODEL_PATH is not set! Pass it via --model_path or as an environment variable")
    return args

def get_model(model_name, num_classes, model_path, device):
    """Load a model with the specified architecture and weights."""
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        last_conv_layer = model.layer4[-1]
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        last_conv_layer = model.layer4[-1]
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        last_conv_layer = model.layer4[-1]
    elif model_name == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        last_conv_layer = model.features[-1]
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
    
    # CRITICAL: Disable inplace operations in ReLU layers (required for Captum)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return model, last_conv_layer

def generate_explanations(args):
    """Generate comprehensive explanations for model predictions."""
    # Set up device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Parse methods
    if args.methods.lower() == 'all':
        methods = ['gradshap', 'ig', 'gradcam', 'occlusion']
    else:
        methods = [m.strip().lower() for m in args.methods.split(',')]
    
    logger.info(f"Using explanation methods: {', '.join(methods)}")
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = CustomDataset(
        root_dir=args.data_dir, 
        split=args.split, 
        transform=transform,
        limit=args.num_samples
    )
    
    logger.info(f"Generating explanations for {len(dataset)} samples")
    
    # Class names and number of classes
    classes = dataset.classes
    num_classes = len(classes)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time for explanations
        shuffle=False
    )
    
    # Load model
    model, last_conv_layer = get_model(args.model_name, num_classes, args.model_path, device)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up explanation methods
    explanation_methods = {}
    
    if 'gradshap' in methods:
        # Gradient SHAP
        grad_shap = GradientShap(model)
        explanation_methods['gradshap'] = grad_shap
    
    if 'ig' in methods:
        # Integrated Gradients
        integrated_grad = IntegratedGradients(model)
        explanation_methods['ig'] = integrated_grad
    
    if 'gradcam' in methods:
        # Grad-CAM
        grad_cam = LayerGradCam(model, last_conv_layer)
        explanation_methods['gradcam'] = grad_cam
    
    if 'occlusion' in methods:
        # Occlusion
        occlusion = Occlusion(model)
        explanation_methods['occlusion'] = occlusion
    
    # Process each sample
    results = []
    
    for i, (image, label, img_path, original_image) in enumerate(tqdm(dataloader, desc="Generating explanations")):
        image = image.to(device)
        label = label.item()
        img_path = img_path[0]
        
        # Fix the original_image shape - remove the batch dimension if it exists
        if len(original_image.shape) == 4:
            original_image = original_image[0]  # Remove batch dimension
        
        # Get model prediction
        with torch.no_grad():
            output = model(image)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_label = torch.argmax(output, dim=1).item()
            confidence = probs[0][pred_label].item()
        
        # File basename for saving
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Result metadata
        result = {
            'image_path': img_path,
            'true_label': classes[label],
            'predicted_label': classes[pred_label],
            'confidence': confidence,
            'correct': (pred_label == label),
            'explanations': {}
        }
        
        # Create figure for all explanations
        plt.figure(figsize=(20, 16))
        plt.suptitle(f"Explanations for {base_filename}\nTrue: {classes[label]}, Pred: {classes[pred_label]}, Conf: {confidence:.4f}", fontsize=16)
        
        # 1. Original image
        plt.subplot(2, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # 2. Gradient SHAP
        if 'gradshap' in explanation_methods:
            try:
                plt.subplot(2, 3, 2)
                # Need a baseline (reference) distribution
                baseline = torch.zeros_like(image).to(device)
                attributions = explanation_methods['gradshap'].attribute(image, baseline, target=pred_label)
                
                # Visualize attributions
                attr_np = attributions[0].cpu().detach().numpy()
                attr_np = np.transpose(attr_np, (1, 2, 0))
                
                # Sum across channels and normalize for visualization
                attr_sum = np.sum(np.abs(attr_np), axis=2)
                attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-10)
                
                plt.imshow(original_image)
                plt.imshow(attr_norm, cmap='jet', alpha=0.7)
                plt.title("Gradient SHAP")
                plt.axis('off')
                
                # Save to results
                result['explanations']['gradshap'] = {
                    'attributions_mean': np.mean(attr_np, axis=(0, 1)).tolist(),
                }
            except Exception as e:
                logger.error(f"Error generating Gradient SHAP: {str(e)}")
                plt.subplot(2, 3, 2)
                plt.text(0.5, 0.5, "Error with Gradient SHAP", ha='center')
                plt.axis('off')
                result['explanations']['gradshap'] = {
                    'error': str(e)
                }
        
        # 3. Integrated Gradients
        if 'ig' in explanation_methods:
            try:
                plt.subplot(2, 3, 3)
                baseline = torch.zeros_like(image).to(device)
                attributions = explanation_methods['ig'].attribute(image, baseline, target=pred_label)
                
                # Visualize attributions
                attr_np = attributions[0].cpu().detach().numpy()
                attr_np = np.transpose(attr_np, (1, 2, 0))
                
                # Sum across channels and normalize for visualization
                attr_sum = np.sum(np.abs(attr_np), axis=2)
                attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-10)
                
                plt.imshow(original_image)
                plt.imshow(attr_norm, cmap='jet', alpha=0.7)
                plt.title("Integrated Gradients")
                plt.axis('off')
                
                # Save to results
                result['explanations']['ig'] = {
                    'attributions_mean': np.mean(attr_np, axis=(0, 1)).tolist(),
                }
            except Exception as e:
                logger.error(f"Error generating Integrated Gradients: {str(e)}")
                plt.subplot(2, 3, 3)
                plt.text(0.5, 0.5, "Error with Integrated Gradients", ha='center')
                plt.axis('off')
                result['explanations']['ig'] = {
                    'error': str(e)
                }
        
        # 4. Occlusion
        if 'occlusion' in explanation_methods:
            try:
                plt.subplot(2, 3, 4)
                window_size = min(64, image.shape[2]//3, image.shape[3]//3)  # Use larger window for faster computation
                attributions = explanation_methods['occlusion'].attribute(
                    image,
                    target=pred_label,
                    strides=(3, max(window_size//4, 1), max(window_size//4, 1)),
                    sliding_window_shapes=(3, window_size, window_size)
                )
                
                # Visualize attributions
                attr_np = attributions[0].cpu().detach().numpy()
                attr_np = np.transpose(attr_np, (1, 2, 0))
                
                # Sum across channels and normalize for visualization
                attr_sum = np.sum(np.abs(attr_np), axis=2)
                attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-10)
                
                plt.imshow(original_image)
                plt.imshow(attr_norm, cmap='jet', alpha=0.7)
                plt.title("Occlusion")
                plt.axis('off')
                
                # Save to results
                result['explanations']['occlusion'] = {
                    'attributions_mean': np.mean(attr_np, axis=(0, 1)).tolist(),
                }
            except Exception as e:
                logger.error(f"Error generating Occlusion: {str(e)}")
                plt.subplot(2, 3, 4)
                plt.text(0.5, 0.5, "Error with Occlusion", ha='center')
                plt.axis('off')
                result['explanations']['occlusion'] = {
                    'error': str(e)
                }
        
        # 5. Grad-CAM
        if 'gradcam' in explanation_methods:
            try:
                plt.subplot(2, 3, 5)
                attributions = explanation_methods['gradcam'].attribute(image, target=pred_label)
                
                # Grad-CAM attributions are typically one value per spatial location
                attr_np = attributions[0].cpu().detach().numpy()
                
                # Average across channels (if necessary)
                if len(attr_np.shape) > 2:
                    attr_np = np.mean(attr_np, axis=0)
                
                # Normalize and resize to match image dimensions
                attr_norm = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-10)
                
                # Resize to match original image dimensions
                attr_resized = resize(attr_norm, (original_image.shape[0], original_image.shape[1]), 
                                    preserve_range=True)
                
                plt.imshow(original_image)
                plt.imshow(attr_resized, cmap='jet', alpha=0.7)
                plt.title("Grad-CAM")
                plt.axis('off')
                
                # Save to results
                result['explanations']['gradcam'] = {
                    'attributions_mean': float(np.mean(attr_np)),
                }
            except Exception as e:
                logger.error(f"Error generating Grad-CAM: {str(e)}")
                plt.subplot(2, 3, 5)
                plt.text(0.5, 0.5, "Error with Grad-CAM", ha='center')
                plt.axis('off')
                result['explanations']['gradcam'] = {
                    'error': str(e)
                }
        
        # 6. Class Probabilities
        plt.subplot(2, 3, 6)
        class_probs = probs[0].detach().cpu().numpy()
        plt.bar(range(len(classes)), class_probs)
        plt.xticks(range(len(classes)), classes)
        plt.ylabel('Probability')
        plt.title("Class Probabilities")
        
        # Save probabilities to results
        result['probabilities'] = class_probs.tolist()
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        explanation_path = os.path.join(args.results_dir, f"{base_filename}_explanations.png")
        plt.savefig(explanation_path, dpi=150)
        plt.close()
        
        # Add to results
        results.append(result)
        
        logger.info(f"Generated explanations for image {i+1}/{len(dataloader)}: {img_path}")
    
    # Save all results to JSON
    results_path = os.path.join(args.results_dir, f"explanation_results.json")
    with open(results_path, 'w') as f:
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
        
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # Generate class activation map summary if at least one sample was explained
    if results:
        logger.info("Generating class activation map summary...")
        
        try:
            # Determine which classes to summarize
            if args.summary_class.lower() == 'all':
                summary_classes = classes
            else:
                summary_classes = [args.summary_class]
            
            # Collect samples by class
            samples_by_class = {cls: [] for cls in summary_classes}
            samples_by_class_original = {cls: [] for cls in summary_classes}
            
            # Limit to 5 samples per class for summary
            for i, (image, label, _, original_image) in enumerate(dataset):
                if i >= len(results):  # Only consider samples we have explanations for
                    break
                    
                class_name = classes[label]
                if class_name in summary_classes and len(samples_by_class[class_name]) < 5:
                    samples_by_class[class_name].append(image)
                    
                    # Convert original_image to numpy if it's a tensor
                    if isinstance(original_image, torch.Tensor):
                        original_image = original_image.numpy()
                    samples_by_class_original[class_name].append(original_image)
            
            # Create a summary figure for each class
            for class_name in summary_classes:
                if not samples_by_class[class_name]:
                    continue
                
                class_idx = dataset.class_to_idx[class_name]
                
                # Create figure
                fig, axes = plt.subplots(len(samples_by_class[class_name]), 3, figsize=(15, 5 * len(samples_by_class[class_name])))
                fig.suptitle(f"Class Activation Maps for '{class_name}' class", fontsize=16)
                
                # For each sample
                for i, (sample, orig_img) in enumerate(zip(samples_by_class[class_name], samples_by_class_original[class_name])):
                    sample = sample.unsqueeze(0).to(device)
                    
                    # Get model prediction
                    with torch.no_grad():
                        output = model(sample)
                        pred_label = torch.argmax(output, dim=1).item()
                    
                    # Get different explanations
                    # Integrated Gradients
                    baseline = torch.zeros_like(sample).to(device)
                    ig_attr = explanation_methods['ig'].attribute(sample, baseline, target=class_idx)
                    
                    # Grad-CAM
                    gc_attr = explanation_methods['gradcam'].attribute(sample, target=class_idx)
                    
                    # Visualize original image
                    orig_img_np = orig_img
                    if isinstance(orig_img_np, torch.Tensor):
                        orig_img_np = orig_img_np.numpy()
                        
                    # Make sure the original image is in the right format for plotting
                    if len(orig_img_np.shape) == 2:  # Grayscale
                        orig_img_np = np.stack([orig_img_np] * 3, axis=2)
                    
                    # Integrated Gradients visualization
                    ig_attr_np = ig_attr[0].cpu().detach().numpy()
                    ig_attr_np = np.transpose(ig_attr_np, (1, 2, 0))
                    ig_attr_sum = np.sum(np.abs(ig_attr_np), axis=2)
                    ig_attr_norm = (ig_attr_sum - ig_attr_sum.min()) / (ig_attr_sum.max() - ig_attr_sum.min() + 1e-10)
                    
                    # Grad-CAM visualization
                    gc_attr_np = gc_attr[0].cpu().detach().numpy()
                    if len(gc_attr_np.shape) > 2:
                        gc_attr_np = np.mean(gc_attr_np, axis=0)
                    gc_attr_norm = (gc_attr_np - gc_attr_np.min()) / (gc_attr_np.max() - gc_attr_np.min() + 1e-10)
                    
                    # Plot
                    if len(samples_by_class[class_name]) == 1:
                        axes[0].imshow(orig_img_np)
                        axes[0].set_title("Original")
                        axes[0].axis('off')
                        
                        axes[1].imshow(orig_img_np)
                        axes[1].imshow(ig_attr_norm, cmap='jet', alpha=0.7)
                        axes[1].set_title("Integrated Gradients")
                        axes[1].axis('off')
                        
                        axes[2].imshow(orig_img_np)
                        # Resize Grad-CAM to match image size
                        gc_resized = resize(gc_attr_norm, (orig_img_np.shape[0], orig_img_np.shape[1]),
                                           preserve_range=True)
                        axes[2].imshow(gc_resized, cmap='jet', alpha=0.7)
                        axes[2].set_title("Grad-CAM")
                        axes[2].axis('off')
                    else:
                        axes[i, 0].imshow(orig_img_np)
                        axes[i, 0].set_title("Original")
                        axes[i, 0].axis('off')
                        
                        axes[i, 1].imshow(orig_img_np)
                        axes[i, 1].imshow(ig_attr_norm, cmap='jet', alpha=0.7)
                        axes[i, 1].set_title("Integrated Gradients")
                        axes[i, 1].axis('off')
                        
                        axes[i, 2].imshow(orig_img_np)
                        # Resize Grad-CAM to match image size
                        gc_resized = resize(gc_attr_norm, (orig_img_np.shape[0], orig_img_np.shape[1]),
                                           preserve_range=True)
                        axes[i, 2].imshow(gc_resized, cmap='jet', alpha=0.7)
                        axes[i, 2].set_title("Grad-CAM")
                        axes[i, 2].axis('off')
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                summary_path = os.path.join(args.results_dir, f"summary_{class_name}.png")
                plt.savefig(summary_path, dpi=150)
                plt.close()
                
                # Add summary path to results
                if 'summary_paths' not in result:
                    result['summary_paths'] = {}
                result['summary_paths'][class_name] = summary_path
        
        except Exception as e:
            logger.error(f"Error generating class activation map summary: {str(e)}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Explanations generated successfully for {len(results)} images")
    logger.info(f"Results saved to {args.results_dir}")
    
    # Log to MLflow
    try:
        # Set MLflow tracking URI
        tracking_uri = args.mlflow_tracking_uri
        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(args.experiment_name)
        
        # Start MLflow run - use existing run ID if provided
        global run_id
        if run_id:
            logger.info(f"Using existing MLflow run ID: {run_id}")
            with mlflow.start_run(run_id=run_id):
                # Log parameters
                mlflow.log_params({
                    "model_name": args.model_name,
                    "num_samples": args.num_samples,
                    "methods": args.methods,
                    "device": args.device,
                    "split": args.split,
                    "summary_class": args.summary_class
                })
                
                # Log metrics
                correct_predictions = sum(1 for result in results if result['correct'])
                accuracy = correct_predictions / len(results) if results else 0
                
                mlflow.log_metrics({
                    "num_samples_explained": len(results),
                    "accuracy_on_explained_samples": accuracy
                })
                
                # Log artifacts
                for file in os.listdir(args.results_dir):
                    if file.endswith(".png") or file.endswith(".json"):
                        artifact_path = os.path.join(args.results_dir, file)
                        mlflow.log_artifact(artifact_path)
                
                logger.info("Successfully logged explanations to MLflow")
        else:
            # Create a new run if no run ID provided
            logger.info("Starting new MLflow run...")
            with mlflow.start_run(run_name=f"explain_{args.model_name}"):
                # Log parameters
                mlflow.log_params({
                    "model_name": args.model_name,
                    "num_samples": args.num_samples,
                    "methods": args.methods,
                    "device": args.device,
                    "split": args.split,
                    "summary_class": args.summary_class
                })
                
                # Log metrics
                correct_predictions = sum(1 for result in results if result['correct'])
                accuracy = correct_predictions / len(results) if results else 0
                
                mlflow.log_metrics({
                    "num_samples_explained": len(results),
                    "accuracy_on_explained_samples": accuracy
                })
                
                # Log artifacts
                for file in os.listdir(args.results_dir):
                    if file.endswith(".png") or file.endswith(".json"):
                        artifact_path = os.path.join(args.results_dir, file)
                        mlflow.log_artifact(artifact_path)
                
                logger.info("Successfully logged explanations to MLflow")
            
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {str(e)}")
        logger.error("Continuing without MLflow logging...")
    
    return results

def generate_single_explanation(args):
    """Generate explanation for a single image."""
    # Set up device
    device = torch.device(args.device)
    
    # Parse methods
    if args.methods.lower() == 'all':
        methods = ['gradshap', 'ig', 'gradcam', 'occlusion']
    else:
        methods = [m.strip().lower() for m in args.methods.split(',')]
        
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Default classes
    classes = ['GOOD', 'BAD']
    num_classes = len(classes)
    # Load model
    model, last_conv_layer = get_model(args.model_name, num_classes, args.model_path, device)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load and preprocess the image
    try:
        image = Image.open(args.image_path).convert('RGB')
        original_image = np.array(image)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_label = torch.argmax(output, dim=1).item()
            confidence = probs[0][pred_label].item()
        
        # File basename for saving
        base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
        
        # Result metadata
        result = {
            'image_path': args.image_path,
            'predicted_label': classes[pred_label],
            'confidence': confidence,
            'explanations': {}
        }
        
        # Set up explanation methods
        explanation_methods = {}
        
        if 'gradshap' in methods:
            grad_shap = GradientShap(model)
            explanation_methods['gradshap'] = grad_shap
        
        if 'ig' in methods:
            integrated_grad = IntegratedGradients(model)
            explanation_methods['ig'] = integrated_grad
        
        if 'gradcam' in methods:
            grad_cam = LayerGradCam(model, last_conv_layer)
            explanation_methods['gradcam'] = grad_cam
        
        if 'occlusion' in methods:
            occlusion = Occlusion(model)
            explanation_methods['occlusion'] = occlusion
        
        # Create figure for all explanations
        plt.figure(figsize=(20, 16))
        plt.suptitle(f"Explanations for {base_filename}\nPred: {classes[pred_label]}, Conf: {confidence:.4f}", fontsize=16)
        
        # 1. Original image
        plt.subplot(2, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Generate explanations for each method
        explanation_positions = {
            'gradshap': (2, 3, 2),
            'ig': (2, 3, 3),
            'occlusion': (2, 3, 4),
            'gradcam': (2, 3, 5)
        }
        
        for method_name, method in explanation_methods.items():
            try:
                plt.subplot(*explanation_positions[method_name])
                
                if method_name == 'gradshap':
                    baseline = torch.zeros_like(image_tensor).to(device)
                    attributions = method.attribute(image_tensor, baseline, target=pred_label)
                elif method_name == 'ig':
                    baseline = torch.zeros_like(image_tensor).to(device)
                    attributions = method.attribute(image_tensor, baseline, target=pred_label)
                elif method_name == 'occlusion':
                    window_size = min(64, image_tensor.shape[2]//3, image_tensor.shape[3]//3)
                    attributions = method.attribute(
                        image_tensor,
                        target=pred_label,
                        strides=(3, max(window_size//4, 1), max(window_size//4, 1)),
                        sliding_window_shapes=(3, window_size, window_size)
                    )
                elif method_name == 'gradcam':
                    attributions = method.attribute(image_tensor, target=pred_label)
                
                # Visualize attributions
                attr_np = attributions[0].cpu().detach().numpy()
                
                if method_name == 'gradcam':
                    # Handle GradCAM differently
                    if len(attr_np.shape) > 2:
                        attr_np = np.mean(attr_np, axis=0)
                    
                    attr_norm = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-10)
                    attr_resized = resize(attr_norm, (original_image.shape[0], original_image.shape[1]),
                                         preserve_range=True)
                    
                    plt.imshow(original_image)
                    plt.imshow(attr_resized, cmap='jet', alpha=0.7)
                else:
                    # Handle other methods
                    attr_np = np.transpose(attr_np, (1, 2, 0))
                    attr_sum = np.sum(np.abs(attr_np), axis=2)
                    attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-10)
                    
                    plt.imshow(original_image)
                    plt.imshow(attr_norm, cmap='jet', alpha=0.7)
                
                plt.title(method_name.capitalize())
                plt.axis('off')
                
                # Save to results
                if method_name == 'gradcam':
                    result['explanations'][method_name] = {
                        'attributions_mean': float(np.mean(attr_np)),
                    }
                else:
                    result['explanations'][method_name] = {
                        'attributions_mean': np.mean(attr_np, axis=(0, 1)).tolist() 
                            if method_name != 'gradcam' else float(np.mean(attr_np)),
                    }
                
            except Exception as e:
                logger.error(f"Error generating {method_name}: {str(e)}")
                plt.subplot(*explanation_positions[method_name])
                plt.text(0.5, 0.5, f"Error with {method_name}", ha='center')
                plt.axis('off')
                result['explanations'][method_name] = {
                    'error': str(e)
                }
        
        # Class Probabilities
        plt.subplot(2, 3, 6)
        class_probs = probs[0].detach().cpu().numpy()
        plt.bar(range(len(classes)), class_probs)
        plt.xticks(range(len(classes)), classes)
        plt.ylabel('Probability')
        plt.title("Class Probabilities")
        
        # Save probabilities to results
        result['probabilities'] = class_probs.tolist()
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        explanation_path = os.path.join(args.results_dir, f"{base_filename}_single_explanation.png")
        plt.savefig(explanation_path, dpi=150)
        plt.close()
        
        # Add the path to the result
        result['explanation_image_path'] = explanation_path
        
        # Log to MLflow
        try:
            # Set MLflow tracking URI
            tracking_uri = args.mlflow_tracking_uri
            logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set experiment
            mlflow.set_experiment(args.experiment_name)
            
            # Start MLflow run - use existing run ID if provided
            global run_id
            if run_id:
                logger.info(f"Using existing MLflow run ID: {run_id}")
                with mlflow.start_run(run_id=run_id):
                    # Log parameters
                    mlflow.log_params({
                        "model_name": args.model_name,
                        "methods": args.methods,
                        "device": args.device,
                        "image_path": args.image_path
                    })
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "confidence": confidence
                    })
                    
                    # Log artifacts
                    mlflow.log_artifact(explanation_path)
                    
                    # Log result as JSON
                    result_path = os.path.join(args.results_dir, f"{base_filename}_result.json")
                    with open(result_path, 'w') as f:
                        class NumpyEncoder(json.JSONEncoder):
                            def default(self, obj):
                                if isinstance(obj, np.integer):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                return super(NumpyEncoder, self).default(obj)
                        
                        json.dump(result, f, indent=4, cls=NumpyEncoder)
                    
                    mlflow.log_artifact(result_path)
                    
                    logger.info("Successfully logged single explanation to MLflow")
            else:
                # Start a new run
                with mlflow.start_run(run_name=f"explain_single_{base_filename}"):
                    # Log parameters
                    mlflow.log_params({
                        "model_name": args.model_name,
                        "methods": args.methods,
                        "device": args.device,
                        "image_path": args.image_path
                    })
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "confidence": confidence
                    })
                    
                    # Log artifacts
                    mlflow.log_artifact(explanation_path)
                    
                    # Log result as JSON
                    result_path = os.path.join(args.results_dir, f"{base_filename}_result.json")
                    with open(result_path, 'w') as f:
                        class NumpyEncoder(json.JSONEncoder):
                            def default(self, obj):
                                if isinstance(obj, np.integer):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                return super(NumpyEncoder, self).default(obj)
                        
                        json.dump(result, f, indent=4, cls=NumpyEncoder)
                    
                    mlflow.log_artifact(result_path)
                    
                    logger.info("Successfully logged single explanation to MLflow")
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {str(e)}")
            logger.error("Continuing without MLflow logging...")
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating explanation for single image: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    args = get_args()

    # If a single image path is provided, generate explanation for that image
    if args.image_path:
        result = generate_single_explanation(args)
        print(json.dumps(result, indent=2))
    else:
        # Otherwise, generate explanations for samples from the dataset
        results = generate_explanations(args)
