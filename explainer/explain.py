import mlflow
from mlflow.tracking import MlflowClient
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import json
import logging
import shap
from captum.attr import IntegratedGradients, visualization
from PIL import Image
import io
import base64
import tempfile
import traceback
from flask import Flask, request, jsonify, send_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/artifacts/explanation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)
PLOT_DIR = "/app/artifacts/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

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

# Add this function that was being imported in app.py
def get_run_id_from_file(file_path="/app/artifacts/latest_run_id.txt"):
    """Get run ID from a file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                run_id = f.read().strip()
                logger.info(f"Read run ID from file: {run_id}")
                return run_id
        else:
            logger.warning(f"Run ID file not found: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error reading run ID from file: {str(e)}")
        return None

def load_model(run_id):
    """Load a model from MLflow by run ID."""
    # Try multiple possible paths for the model
    model = None
    paths_to_try = ["cutmix-model", "model", "models/CutMix_Model/version-52"]
    
    for model_path in paths_to_try:
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            logger.info(f"Attempting to load model from: {model_uri}")
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load from {model_path}: {str(e)}")
            continue
    
    # If we've tried all paths and failed, try to find the model in artifacts
    try:
        client = MlflowClient(tracking_uri="http://mlflow:5000")
        artifacts = client.list_artifacts(run_id)
        model_paths = [a.path for a in artifacts if 'model' in a.path.lower()]
        if model_paths:
            model_uri = f"runs:/{run_id}/{model_paths[0]}"
            logger.info(f"Attempting to load model from discovered path: {model_uri}")
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Successfully loaded model from discovered path: {model_paths[0]}")
            return model
    except Exception as e:
        logger.error(f"Failed to discover model path: {str(e)}")
    
    raise ValueError(f"Could not find model for run {run_id}")

def load_data(batch_size=32):
    """Load a small subset of data for explanations"""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    try:
        data_dir = "/app/data/train"
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        logger.info(f"Loaded dataset with {len(dataset)} images from {data_dir}")
        
        # Create a smaller subset for explanations
        indices = torch.randperm(len(dataset))[:100]
        subset = torch.utils.data.Subset(dataset, indices)
        
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        return loader, dataset.classes
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def compute_feature_importance(model, data_loader, num_samples=5):
    """Compute feature importance using SHAP"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get a batch of images for background
    background = []
    for images, _ in data_loader:
        background.append(images[:10])
        if len(background) >= 2:  # Limit number of background samples
            break
    background = torch.cat(background, dim=0).to(device)
    
    # Initialize SHAP explainer
    explainer = shap.DeepExplainer(model, background)
    
    # Get images to explain
    samples = []
    sample_labels = []
    sample_paths = []
    for images, labels in data_loader:
        for i in range(min(num_samples, len(images))):
            samples.append(images[i:i+1])
            sample_labels.append(labels[i].item())
        if len(samples) >= num_samples:
            break
    
    # Compute SHAP values
    shap_values = []
    channel_importance = {'r_importance': [], 'g_importance': [], 'b_importance': []}
    
    try:
        for i, sample in enumerate(samples):
            sample = sample.to(device)
            batch_shap_values = explainer.shap_values(sample)
            
            # Extract channel importance
            abs_shap = [np.abs(s) for s in batch_shap_values]
            
            # For binary classification, we often get one class's shap values
            if len(abs_shap) == 1:
                abs_shap = abs_shap[0]
                
            # Calculate channel importance (R, G, B)
            if len(abs_shap.shape) == 4:  # [batch, channel, height, width]
                for c, channel in enumerate(['r', 'g', 'b']):
                    imp = np.mean(abs_shap[0, c, :, :])
                    channel_importance[f'{channel}_importance'].append(float(imp))
            
            shap_values.append(batch_shap_values)
            
    except Exception as e:
        logger.error(f"Error computing SHAP values: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Compute average channel importance
    avg_importance = {}
    for channel, values in channel_importance.items():
        if values:
            avg_importance[channel] = float(np.mean(values))
    
    # Generate a SHAP summary plot
    shap_plot_path = None
    if shap_values:
        try:
            plt.figure(figsize=(10, 6))
            # Convert to numpy for plotting
            sample_images = [s.cpu().numpy() for s in samples]
            
            # For first class (typically in binary classification)
            if isinstance(shap_values[0], list):
                plot_shap_values = [sv[0] for sv in shap_values]
            else:
                plot_shap_values = shap_values
                
            # Create summary plot
            shap.image_plot(plot_shap_values, np.array(sample_images))
            
            # Save plot
            shap_plot_path = os.path.join(PLOT_DIR, "shap_summary.png")
            plt.savefig(shap_plot_path)
            plt.close()
            logger.info(f"Saved SHAP summary plot to {shap_plot_path}")
        except Exception as e:
            logger.error(f"Error creating SHAP plot: {str(e)}")
            logger.error(traceback.format_exc())
    
    return {
        "channel_importance": avg_importance,
        "plots": {"shap_plot": "shap_summary.png"} if shap_plot_path else {}
    }

def compute_integrated_gradients(model, data_loader, num_samples=5):
    """Compute feature attributions using Integrated Gradients"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Get samples to explain
    samples = []
    sample_labels = []
    sample_names = []
    idx = 0
    
    for images, labels in data_loader:
        for i in range(min(num_samples, len(images))):
            samples.append(images[i:i+1].clone())  # Clone to avoid views
            sample_labels.append(labels[i].item())
            sample_names.append(f"sample_{idx}")
            idx += 1
        if len(samples) >= num_samples:
            break
    
    # Compute attributions
    explanations = []
    heatmap_plot_path = None
    
    try:
        for i, (sample, label, name) in enumerate(zip(samples, sample_labels, sample_names)):
            sample = sample.to(device)
            target = label
            sample.requires_grad = True  # Enable gradients for this tensor
            
            # Get attributions - we need to detach and clone to avoid backward hook issues
            with torch.no_grad():
                attributions = ig.attribute(sample, target=target, return_convergence_delta=False)
                # Make a copy of the attributions to avoid in-place operations
                attributions_copy = attributions.detach().clone()
            
            # Extract channel importance safely (no in-place operations)
            with torch.no_grad():
                attr_sum = torch.sum(torch.abs(attributions_copy), dim=[2, 3])
                r_imp = float(attr_sum[0, 0].item())
                g_imp = float(attr_sum[0, 1].item())
                b_imp = float(attr_sum[0, 2].item())
            
            # Create visualization only for the first sample
            if i == 0:
                # Convert to numpy for visualization
                # We need to detach and make copies before moving to CPU to avoid view issues
                with torch.no_grad():
                    img_np = sample.detach().clone().cpu().squeeze(0).permute(1, 2, 0).numpy()
                    attr_np = attributions_copy.detach().clone().cpu().squeeze(0).permute(1, 2, 0).numpy()
                
                # Normalize attributions for visualization
                attr_np = visualization.normalize_image_attr(attr_np, 'absolute_value', 0)
                
                # Create visualization
                fig, _ = visualization.visualize_image_attr_multiple(
                    attr_np, img_np, 
                    ["original_image", "heat_map"],
                    ["Input", "Attribution Magnitude"],
                    show_colorbar=True,
                    use_pyplot=False
                )
                
                # Save plot
                heatmap_plot_path = os.path.join(PLOT_DIR, "attributions_heatmap.png")
                fig.savefig(heatmap_plot_path)
                plt.close(fig)
                logger.info(f"Saved attribution heatmap to {heatmap_plot_path}")
            
            # Add explanation entry
            explanations.append({
                "image": name,
                "r_importance": r_imp,
                "g_importance": g_imp,
                "b_importance": b_imp
            })
    except Exception as e:
        logger.error(f"Error computing Integrated Gradients: {str(e)}")
        logger.error(traceback.format_exc())
    
    return {
        "explanations": explanations,
        "plots": {"heatmap_plot": "attributions_heatmap.png"} if heatmap_plot_path else {}
    }

def explain_model(run_id):
    """Generate explanations for a model"""
    try:
        model = load_model(run_id)
        model.eval()
        
        # Load data
        data_loader, class_names = load_data()
        
        # Get feature importance using SHAP
        logger.info("Computing SHAP feature importance")
        shap_results = compute_feature_importance(model, data_loader)
        
        # Get attributions using Integrated Gradients
        logger.info("Computing Integrated Gradients attributions")
        ig_results = compute_integrated_gradients(model, data_loader)
        
        # Combine results
        results = {
            "run_id": run_id,
            "num_classes": len(class_names),
            "class_names": class_names,
            "explanations": ig_results["explanations"],
            "channel_importance": shap_results["channel_importance"],
            "plots": {**shap_results["plots"], **ig_results["plots"]}
        }
        
        # Log to MLflow
        try:
            with mlflow.start_run(run_id=run_id):
                for channel, value in shap_results["channel_importance"].items():
                    mlflow.log_metric(channel, value)
                
                # Log plots
                for plot_name, plot_file in results["plots"].items():
                    mlflow.log_artifact(os.path.join(PLOT_DIR, plot_file))
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {str(e)}")
        
        return results
    except Exception as e:
        logger.error(f"Explanation generation failed: {str(e)}")
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.route('/explain', methods=['POST'])
def explain_endpoint():
    """API endpoint for model explanations"""
    try:
        data = request.json
        run_id = data.get('run_id')
        
        if not run_id:
            # Try to get run_id from file first
            run_id = get_run_id_from_file()
            
            # If not in file, get latest run
            if not run_id:
                run_id = get_latest_run_id()
                
            if not run_id:
                return jsonify({"error": "No run_id provided and couldn't find latest run"}), 400
        
        logger.info(f"Generating explanations for run: {run_id}")
        results = explain_model(run_id)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in explain endpoint: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve plot images"""
    try:
        return send_file(os.path.join(PLOT_DIR, filename))
    except Exception as e:
        logger.error(f"Error serving plot {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 404

if __name__ == '__main__':
    # Create plot directory
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5005, debug=False)
