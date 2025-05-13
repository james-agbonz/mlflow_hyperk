"""
Dashboard module for the XAI pipeline.
Provides visualization of model performance and explanations.
"""

import os
import sys
import argparse
import json
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
from datetime import datetime
import glob

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run XAI Dashboard')
    parser.add_argument('--evaluation_dir', type=str, default='/app/results/evaluation',
                       help='Directory containing evaluation results')
    parser.add_argument('--explanations_dir', type=str, default='/app/results/explanations',
                       help='Directory containing explanation visualizations')
    parser.add_argument('--mlflow_uri', type=str, default=os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000'),
                       help='MLflow tracking URI')
    
    return parser.parse_args()


def load_evaluation_results(eval_dir):
    """Load evaluation results from JSON"""
    json_path = os.path.join(eval_dir, 'evaluation_results.json')
    if not os.path.exists(json_path):
        st.error(f"Evaluation results not found at {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    return results


def load_explanation_results(exp_dir):
    """Load explanation results from JSON"""
    json_path = os.path.join(exp_dir, 'explanation_results.json')
    if not os.path.exists(json_path):
        st.error(f"Explanation results not found at {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    return results


def find_explanation_images(exp_dir):
    """Find all explanation images in the directory"""
    image_paths = glob.glob(os.path.join(exp_dir, '*_explanations.png'))
    return image_paths


def run_pipeline_with_shared_run_id():
    """Run the entire pipeline with a shared MLflow run ID"""
    st.write("Starting XAI pipeline with shared MLflow run ID...")
    
    try:
        # Step 1: Start the trainer to create a new run and capture the run ID
        st.write("Step 1: Training model...")
        trainer_cmd = ["docker-compose", "run", "--rm", "trainer"]
        trainer_process = subprocess.run(trainer_cmd, capture_output=True, text=True)
        
        if trainer_process.returncode != 0:
            st.error("Training failed!")
            st.code(trainer_process.stderr)
            return
        
        st.success("Training completed!")
        
        # Step 2: Read the run ID from the file
        run_id_path = "./models/latest_run_id.txt"
        if not os.path.exists(run_id_path):
            st.error("Run ID file not found! Cannot continue with shared run.")
            return
            
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
            
        st.info(f"Using MLflow run ID: {run_id}")
        
        # Step 3: Run evaluator with the same run ID
        st.write("Step 2: Evaluating model...")
        eval_cmd = ["docker-compose", "run", "--rm", "-e", f"MLFLOW_RUN_ID={run_id}", "evaluator"]
        eval_process = subprocess.run(eval_cmd, capture_output=True, text=True)
        
        if eval_process.returncode != 0:
            st.error("Evaluation failed!")
            st.code(eval_process.stderr)
            return
            
        st.success("Evaluation completed!")
        
        # Step 4: Run explainer with the same run ID
        st.write("Step 3: Generating explanations...")
        explain_cmd = ["docker-compose", "run", "--rm", "-e", f"MLFLOW_RUN_ID={run_id}", "explainer"]
        explain_process = subprocess.run(explain_cmd, capture_output=True, text=True)
        
        if explain_process.returncode != 0:
            st.error("Explanation generation failed!")
            st.code(explain_process.stderr)
            return
            
        st.success("Explanations generated!")
        
        # Final success message
        st.success(f"Pipeline completed successfully with shared run ID: {run_id}")
        st.info("Please refresh the page to see the results in the MLflow section.")
        
    except Exception as e:
        st.error(f"Error running pipeline: {str(e)}")
def run_dashboard(eval_dir, exp_dir, mlflow_uri):
    """Run the Streamlit dashboard"""
    st.set_page_config(
        page_title="XAI Pipeline Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title("Explainable AI (XAI) Pipeline Dashboard")
    st.markdown("### Model Performance and Explanations with CutMix Augmentation")
    
    # Sidebar - MLflow connection
    st.sidebar.header("MLflow Connection")
    mlflow_uri_input = st.sidebar.text_input("MLflow URI", value=mlflow_uri)
    
    if st.sidebar.button("Connect to MLflow"):
        mlflow_uri = mlflow_uri_input
    
    # Run Pipeline button
    st.sidebar.header("Pipeline Control")
    if st.sidebar.button("Run Complete Pipeline"):
        with st.spinner("Running complete pipeline (training, evaluation, explanations)..."):
            run_pipeline_with_shared_run_id()
    
    # Get MLflow experiments
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Get all experiments
        experiments = mlflow.search_experiments()
        
        if experiments:
            experiment_names = [exp.name for exp in experiments]
            selected_experiment = st.sidebar.selectbox("Select Experiment", experiment_names)
            
            # Get the selected experiment
            experiment = next((exp for exp in experiments if exp.name == selected_experiment), None)
            
            if experiment:
                # Get runs for the selected experiment
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                
                if not runs.empty:
                    # Display runs in the sidebar
                    run_options = []
                    for _, run in runs.iterrows():
                        # Handle different timestamp formats safely
                        try:
                            if isinstance(run.start_time, pd.Timestamp):
                                timestamp = run.start_time.to_pydatetime()
                            else:
                                timestamp = datetime.fromtimestamp(run.start_time/1000)
                            run_name = f"{run.run_id[:8]}... ({timestamp.strftime('%Y-%m-%d %H:%M')})"
                        except Exception as e:
                            run_name = f"{run.run_id[:8]}..."
                        run_options.append((run_name, run.run_id))
                    
                    selected_run_name = st.sidebar.selectbox(
                        "Select Run", 
                        [name for name, _ in run_options]
                    )
                    
                    # Get the selected run ID
                    selected_run_id = next((run_id for name, run_id in run_options if name == selected_run_name), None)
    except Exception as e:
        st.sidebar.error(f"Error connecting to MLflow: {str(e)}")
    
    # Create tabs
    tabs = st.tabs(["Model Training", "Model Performance", "Confusion Matrix", "Class Metrics", "Explanations"])
    
    # Model Training Tab
    with tabs[0]:
        st.header("Model Training Metrics")
        
        if 'experiments' in locals() and 'runs' in locals() and 'selected_run_id' in locals() and selected_run_id:
            # Get the selected run
            run = runs[runs.run_id == selected_run_id].iloc[0]
            
            # Extract metrics and parameters
            metrics = {col[8:]: run[col] for col in run.keys() if col.startswith('metrics.')}
            params = {col[7:]: run[col] for col in run.keys() if col.startswith('params.')}
            
            # Create columns for metrics and parameters
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Parameters")
                params_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
                st.dataframe(params_df, use_container_width=True)
            
            with col2:
                st.subheader("Training Metrics")
                # Filter for the final metrics
                final_metrics = {}
                for key, value in metrics.items():
                    if not key.endswith('_step'):
                        final_metrics[key] = value
                
                metrics_df = pd.DataFrame(list(final_metrics.items()), columns=["Metric", "Value"])
                st.dataframe(metrics_df, use_container_width=True)
            
            # Plot metrics over time if available
            epoch_metrics = {}
            for key in metrics.keys():
                if key in ['train_loss', 'test_loss', 'accuracy']:
                    epoch_metrics[key] = []
            
            if epoch_metrics:
                st.subheader("Metrics Over Time")
                
                # Get run data with metrics history
                client = mlflow.tracking.MlflowClient()
                run_data = client.get_run(selected_run_id)
                
                # Extract metrics history
                for metric_key in epoch_metrics.keys():
                    metric_history = client.get_metric_history(selected_run_id, metric_key)
                    if metric_history:
                        epochs = [m.step for m in metric_history]
                        values = [m.value for m in metric_history]
                        epoch_metrics[metric_key] = (epochs, values)
                
                if any(len(v[0]) > 0 for v in epoch_metrics.values()):
                    fig, axes = plt.subplots(len(epoch_metrics), 1, figsize=(10, 4 * len(epoch_metrics)))
                    
                    if len(epoch_metrics) == 1:
                        axes = [axes]
                    
                    for i, (metric_name, (epochs, values)) in enumerate(epoch_metrics.items()):
                        if len(epochs) > 0:
                            axes[i].plot(epochs, values, marker='o')
                            axes[i].set_title(f"{metric_name} Over Time")
                            axes[i].set_xlabel("Epoch")
                            axes[i].set_ylabel(metric_name)
                            axes[i].grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.info("Select an experiment and run from the sidebar to view training metrics.")
# Model Performance Tab
    with tabs[1]:
        st.header("Model Performance Overview")
        
        # Load evaluation results
        eval_results = load_evaluation_results(eval_dir)
        
        if eval_results:
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{eval_results['metrics']['accuracy']:.2f}%")
                st.metric("Average Loss", f"{eval_results['metrics']['avg_loss']:.4f}")
            
            with col2:
                # Extract model info
                model_name = eval_results['model']['model_name']
                model_path = eval_results['model']['model_path']
                
                st.subheader("Model Information")
                st.write(f"**Name:** {model_name}")
                st.write(f"**Path:** {model_path}")
                st.write(f"**Parameters:** {eval_results['model']['num_parameters']:,}")
            
            with col3:
                # Extract environment and timing info
                device = eval_results['environment']['device']
                eval_time = eval_results['performance']['evaluation_time_seconds']
                inference_time = eval_results['performance']['inference_time_seconds_per_image']
                
                st.subheader("Performance Metrics")
                st.write(f"**Device:** {device}")
                st.write(f"**Evaluation Time:** {eval_time:.2f} seconds")
                st.write(f"**Inference Time:** {inference_time * 1000:.2f} ms/image")
                st.write(f"**Evaluation Date:** {eval_results['evaluation_datetime']}")
            
            # Dataset information
            st.subheader("Dataset Information")
            st.write(f"**Path:** {eval_results['dataset']['path']}")
            st.write(f"**Split:** {eval_results['dataset']['split']}")
            st.write(f"**Size:** {eval_results['dataset']['size']} images")
            st.write(f"**Classes:** {', '.join(eval_results['dataset']['classes'])}")
            
            # ROC Curve for binary classification
            if eval_results['metrics']['roc_auc'] is not None:
                st.subheader("ROC Curve")
                
                # Load from file
                roc_img_path = os.path.join(eval_dir, 'roc_curve.png')
                if os.path.exists(roc_img_path):
                    roc_img = Image.open(roc_img_path)
                    st.image(roc_img, caption=f"ROC Curve (AUC: {eval_results['metrics']['roc_auc']:.4f})")
                else:
                    st.warning("ROC Curve visualization not found.")
            
            # PR Curves
            st.subheader("Precision-Recall Curves")
            
            # Load from file
            pr_img_path = os.path.join(eval_dir, 'precision_recall_curves.png')
            if os.path.exists(pr_img_path):
                pr_img = Image.open(pr_img_path)
                st.image(pr_img, caption="Precision-Recall Curves")
            else:
                st.warning("Precision-Recall Curves visualization not found.")
        else:
            st.info("No evaluation results found. Run the complete pipeline from the sidebar.")
    
    # Confusion Matrix Tab
    with tabs[2]:
        st.header("Confusion Matrix")
        
        # Load from file
        cm_img_path = os.path.join(eval_dir, 'confusion_matrix.png')
        if os.path.exists(cm_img_path):
            cm_img = Image.open(cm_img_path)
            st.image(cm_img, caption="Confusion Matrix")
        elif 'eval_results' in locals() and eval_results:
            # Create confusion matrix from data
            cm_data = np.array(eval_results['metrics']['confusion_matrix'])
            classes = eval_results['dataset']['classes']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                       xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            st.pyplot(fig)
        else:
            st.info("No confusion matrix found. Run the complete pipeline from the sidebar.")
    
    # Class Metrics Tab
    with tabs[3]:
        st.header("Class-wise Metrics")
        
        if not 'eval_results' in locals() or eval_results is None:
            eval_results = load_evaluation_results(eval_dir)
        
        if eval_results and 'metrics' in eval_results and 'classification_report' in eval_results['metrics']:
            # Create a DataFrame from the classification report
            cr = eval_results['metrics']['classification_report']
            
            # Remove averages from the report
            metrics_df = {k: v for k, v in cr.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
            
            # Convert to DataFrame
            df = pd.DataFrame(metrics_df).T
            df = df.reset_index().rename(columns={'index': 'Class'})
            
            # Display metrics
            st.dataframe(df)
            
            # Plot metrics
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            metrics = ['precision', 'recall', 'f1-score']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, metric in enumerate(metrics):
                axes[i].bar(df['Class'], df[metric], color=colors[i])
                axes[i].set_title(f'Class-wise {metric.capitalize()}')
                axes[i].set_ylim(0, 1)
                for j, v in enumerate(df[metric]):
                    axes[i].text(j, v + 0.02, f'{v:.2f}', ha='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display incorrect predictions if available
            if 'incorrect_predictions' in eval_results and eval_results['incorrect_predictions']:
                st.subheader("Sample Incorrect Predictions")
                
                incorrect_df = pd.DataFrame(eval_results['incorrect_predictions'])
                st.dataframe(incorrect_df)
        else:
            st.info("No class metrics found. Run the complete pipeline from the sidebar.")
# Explanations Tab
    with tabs[4]:
        st.header("Model Explanations")
        
        # Load explanation results directly from file
        explanation_results = load_explanation_results(exp_dir)
        explanation_images = find_explanation_images(exp_dir)
        
        if not explanation_images:
            st.warning("No explanation visualizations found. Run the complete pipeline from the sidebar.")
        else:
            # Allow filtering by class
            if 'eval_results' in locals() and eval_results:
                classes = eval_results['dataset']['classes']
            else:
                classes = ["GOOD", "BAD"]  # Default classes
                
            selected_class = st.selectbox("Filter by class:", ["All"] + classes)
            
            # Allow sorting by confidence
            sort_by = st.selectbox("Sort by:", ["Confidence (High to Low)", "Confidence (Low to High)", "Filename"])
            
            # Filter and sort explanations
            if explanation_results:
                explanations_df = pd.DataFrame(explanation_results)
                
                # Filter by class if selected
                if selected_class != "All":
                    explanations_df = explanations_df[
                        (explanations_df['true_label'] == selected_class) | 
                        (explanations_df['predicted_label'] == selected_class)
                    ]
                
                # Sort explanations
                if sort_by == "Confidence (High to Low)":
                    explanations_df = explanations_df.sort_values('confidence', ascending=False)
                elif sort_by == "Confidence (Low to High)":
                    explanations_df = explanations_df.sort_values('confidence', ascending=True)
                else:  # Filename
                    explanations_df = explanations_df.sort_values('image_path')
                
                # Display explanations
                for i, row in explanations_df.iterrows():
                    img_name = os.path.splitext(os.path.basename(row['image_path']))[0]
                    explanation_path = os.path.join(exp_dir, f"{img_name}_explanations.png")
                    
                    if os.path.exists(explanation_path):
                        st.subheader(f"Image: {img_name}")
                        
                        # Show metadata about the prediction
                        col1, col2, col3 = st.columns(3)
                        col1.metric("True Label", row['true_label'])
                        col2.metric("Predicted Label", row['predicted_label'])
                        col3.metric("Confidence", f"{row['confidence']:.4f}")
                        
                        # Display the explanation image
                        exp_img = Image.open(explanation_path)
                        st.image(exp_img, caption=f"Explanations for {img_name}")
                        
                        st.markdown("---")
            else:
                # Just show all images if no metadata
                for img_path in explanation_images:
                    img_name = os.path.splitext(os.path.basename(img_path))[0].replace("_explanations", "")
                    st.subheader(f"Image: {img_name}")
                    exp_img = Image.open(img_path)
                    st.image(exp_img, caption=f"Explanations for {img_name}")
                    st.markdown("---")
            
            # Display class summaries if available
            st.subheader("Class Activation Map Summaries")
            
            # Check for summary files
            for class_name in classes:
                summary_path = os.path.join(exp_dir, f"summary_{class_name}.png")
                if os.path.exists(summary_path):
                    st.write(f"### Summary for class: {class_name}")
                    summary_img = Image.open(summary_path)
                    st.image(summary_img, caption=f"Class Activation Maps for {class_name}")
    
    # Footer
    st.markdown("---")
    st.markdown("**XAI Pipeline Dashboard** | Powered by MLflow, CutMix augmentation, and Captum for explainability")


if __name__ == "__main__":
    # Get command line arguments
    args = get_args()
    
    # Run the dashboard
    run_dashboard(
        args.evaluation_dir,
        args.explanations_dir,
        args.mlflow_uri
    )
