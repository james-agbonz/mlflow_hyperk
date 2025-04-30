import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from mlflow.tracking import MlflowClient
import numpy as np
import time
from PIL import Image
import io
import base64

# ----- PAGE CONFIG -----
st.set_page_config(layout="wide", page_title="ML Pipeline Dashboard")
st.title("🔍 ML Pipeline Dashboard")

# ----- MLFLOW CLIENT SETUP -----
@st.cache_resource
def get_mlflow_client():
    return MlflowClient(tracking_uri="http://mlflow:5000")

try:
    client = get_mlflow_client()
except Exception as e:
    st.error(f"Failed to connect to MLflow: {e}")
    client = None

# ----- UTILITY FUNCTIONS -----
def get_run_id_from_file():
    paths = ["/app/run_id.txt", "/app/artifacts/run_id.txt"]
    for path in paths:
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            continue
    return None

def get_all_runs():
    try:
        experiments = client.search_experiments()
        runs = []
        for exp in experiments:
            exp_runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
            runs.extend([(run.info.run_id, f"{exp.name}: {run.info.run_id}") for run in exp_runs])
        return runs
    except Exception as e:
        st.error(f"Failed to get runs: {e}")
        return []

def fetch_json(endpoint, run_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Fetching data from {endpoint} (attempt {attempt+1}/{max_retries})..."):
                response = requests.post(f"http://{endpoint}", json={"run_id": run_id}, timeout=90)
                if response.status_code == 200:
                    return response.json()
                st.warning(f"Failed to fetch from {endpoint} — Status: {response.status_code}. Retrying...")
                time.sleep(2)  # Wait before retrying
        except Exception as e:
            st.warning(f"Request error: {e}. Retrying...")
            time.sleep(2)  # Wait before retrying
    st.error(f"Failed to fetch from {endpoint} after {max_retries} attempts")
    return None

def fetch_image(url):
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        return None
    except Exception as e:
        st.error(f"Failed to fetch image: {e}")
        return None

def load_artifact(run_id, filename):
    try:
        uri = client.get_run(run_id).info.artifact_uri
        if uri.startswith("file:"):
            path = uri.replace("file:", "") + f"/{filename}"
            if os.path.exists(path):
                return path
        return None
    except Exception as e:
        st.error(f"Failed to load artifact: {e}")
        return None

def display_metrics_over_time(run):
    for metric in run.data.metrics:
        history = client.get_metric_history(run.info.run_id, metric)
        if history:
            steps, values = zip(*[(h.step, h.value) for h in history])
            fig, ax = plt.subplots()
            ax.plot(steps, values, marker="o")
            ax.set_title(metric)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

def display_dataset_stats(run_id):
    try:
        # Try to load dataset stats if they exist
        stats_path = load_artifact(run_id, "dataset_stats.json")
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            st.subheader("Dataset Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Samples", stats.get("total_samples", "N/A"))
                st.metric("Number of Classes", stats.get("num_classes", "N/A"))
            
            with col2:
                st.metric("Training Samples", stats.get("train_samples", "N/A"))
                st.metric("Validation Samples", stats.get("val_samples", "N/A"))
            
            # Class distribution if available
            if "class_distribution" in stats:
                st.subheader("Class Distribution")
                class_df = pd.DataFrame(list(stats["class_distribution"].items()), 
                                        columns=["Class", "Count"])
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=class_df, x="Class", y="Count", ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not load dataset statistics: {e}")

# ----- SIDEBAR -----
st.sidebar.header("🔄 Run Selection")
file_run_id = get_run_id_from_file()
if file_run_id and st.sidebar.checkbox("Use Latest Training Run ID", value=True):
    selected_run = file_run_id
else:
    run_options = get_all_runs()
    selected_run = st.sidebar.selectbox("Select a run", options=[r[0] for r in run_options],
                                      format_func=lambda x: dict(run_options).get(x, x)) if run_options else None
if selected_run:
    st.sidebar.info(f"Current run ID: {selected_run}")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Display Options")
show_raw_data = st.sidebar.checkbox("Show Raw Data Tables", value=False)
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)

if auto_refresh:
    st.sidebar.write("Auto-refreshing every 30 seconds")
    st.rerun()
    time.sleep(30)

# ----- TABS -----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Metrics", 
    "🧩 Artifacts", 
    "📊 Evaluation", 
    "🔍 Explanations",
    "📷 Visualizations"
])

# ----- TAB 1: METRICS -----
with tab1:
    st.header("📈 Model Metrics")
    if selected_run and client:
        try:
            run = client.get_run(selected_run)
            
            # Display parameters and metrics in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Parameters")
                params_df = pd.DataFrame(run.data.params.items(), columns=["Parameter", "Value"])
                st.dataframe(params_df, use_container_width=True)
            
            with col2:
                st.subheader("Metrics")
                metrics_df = pd.DataFrame(run.data.metrics.items(), columns=["Metric", "Value"])
                st.dataframe(metrics_df, use_container_width=True)
            
            # Display metrics over time
            st.subheader("Metrics Over Time")
            display_metrics_over_time(run)
            
            # Display dataset statistics if available
            display_dataset_stats(selected_run)
            
        except Exception as e:
            st.error(f"Error loading run data: {e}")
    else:
        st.warning("No run selected or MLflow client unavailable")

# ----- TAB 2: ARTIFACTS -----
with tab2:
    st.header("🧩 Artifacts")
    if selected_run and client:
        try:
            artifacts = client.list_artifacts(selected_run)
            if artifacts:
                df = pd.DataFrame([{
                    "Name": a.path, 
                    "Size": f"{a.file_size/1024:.2f} KB" if a.file_size else "N/A",
                    "Type": "Directory" if a.is_dir else a.path.split('.')[-1].upper()
                } for a in artifacts])
                
                st.dataframe(df, use_container_width=True)
                
                # Filter artifacts by type
                st.subheader("View Artifact")
                artifact_types = {
                    "Images": [a.path for a in artifacts if a.path.lower().endswith((".png", ".jpg", ".jpeg"))],
                    "Text": [a.path for a in artifacts if a.path.lower().endswith((".txt", ".log", ".md"))],
                    "Data": [a.path for a in artifacts if a.path.lower().endswith((".csv", ".json"))]
                }
                
                artifact_type = st.radio("Artifact Type", list(artifact_types.keys()))
                
                if artifact_types[artifact_type]:
                    selected_artifact = st.selectbox("Select artifact", options=artifact_types[artifact_type])
                    artifact_path = load_artifact(selected_run, selected_artifact)
                    
                    if artifact_path:
                        if artifact_type == "Images":
                            st.image(artifact_path, caption=selected_artifact)
                        elif artifact_type == "Text":
                            with open(artifact_path, 'r') as f:
                                st.code(f.read())
                        elif artifact_type == "Data":
                            if selected_artifact.endswith(".csv"):
                                df = pd.read_csv(artifact_path)
                                st.dataframe(df, use_container_width=True)
                            elif selected_artifact.endswith(".json"):
                                with open(artifact_path, 'r') as f:
                                    data = json.load(f)
                                st.json(data)
                else:
                    st.info(f"No {artifact_type.lower()} artifacts found")
            else:
                st.info("No artifacts found for this run")
        except Exception as e:
            st.error(f"Error loading artifacts: {e}")
    else:
        st.warning("No run selected or MLflow client unavailable")

# ----- TAB 3: EVALUATION -----
with tab3:
    st.header("📊 Evaluation Results")
    if selected_run:
        data = fetch_json("evaluator:5006/evaluate", selected_run)
        if data:
            # Filter out any non-numeric values for visualization
            numeric_data = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            if numeric_data:
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("Evaluation Metrics")
                    df = pd.DataFrame({"Metric": list(numeric_data.keys()), "Value": list(numeric_data.values())})
                    st.dataframe(df, use_container_width=True)
                
                with col2:
                    st.subheader("Metrics Visualization")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    metrics_df = df.copy()
                    sns.barplot(data=metrics_df, x="Metric", y="Value", ax=ax)
                    ax.set_ylim(0, 1)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
            
            # Show confusion matrix if available
            if "confusion_matrix" in data:
                st.subheader("Confusion Matrix")
                cm = np.array(data["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            
            # Show all data if requested
            if show_raw_data:
                st.subheader("All Evaluation Data")
                st.json(data)
        else:
            st.warning("No evaluation data available. Run the evaluator service first.")
    else:
        st.warning("No run selected")

# ----- TAB 4: EXPLANATIONS -----
with tab4:
    st.header("🧠 Model Explanations")
    if selected_run:
        explain_data = fetch_json("explainer:5005/explain", selected_run)
        
        if explain_data:
            st.subheader("Feature Importance")
            
            # Check if we have a list of dictionaries or a single dictionary
            if isinstance(explain_data, list):
                df = pd.DataFrame(explain_data)
            elif isinstance(explain_data, dict):
                if "explanations" in explain_data:
                    df = pd.DataFrame(explain_data["explanations"])
                elif "error" in explain_data:
                    st.error(f"Error from explainer: {explain_data['error']}")
                    st.code(explain_data.get("traceback", "No traceback available"))
                    df = pd.DataFrame()
                else:
                    df = pd.DataFrame([explain_data])
            else:
                st.error("Unexpected data format from explainer")
                df = pd.DataFrame()
            
            if not df.empty:
                # Extract importance columns and display feature importance
                importance_cols = [col for col in df.columns if "importance" in col]
                if importance_cols:
                    # Create feature importance plot
                    st.subheader("Channel Importance")
                    
                    # Calculate average importance for each feature
                    feature_importance = {}
                    for col in importance_cols:
                        feature = col.split("_")[0]
                        importance_type = "_".join(col.split("_")[1:])
                        if feature not in feature_importance:
                            feature_importance[feature] = {}
                        feature_importance[feature][importance_type] = df[col].mean()
                    
                    # Convert to dataframe for plotting
                    fi_df = pd.DataFrame([
                        {"Feature": feature, "Type": imp_type, "Importance": value}
                        for feature, imp_dict in feature_importance.items()
                        for imp_type, value in imp_dict.items()
                    ])
                    
                    if not fi_df.empty:
                        # Plot feature importance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=fi_df, x="Feature", y="Importance", hue="Type", ax=ax)
                        ax.set_title("Average Feature Importance")
                        ax.set_xlabel("Feature")
                        ax.set_ylabel("Importance")
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                
                # Check if we have sample-level data
                if 'image' in df.columns:
                    st.subheader("Sample-Level Explanations")
                    
                    # Get a list of unique images
                    images = df['image'].unique()
                    selected_image = st.selectbox("Select image", options=images)
                    
                    # Filter data for the selected image
                    image_data = df[df['image'] == selected_image]
                    
                    # Create a bar plot of feature importance for the selected image
                    if not image_data.empty:
                        importance_cols = [col for col in image_data.columns if "importance" in col]
                        if importance_cols:
                            # Reshape data for plotting
                            plot_data = []
                            for col in importance_cols:
                                feature, imp_type = col.split("_", 1)
                                plot_data.append({
                                    "Feature": feature,
                                    "Importance Type": imp_type,
                                    "Value": image_data[col].iloc[0]
                                })
                            
                            plot_df = pd.DataFrame(plot_data)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(data=plot_df, x="Feature", y="Value", hue="Importance Type", ax=ax)
                            ax.set_title(f"Feature Importance for {selected_image}")
                            ax.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                
                # Display raw data if requested
                if show_raw_data:
                    st.subheader("Raw Explanation Data")
                    st.dataframe(df, use_container_width=True)
            
            # Check for explanation plots
            plot_info = None
            
            # Look for plot information in the data
            if isinstance(explain_data, list) and explain_data:
                if "plots" in explain_data[0]:
                    plot_info = explain_data[0]["plots"]
            elif isinstance(explain_data, dict):
                if "plots" in explain_data:
                    plot_info = explain_data["plots"]
            
            if plot_info:
                st.subheader("Explanation Visualizations")
                
                col1, col2 = st.columns(2)
                
                # Display SHAP plot if available
                if "shap_plot" in plot_info:
                    with col1:
                        st.subheader("SHAP Summary Plot")
                        shap_img = fetch_image(f"http://explainer:5005/plots/{plot_info['shap_plot']}")
                        if shap_img:
                            st.image(shap_img, use_column_width=True)
                        else:
                            st.warning("Failed to load SHAP plot")
                
                # Display heatmap if available
                if "heatmap_plot" in plot_info:
                    with col2:
                        st.subheader("Integrated Gradients Heatmap")
                        heatmap_img = fetch_image(f"http://explainer:5005/plots/{plot_info['heatmap_plot']}")
                        if heatmap_img:
                            st.image(heatmap_img, use_column_width=True)
                        else:
                            st.warning("Failed to load heatmap plot")
            
        else:
            st.warning("No explanation data available. Run the explainer service first.")
    else:
        st.warning("No run selected")
