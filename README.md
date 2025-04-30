🏥 Medical Image Pipeline Dashboard
This project is a Streamlit dashboard that connects to an MLflow Tracking Server, fetches experiment data, displays it interactively, and allows users to select a trained model for download and further use.

It is part of a medical imaging pipeline consisting of services like:

Dataloader (loading medical images)

Augmenter (e.g., CutMix augmentations)

Trainer (training models and tracking experiments with MLflow)

Evaluator (evaluating and logging metrics)

Explainer (building model explanations)

This dashboard focuses on the model management and selection side of the pipeline.
____________________________________________________________________________________________________

✨ Features
____________________________________________________________________________________________________
Connects to a remote or local MLflow Tracking Server

Lists all available experiments

Shows available runs for each experiment

Displays:

Parameters (e.g., optimizer, augmentation used)

Metrics (e.g., accuracy, loss)

Run Info (run ID, status)

Visualizes metrics with Seaborn and Matplotlib plots

Allows users to select a run and download its model artifacts

Saves the selected run_id locally (run_id.txt) for integration into downstream services

Completely browser-based via Streamlit

🛠️ Project Structure
MLpipeline/
├── trainer/
│   ├── train.py
│   └── Dockerfile
├── augmenter/
│   └── cutmix.py
├── evaluator/
│   ├── app.py  
│   ├── evaluate.py  
│   └── Dockerfile
├── explainer/
│   ├── app.py
│   ├── explain.py  
│   └── Dockerfile
├── dashboard/
│   ├── dashboard.py
│   └── Dockerfile
├── data/
│   ├── train/
│   └── test/
├── requirements.txt      
└── docker-compose.yml


Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Make sure your MLflow server is running and accessible (local or cloud).

Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Follow the UI:

Enter the MLflow tracking URI (e.g., http://127.0.0.1:5000 or your remote endpoint)

Explore experiments, runs, metrics

Download models!

🧩 Requirements
All needed packages are listed in requirements.txt:

txt
Copy
Edit
streamlit
mlflow
pandas
numpy
matplotlib
seaborn
requests
scikit-learn
torch  # (if using PyTorch models)
Optional: if you later add explanation visualizations

txt
Copy
Edit
captum
📦 Artifact Downloads
When you select a model and click Download Model, the artifacts are downloaded from MLflow and stored inside the /artifacts/ directory.
You can then use these models directly in downstream applications (e.g., inference services, evaluators, explainers).

📚 Background Context
This dashboard is a component of a larger medical image processing pipeline aimed at:

Enhancing data augmentation for medical datasets

Training efficient deep learning models (with minimal augmentation)

Tracking experiments using MLflow

Evaluating and explaining model predictions

The pipeline prioritizes transparency, traceability, and ease of model selection for deployment.

🚀 Future Enhancements
Add Captum explanations directly into the dashboard

Enable batch download of all top-performing models

Integrate evaluation visualizations (e.g., confusion matrices, ROC curves)

Build microservices from the selected models dynamically

🤝 Contributions
Open to improvements, ideas, and collaboration!
Feel free to fork, improve, or propose new features.

📜 License
This project is currently private for academic, research, and development purposes.
License can be specified later depending on open-source or organizational preferences.

🩺 Built for smarter medical AI pipelines.
Would you also like me to quickly show you what the artifacts folder will look like after downloading a model (with example filenames)?
👉 (It'll help if you later connect this dashboard to your evaluator or explainer services.) 🚀




*******************************************************
Stage | Service | What it Should Do
1 | Data Loader | Load dataset
2 | Augmenter | Apply augmentation (e.g., CutMix)
3 | Trainer | Train a model, log model to MLflow
4 | Evaluator | Evaluate the trained model: calculate metrics (accuracy, precision, etc), save CSV of top-5 misclassified images, generate confusion matrix image
5 | Explainer | Take the trained model + data and:  - Generate SHAP values  - Generate Integrated Gradients heatmaps  - Save both as MLflow artifacts

*********************************************
dashoard
***********************************************
📈 Metrics

What it shows: Model performance metrics and training parameters
Content:

Parameters used to train the model (displayed in a table)
Metrics like accuracy, loss, precision, recall (displayed in a table)
Charts showing how metrics changed over time during training
Dataset statistics showing sample counts and class distribution



🧩 Artifacts

What it shows: Files saved during the model training process
Content:

List of all artifacts with name, size, and type
Viewer for different artifact types:

Images (shown as images)
Text files (shown as code)
Data files (CSV shown as tables, JSON shown formatted)





📊 Evaluation

What it shows: Detailed model evaluation results
Content:

Evaluation metrics in a table format
Bar chart visualizing the metrics
Confusion matrix heatmap showing prediction accuracy across classes
Raw evaluation data (optional)



🔍 Explanations

What it shows: Why the model makes specific predictions
Content:

Feature importance charts showing which channels (RGB) influence predictions most
Sample-level explanations for individual images
SHAP summary plot showing feature impact across all samples
Integrated Gradients heatmaps showing which parts of images influence predictions
Raw explanation data (optional)



📷 Visualizations

What it shows: Visual representations of the model's behavior
Content:

This tab wasn't fully implemented in the code we saw, but likely would contain:
Visualizations of model activations
Sample predictions with visual overlays
t-SNE or other dimensionality reduction plots
Class activation maps or other model interpretation visualizations



Each tab works with the selected model run (specified by run ID) and automatically pulls the relevant data from the appropriate services (evaluator, explainer, etc.).



*************************************************
DEPPLOYMENT
*************************************************

# ML Pipeline with CutMix

This repository contains an end-to-end machine learning pipeline with CutMix augmentation, model training, evaluation, explanation, and visualization through a dashboard.

## Project Structure

```
MLpipeline/
├── trainer/
│   ├── train.py
│   └── Dockerfile
├── augmenter/
│   └── cutmix.py
├── evaluator/
│   ├── app.py  
│   ├── evaluate.py  
│   └── Dockerfile
├── explainer/
│   ├── app.py
│   ├── explain.py  
│   └── Dockerfile
├── dashboard/
│   ├── dashboard.py
│   └── Dockerfile
├── data/
│   ├── train/
│   └── test/
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ml-pipeline-ci.yml
k8s/
├── deployments/
│   ├── evaluator-deployment.yaml
│   ├── explainer-deployment.yaml
│   └── dashboard-deployment.yaml
├── services/
│   ├── evaluator-service.yaml
│   ├── explainer-service.yaml
│   └── dashboard-service.yaml
└── trainer-job.yaml

```

## Services

- **MongoDB**: Database for experiment tracking and storage
- **MLflow**: Experiment tracking and model registry
- **Trainer**: Trains models using CutMix augmentation
- **Evaluator**: Evaluates model performance
- **Explainer**: Provides model explanations using SHAP and Integrated Gradients
- **Dashboard**: Streamlit dashboard for visualization and monitoring

## CI/CD Pipeline

The project includes a GitHub Actions workflow for continuous integration and deployment:

### Workflow Steps

1. **Lint**: Check code formatting and style
2. **Test**: Run unit tests with coverage reporting
3. **Build Images**: Build Docker images for all services
4. **Docker Compose Test**: Verify Docker Compose configuration and service health
5. **Deploy to Development**: Deploy to development environment (on `dev` branch)
6. **Deploy to Production**: Deploy to production environment (on `main` or `master` branch)

### Deployment Environments

- **Development**: Server-based deployment using Docker Compose
- **Production**: Kubernetes-based deployment to AWS EKS

## Getting Started

### Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MLpipeline
   ```

2. Create data directories:
   ```bash
   mkdir -p data/train data/test mlruns artifacts
   ```

3. Run the pipeline locally:
   ```bash
   docker-compose up -d
   ```

### Kubernetes Deployment

1. Create Kubernetes resources:
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/secrets.yaml
   kubectl apply -f k8s/mongodb.yaml
   kubectl apply -f k8s/mlflow.yaml
   kubectl apply -f k8s/trainer.yaml
   kubectl apply -f k8s/evaluator.yaml
   kubectl apply -f k8s/explainer.yaml
   kubectl apply -f k8s/dashboard.yaml
   ```

## GitHub Actions

The workflow can be triggered:
- Automatically on push to `main`, `master`, or `dev` branches
- Automatically on pull requests to these branches
- Manually via workflow dispatch with environment selection

## Monitoring

Access the different services:

- **MLflow**: http://mlflow.example.com (or http://localhost:5000 locally)
- **Dashboard**: http://dashboard.example.com (or http://localhost:8501 locally)
- **Explainer API**: http://explainer:5005/explain
- **Evaluator API**: http://evaluator:5006/evaluate

## Pipeline Workflow

1. The trainer service loads data, applies CutMix augmentation, trains the model, and logs metrics to MLflow
2. The evaluator service measures model performance on test data
3. The explainer service generates feature importance and attributions
4. The dashboard provides a unified view of model performance, explanations, and artifacts

## Environment Variables

The pipeline uses the following environment variables:

- `MLFLOW_TRACKING_URI`: URL of the MLflow tracking server
- `PYTHONUNBUFFERED`: Ensures Python output is not buffered
