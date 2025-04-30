from flask import Flask, jsonify, request
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluate import evaluate_model, get_latest_run_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/artifacts/evaluation_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def get_run_id_from_file():
    try:
        paths = ["/app/run_id.txt", "/app/artifacts/run_id.txt"]
        for path in paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    run_id = f.read().strip()
                    if run_id:
                        return run_id
        return None
    except Exception as e:
        logger.error(f"Error reading run ID from file: {str(e)}")
        return None

@app.route("/evaluate", methods=["GET", "POST"])
def evaluate_endpoint():
    """Handle both GET and POST requests for model evaluation"""
    # Get run_id either from POST body or GET query parameter
    if request.method == "POST" and request.json:
        run_id = request.json.get("run_id")
    else:
        run_id = request.args.get("run_id")
    
    # If no run_id provided, try reading from file or getting latest
    if not run_id:
        run_id = get_run_id_from_file()
        
    if not run_id:
        # Last resort: get latest run from MLflow
        run_id = get_latest_run_id()
        
    if not run_id:
        logger.error("No run ID found")
        return jsonify({"error": "No runs found"}), 404

    try:
        logger.info(f"Evaluating model for run ID: {run_id}")
        metrics = evaluate_model(run_id)
        logger.info(f"Evaluation successful: {metrics}")
        return jsonify({"run_id": run_id, **metrics})
    except Exception as e:
        import traceback
        logger.error(f"Evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For debugging, set debug=True
    app.run(host="0.0.0.0", port=5006, debug=True)
