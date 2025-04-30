from flask import Flask, request, jsonify, send_file
import logging
import os
import traceback
from explain import explain_model, get_run_id_from_file

# Configure logging if not already configured in explain.py
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/explain', methods=['GET', 'POST'])
def explain():
    """Handle both GET and POST requests for explanations"""
    try:
        # Get run_id from either POST data or GET parameters
        if request.method == 'POST' and request.json:
            run_id = request.json.get('run_id')
        else:
            run_id = request.args.get('run_id')
            
        if not run_id:
            run_id = get_run_id_from_file()
            
        if not run_id:
            return jsonify({"error": "run_id is required and could not be found automatically"}), 400
        
        logger.info(f"Received explanation request for run_id: {run_id}")
        
        # Generate explanations
        results = explain_model(run_id)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error processing explanation request: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/plots/<filename>', methods=['GET'])
def get_plot(filename):
    """Serve plot images"""
    try:
        plot_path = os.path.join("/app/artifacts/plots", filename)
        if os.path.exists(plot_path):
            return send_file(plot_path, mimetype='image/png')
        else:
            return jsonify({"error": f"Plot file not found: {filename}"}), 404
    except Exception as e:
        logger.error(f"Error serving plot: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5005)
