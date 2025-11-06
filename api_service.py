"""
Production Inference API Service for DDoS Detection
Serves predictions via REST endpoints with model optimization
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = Path('results/ddos_model.h5')
QUANTIZED_MODEL_PATH = Path('results/ddos_model_quantized_int8.tflite')
FEATURE_NAMES_PATH = Path(
    'data/optimized/clean_partitions/selected_features.json')

# Global state
model = None
quantized_interpreter = None
feature_names = None
request_count = 0
request_lock = threading.Lock()
prediction_history = []
prediction_history_lock = threading.Lock()
MAX_HISTORY = 200  # Keep last 200 predictions


class ModelManager:
    """Handles model loading and inference"""

    def __init__(self):
        self.standard_model = None
        self.quantized_model = None
        self.feature_names = None
        self.model_loaded_at = None

    def load_models(self):
        """Load both standard and quantized models"""
        try:
            logger.info("🔄 Loading standard model...")
            self.standard_model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"✅ Standard model loaded: {MODEL_PATH}")

            # Try to load quantized model for inference
            if QUANTIZED_MODEL_PATH.exists():
                self.quantized_model = tf.lite.Interpreter(
                    model_path=str(QUANTIZED_MODEL_PATH)
                )
                self.quantized_model.allocate_tensors()
                logger.info(
                    f"✅ Quantized model loaded: {QUANTIZED_MODEL_PATH}")
            else:
                logger.warning(
                    f"⚠️ Quantized model not found: {QUANTIZED_MODEL_PATH}")

            # Load feature names
            if FEATURE_NAMES_PATH.exists():
                with open(FEATURE_NAMES_PATH) as f:
                    self.feature_names = json.load(f)
                logger.info(
                    f"✅ Feature names loaded: {len(self.feature_names)} features")
            else:
                logger.warning(
                    f"⚠️ Feature names not found: {FEATURE_NAMES_PATH}")

            self.model_loaded_at = datetime.now()
            logger.info("✅ All models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False

    def predict_standard(self, features):
        """Inference using standard TensorFlow model"""
        try:
            features = np.array(features, dtype=np.float32).reshape(1, -1)
            prediction = self.standard_model.predict(features, verbose=0)[0]
            confidence = float(prediction[0])

            return {
                'model': 'standard',
                'prediction': 'Attack' if confidence > 0.5 else 'Benign',
                'confidence': confidence,
                'risk_score': confidence
            }
        except Exception as e:
            logger.error(f"❌ Standard model inference failed: {e}")
            return None

    def predict_quantized(self, features):
        """Inference using quantized TFLite model (faster, smaller)"""
        try:
            # Get input/output details
            input_details = self.quantized_model.get_input_details()
            output_details = self.quantized_model.get_output_details()

            # Prepare input
            features = np.array(features, dtype=np.float32).reshape(1, -1)

            # Quantize if needed
            if input_details[0]['dtype'] == np.int8:
                input_scale, input_zero_point = input_details[0]['quantization']
                features = (features / input_scale +
                            input_zero_point).astype(np.int8)

            # Inference
            self.quantized_model.set_tensor(
                input_details[0]['index'], features)
            self.quantized_model.invoke()

            # Get output
            output_data = self.quantized_model.get_tensor(
                output_details[0]['index'])

            # Dequantize if needed
            if output_details[0]['dtype'] == np.int8:
                output_scale, output_zero_point = output_details[0]['quantization']
                confidence = float(
                    (output_data[0][0] - output_zero_point) * output_scale)
            else:
                confidence = float(output_data[0][0])

            # Clamp to [0, 1]
            confidence = np.clip(confidence, 0, 1)

            return {
                'model': 'quantized',
                'prediction': 'Attack' if confidence > 0.5 else 'Benign',
                'confidence': confidence,
                'risk_score': confidence
            }
        except Exception as e:
            logger.error(f"❌ Quantized model inference failed: {e}")
            return None


# Initialize model manager
model_manager = ModelManager()


@app.before_request
def track_request():
    """Track request statistics"""
    global request_count
    with request_lock:
        request_count += 1


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': model_manager.standard_model is not None,
        'uptime_seconds': (datetime.now() - model_manager.model_loaded_at).total_seconds()
    }), 200


@app.route('/info', methods=['GET'])
def info():
    """Get API and model information"""
    return jsonify({
        'service': 'DDoS Detection API',
        'version': '1.0.0',
        'status': 'production',
        'models': {
            'standard': MODEL_PATH.exists(),
            'quantized': QUANTIZED_MODEL_PATH.exists()
        },
        'features_count': len(model_manager.feature_names) if model_manager.feature_names else 0,
        'total_requests': request_count,
        'model_loaded_at': model_manager.model_loaded_at.isoformat() if model_manager.model_loaded_at else None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single sample prediction endpoint

    Expected JSON:
    {
        "features": [f1, f2, ..., f30],
        "model": "standard" or "quantized" (default: "standard")
    }

    Returns:
    {
        "prediction": "Attack" or "Benign",
        "confidence": 0.0-1.0,
        "risk_score": 0.0-1.0,
        "processing_time_ms": X
    }
    """
    try:
        start_time = time.time()
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" field'}), 400

        features = data['features']
        model_choice = data.get('model', 'standard')

        # Validate feature count
        if len(features) != 30:
            return jsonify({
                'error': f'Expected 30 features, got {len(features)}'
            }), 400

        # Run inference
        if model_choice == 'quantized' and model_manager.quantized_model:
            result = model_manager.predict_quantized(features)
        else:
            result = model_manager.predict_standard(features)

        if result is None:
            return jsonify({'error': 'Inference failed'}), 500

        result['processing_time_ms'] = (time.time() - start_time) * 1000
        result['timestamp'] = datetime.now().isoformat()

        # Store in prediction history
        with prediction_history_lock:
            prediction_history.append(result)
            if len(prediction_history) > MAX_HISTORY:
                prediction_history.pop(0)

        logger.info(
            f"✅ Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint (process multiple samples)

    Expected JSON:
    {
        "samples": [
            {"features": [f1, f2, ..., f30]},
            {"features": [f1, f2, ..., f30]},
            ...
        ],
        "model": "standard" or "quantized"
    }

    Returns:
    {
        "predictions": [
            {"prediction": "...", "confidence": ..., "risk_score": ...},
            ...
        ],
        "total_samples": X,
        "attacks_detected": X,
        "processing_time_ms": X
    }
    """
    try:
        start_time = time.time()
        data = request.get_json()

        if not data or 'samples' not in data:
            return jsonify({'error': 'Missing "samples" field'}), 400

        samples = data['samples']
        model_choice = data.get('model', 'standard')

        if not isinstance(samples, list) or len(samples) == 0:
            return jsonify({'error': 'samples must be non-empty list'}), 400

        predictions = []
        attacks_detected = 0

        for i, sample in enumerate(samples):
            if 'features' not in sample:
                return jsonify({
                    'error': f'Sample {i} missing "features" field'
                }), 400

            features = sample['features']
            if len(features) != 30:
                return jsonify({
                    'error': f'Sample {i}: Expected 30 features, got {len(features)}'
                }), 400

            # Run inference
            if model_choice == 'quantized' and model_manager.quantized_model:
                result = model_manager.predict_quantized(features)
            else:
                result = model_manager.predict_standard(features)

            if result is None:
                return jsonify({
                    'error': f'Inference failed on sample {i}'
                }), 500

            if result['prediction'] == 'Attack':
                attacks_detected += 1

            predictions.append(result)

        response = {
            'predictions': predictions,
            'total_samples': len(samples),
            'attacks_detected': attacks_detected,
            'attack_rate': attacks_detected / len(samples),
            'processing_time_ms': (time.time() - start_time) * 1000,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(
            f"✅ Batch prediction: {len(samples)} samples, {attacks_detected} attacks detected")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Get recent predictions history (for dashboard)"""
    with prediction_history_lock:
        recent = prediction_history[-50:] if len(
            prediction_history) > 50 else prediction_history

    return jsonify({
        'predictions': recent,
        'total_count': len(prediction_history),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/metrics', methods=['GET'])
def metrics():
    """Get current API metrics and statistics"""
    uptime = (datetime.now(
    ) - model_manager.model_loaded_at).total_seconds() if model_manager.model_loaded_at else 0

    return jsonify({
        'requests_total': request_count,
        'uptime_seconds': uptime,
        'models_available': {
            'standard': model_manager.standard_model is not None,
            'quantized': model_manager.quantized_model is not None
        },
        'features_count': len(model_manager.feature_names) if model_manager.feature_names else 0,
        'api_version': '1.0.0',
        'status': 'active'
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/health',
            '/info',
            '/predict (POST)',
            '/batch (POST)',
            '/metrics'
        ]
    }), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    logger.error(f"❌ Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load models
    if model_manager.load_models():
        logger.info("🚀 Starting DDoS Detection API Service")
        logger.info("📌 Endpoints: /health, /info, /predict, /batch, /metrics")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        logger.error("❌ Failed to load models. Exiting.")
        exit(1)
