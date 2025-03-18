"""
Machine Learning Models for Confidence Prediction.

This module provides machine learning models for predicting confidence
levels based on problem characteristics and historical performance.
"""

import logging
import json
import os
import pickle
import random
from typing import Dict, Any, List, Optional, Union, Tuple
import statistics
from datetime import datetime
import math

logger = logging.getLogger(__name__)

class ConfidencePredictor:
    """
    Machine learning model for predicting confidence levels.
    
    Supports:
    1. Linear regression for confidence prediction
    2. Decision trees for confidence bucketing
    3. Feature importance analysis
    4. Model validation and testing
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the confidence predictor.
        
        Args:
            model_dir: Directory to store trained models (defaults to app/data/models)
        """
        self.model_dir = model_dir or "app/data/models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models for different component types
        self.models = {
            "feedback": None,
            "hints": None,
            "analysis": None,
            "chat": None
        }
        
        # Feature weights for linear models
        self.feature_weights = {
            "feedback": {},
            "hints": {},
            "analysis": {},
            "chat": {}
        }
        
        # Model performance metrics
        self.performance = {
            "feedback": {},
            "hints": {},
            "analysis": {},
            "chat": {}
        }
        
        # Load existing models if available
        self._load_models()
        
        logger.info(f"Initialized ConfidencePredictor with model dir: {self.model_dir}")
    
    def train_linear_model(self, 
                          component_type: str, 
                          features: List[Dict[str, float]], 
                          labels: List[float],
                          test_split: float = 0.2) -> Dict[str, Any]:
        """
        Train a simple linear regression model for confidence prediction.
        
        Args:
            component_type: Type of component (feedback, hints, etc.)
            features: List of feature dictionaries
            labels: List of actual outcomes (0.0 or 1.0)
            test_split: Fraction of data to use for testing
            
        Returns:
            Dictionary of model performance metrics
        """
        if len(features) < 10:
            logger.warning(f"Not enough data to train model for {component_type}")
            return {"error": "Not enough training data"}
        
        # Split data into training and test sets
        split_idx = int(len(features) * (1 - test_split))
        if split_idx <= 0:
            split_idx = 1
            
        # Shuffle data together
        combined = list(zip(features, labels))
        random.shuffle(combined)
        features, labels = zip(*combined)
        
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        test_features = features[split_idx:]
        test_labels = labels[split_idx:]
        
        # Extract feature names from the first feature dictionary
        feature_names = list(train_features[0].keys())
        
        # Train the model (simple linear regression)
        weights = self._train_linear_regression(train_features, train_labels, feature_names)
        
        # Save the model
        self.feature_weights[component_type] = weights
        self.models[component_type] = {
            "type": "linear",
            "weights": weights,
            "feature_names": feature_names,
            "trained_at": datetime.now().isoformat()
        }
        
        # Test the model
        predictions = [
            self._predict_with_linear_model(self.models[component_type], feature)
            for feature in test_features
        ]
        
        # Calculate performance metrics
        mse = sum((p - a) ** 2 for p, a in zip(predictions, test_labels)) / len(predictions)
        mae = sum(abs(p - a) for p, a in zip(predictions, test_labels)) / len(predictions)
        
        # Calculate calibration by buckets
        calibration = self._calculate_calibration(predictions, test_labels)
        
        # Save performance metrics
        performance = {
            "mse": mse,
            "mae": mae,
            "test_count": len(test_features),
            "train_count": len(train_features),
            "calibration": calibration
        }
        
        self.performance[component_type] = performance
        
        # Save the model to disk
        self._save_models()
        
        logger.info(f"Trained linear model for {component_type} with MSE: {mse:.4f}")
        return performance
    
    def predict_confidence(self, 
                          component_type: str, 
                          features: Dict[str, float]) -> float:
        """
        Predict confidence for a new instance.
        
        Args:
            component_type: Type of component (feedback, hints, etc.)
            features: Feature dictionary for the instance
            
        Returns:
            Predicted confidence (0.0-1.0)
        """
        # Check if we have a model for this component
        model = self.models.get(component_type)
        if not model:
            logger.warning(f"No model found for {component_type}, using default confidence")
            return 0.7  # Default confidence if no model
        
        # Make prediction based on model type
        if model["type"] == "linear":
            prediction = self._predict_with_linear_model(model, features)
        else:
            # Fallback if unknown model type
            logger.warning(f"Unknown model type for {component_type}")
            prediction = 0.7
        
        # Ensure prediction is between 0 and 1
        prediction = max(0.0, min(1.0, prediction))
        
        return prediction
    
    def get_feature_importance(self, component_type: str) -> Dict[str, float]:
        """
        Get feature importance for a component type.
        
        Args:
            component_type: Type of component (feedback, hints, etc.)
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        model = self.models.get(component_type)
        if not model or model["type"] != "linear":
            return {}
        
        # For linear models, feature importance is the absolute weight
        weights = model["weights"]
        feature_names = model["feature_names"]
        
        # Calculate importance as normalized absolute weights
        abs_weights = {name: abs(weights.get(name, 0.0)) for name in feature_names}
        max_weight = max(abs_weights.values()) if abs_weights else 1.0
        
        # Normalize to 0-1 scale
        importance = {name: weight / max_weight for name, weight in abs_weights.items()}
        
        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def get_model_summary(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of the model(s).
        
        Args:
            component_type: Optional specific component type
            
        Returns:
            Dictionary with model summaries
        """
        result = {}
        
        if component_type:
            model = self.models.get(component_type)
            if model:
                result[component_type] = {
                    "type": model.get("type", "unknown"),
                    "trained_at": model.get("trained_at", "unknown"),
                    "feature_count": len(model.get("feature_names", [])),
                    "performance": self.performance.get(component_type, {})
                }
            else:
                result[component_type] = {"error": "No model found"}
        else:
            # Get summary for all components
            for comp_type, model in self.models.items():
                if model:
                    result[comp_type] = {
                        "type": model.get("type", "unknown"),
                        "trained_at": model.get("trained_at", "unknown"),
                        "feature_count": len(model.get("feature_names", [])),
                        "performance": self.performance.get(comp_type, {})
                    }
                else:
                    result[comp_type] = {"error": "No model found"}
        
        return result
    
    def _train_linear_regression(self, 
                               features: List[Dict[str, float]], 
                               labels: List[float],
                               feature_names: List[str]) -> Dict[str, float]:
        """
        Train a simple linear regression model.
        
        Args:
            features: List of feature dictionaries
            labels: List of target values
            feature_names: List of feature names to use
            
        Returns:
            Dictionary of feature weights
        """
        # Initialize weights with small random values
        weights = {name: random.uniform(-0.1, 0.1) for name in feature_names}
        
        # Add bias term
        weights["bias"] = random.uniform(-0.1, 0.1)
        
        # Simple stochastic gradient descent
        learning_rate = 0.01
        epochs = 100
        batch_size = min(32, len(features))
        
        for epoch in range(epochs):
            # Shuffle data
            combined = list(zip(features, labels))
            random.shuffle(combined)
            features_shuffled, labels_shuffled = zip(*combined)
            
            total_loss = 0.0
            
            # Process in batches
            for i in range(0, len(features_shuffled), batch_size):
                batch_features = features_shuffled[i:i+batch_size]
                batch_labels = labels_shuffled[i:i+batch_size]
                
                # Calculate gradients for the batch
                gradients = {name: 0.0 for name in weights.keys()}
                
                for feature_dict, label in zip(batch_features, batch_labels):
                    # Make prediction
                    prediction = weights["bias"]
                    for name in feature_names:
                        prediction += weights[name] * feature_dict.get(name, 0.0)
                    
                    # Calculate error
                    error = prediction - label
                    total_loss += error ** 2
                    
                    # Update gradients
                    gradients["bias"] += error
                    for name in feature_names:
                        gradients[name] += error * feature_dict.get(name, 0.0)
                
                # Apply gradients
                for name in weights.keys():
                    weights[name] -= learning_rate * gradients[name] / len(batch_features)
            
            # Calculate epoch loss
            avg_loss = total_loss / len(features)
            
            # Early stopping if loss is low enough
            if avg_loss < 0.01:
                logger.debug(f"Early stopping at epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
                break
            
            # Decrease learning rate over time
            if epoch % 10 == 0:
                learning_rate *= 0.9
        
        return weights
    
    def _predict_with_linear_model(self, model: Dict[str, Any], features: Dict[str, float]) -> float:
        """
        Make a prediction using a linear model.
        
        Args:
            model: The linear model
            features: Feature dictionary
            
        Returns:
            Predicted value
        """
        weights = model["weights"]
        feature_names = model["feature_names"]
        
        # Start with bias
        prediction = weights.get("bias", 0.0)
        
        # Add weighted features
        for name in feature_names:
            if name in features:
                prediction += weights.get(name, 0.0) * features[name]
        
        # Apply sigmoid function for probability output
        prediction = 1.0 / (1.0 + math.exp(-prediction))
        
        return prediction
    
    def _calculate_calibration(self, predictions: List[float], actuals: List[float]) -> Dict[str, Any]:
        """
        Calculate calibration metrics for predictions.
        
        Args:
            predictions: List of predictions
            actuals: List of actual values
            
        Returns:
            Dictionary of calibration metrics
        """
        # Calculate calibration by buckets
        buckets = {}
        
        for pred, actual in zip(predictions, actuals):
            # Round to nearest 0.1
            bucket = round(pred * 10) / 10
            if bucket not in buckets:
                buckets[bucket] = {"count": 0, "correct": 0}
            
            buckets[bucket]["count"] += 1
            buckets[bucket]["correct"] += actual
        
        # Calculate accuracy per bucket
        for bucket_val, data in buckets.items():
            if data["count"] > 0:
                data["accuracy"] = data["correct"] / data["count"]
                data["calibration_error"] = abs(bucket_val - data["accuracy"])
            else:
                data["accuracy"] = 0.0
                data["calibration_error"] = 0.0
        
        # Calculate ECE (Expected Calibration Error)
        total_samples = len(predictions)
        ece = 0.0
        
        for bucket_val, data in buckets.items():
            weight = data["count"] / total_samples
            ece += weight * data["calibration_error"]
        
        return {
            "buckets": buckets,
            "ece": ece
        }
    
    def _save_models(self) -> None:
        """Save models to disk."""
        try:
            for component_type, model in self.models.items():
                if model:
                    # Save as JSON for simpler models
                    model_path = os.path.join(self.model_dir, f"{component_type}_model.json")
                    with open(model_path, 'w') as f:
                        json.dump(model, f, indent=2)
            
            # Save performance metrics
            perf_path = os.path.join(self.model_dir, "model_performance.json")
            with open(perf_path, 'w') as f:
                json.dump(self.performance, f, indent=2)
            
            logger.debug("Saved models to disk")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self) -> None:
        """Load models from disk if available."""
        try:
            # Load models
            for component_type in self.models.keys():
                model_path = os.path.join(self.model_dir, f"{component_type}_model.json")
                if os.path.exists(model_path):
                    with open(model_path, 'r') as f:
                        self.models[component_type] = json.load(f)
                        
                        # Extract weights for convenience
                        if self.models[component_type]["type"] == "linear":
                            self.feature_weights[component_type] = self.models[component_type]["weights"]
            
            # Load performance metrics
            perf_path = os.path.join(self.model_dir, "model_performance.json")
            if os.path.exists(perf_path):
                with open(perf_path, 'r') as f:
                    self.performance = json.load(f)
            
            logger.info("Loaded existing models")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

class ConfidenceCalibrator:
    """
    Calibrates confidence scores to better match empirical outcomes.
    
    Implements:
    1. Temperature scaling for confidence calibration
    2. Platt scaling for binary classification confidence
    3. Isotonic regression for flexible calibration mapping
    """
    
    def __init__(self):
        """Initialize the confidence calibrator."""
        # Calibration maps for different components
        self.calibration_maps = {
            "feedback": {},
            "hints": {},
            "analysis": {},
            "chat": {}
        }
        
        # Calibration parameters
        self.params = {
            "feedback": {"temperature": 1.0},
            "hints": {"temperature": 1.0},
            "analysis": {"temperature": 1.0},
            "chat": {"temperature": 1.0}
        }
        
        logger.info("Initialized ConfidenceCalibrator")
    
    def fit_temperature_scaling(self, 
                              component_type: str, 
                              raw_confidences: List[float],
                              actuals: List[float]) -> float:
        """
        Fit temperature scaling parameter to calibrate confidence scores.
        
        Args:
            component_type: Type of component (feedback, hints, etc.)
            raw_confidences: Uncalibrated confidence scores
            actuals: Actual outcomes (0.0 or 1.0)
            
        Returns:
            Optimal temperature value
        """
        if len(raw_confidences) < 10:
            logger.warning(f"Not enough data for {component_type} to fit temperature")
            return 1.0
        
        # Find optimal temperature using binary search
        min_t = 0.1
        max_t = 5.0
        best_t = 1.0
        best_loss = float('inf')
        
        for _ in range(20):  # 20 iterations should be enough for good precision
            mid_t = (min_t + max_t) / 2
            
            # Calculate loss for this temperature
            calibrated = [self._apply_temperature(c, mid_t) for c in raw_confidences]
            loss = sum((c - a) ** 2 for c, a in zip(calibrated, actuals)) / len(calibrated)
            
            # Check if this is better
            if loss < best_loss:
                best_loss = loss
                best_t = mid_t
            
            # Search left or right half
            left_t = (min_t + mid_t) / 2
            right_t = (mid_t + max_t) / 2
            
            left_calibrated = [self._apply_temperature(c, left_t) for c in raw_confidences]
            left_loss = sum((c - a) ** 2 for c, a in zip(left_calibrated, actuals)) / len(left_calibrated)
            
            right_calibrated = [self._apply_temperature(c, right_t) for c in raw_confidences]
            right_loss = sum((c - a) ** 2 for c, a in zip(right_calibrated, actuals)) / len(right_calibrated)
            
            if left_loss < right_loss:
                max_t = mid_t
            else:
                min_t = mid_t
        
        # Save optimal temperature
        self.params[component_type]["temperature"] = best_t
        
        logger.info(f"Fitted temperature for {component_type}: {best_t:.3f}")
        return best_t
    
    def fit_isotonic_calibration(self,
                               component_type: str,
                               raw_confidences: List[float],
                               actuals: List[float],
                               num_bins: int = 10) -> Dict[float, float]:
        """
        Fit isotonic calibration map using binning.
        
        Args:
            component_type: Type of component (feedback, hints, etc.)
            raw_confidences: Uncalibrated confidence scores
            actuals: Actual outcomes (0.0 or 1.0)
            num_bins: Number of bins to use
            
        Returns:
            Calibration map (bin value -> calibrated value)
        """
        if len(raw_confidences) < num_bins * 5:
            logger.warning(f"Not enough data for {component_type} to fit isotonic calibration")
            return {i/10: i/10 for i in range(11)}  # Identity mapping
        
        # Create bins
        bins = {}
        for conf, actual in zip(raw_confidences, actuals):
            bin_idx = min(int(conf * num_bins), num_bins - 1)
            bin_val = (bin_idx + 0.5) / num_bins
            
            if bin_val not in bins:
                bins[bin_val] = {"sum": 0.0, "count": 0}
            
            bins[bin_val]["sum"] += actual
            bins[bin_val]["count"] += 1
        
        # Calculate average actual value per bin
        calibration_map = {}
        for bin_val, data in bins.items():
            if data["count"] > 0:
                calibration_map[bin_val] = data["sum"] / data["count"]
            else:
                calibration_map[bin_val] = bin_val  # Identity if no data
        
        # Enforce monotonicity (isotonic regression constraint)
        sorted_bins = sorted(calibration_map.keys())
        for i in range(1, len(sorted_bins)):
            prev_bin = sorted_bins[i-1]
            curr_bin = sorted_bins[i]
            
            if calibration_map[curr_bin] < calibration_map[prev_bin]:
                # Merge bins
                combined_count = bins[prev_bin]["count"] + bins[curr_bin]["count"]
                combined_sum = bins[prev_bin]["sum"] + bins[curr_bin]["sum"]
                combined_value = combined_sum / combined_count if combined_count > 0 else (prev_bin + curr_bin) / 2
                
                calibration_map[prev_bin] = combined_value
                calibration_map[curr_bin] = combined_value
        
        # Save calibration map
        self.calibration_maps[component_type] = calibration_map
        
        logger.info(f"Fitted isotonic calibration for {component_type} with {len(calibration_map)} bins")
        return calibration_map
    
    def calibrate_confidence(self,
                           component_type: str,
                           raw_confidence: float,
                           method: str = "temperature") -> float:
        """
        Calibrate a confidence score.
        
        Args:
            component_type: Type of component (feedback, hints, etc.)
            raw_confidence: Uncalibrated confidence score
            method: Calibration method ('temperature' or 'isotonic')
            
        Returns:
            Calibrated confidence score
        """
        if method == "temperature":
            # Apply temperature scaling
            temperature = self.params[component_type].get("temperature", 1.0)
            return self._apply_temperature(raw_confidence, temperature)
        
        elif method == "isotonic":
            # Apply isotonic calibration
            calibration_map = self.calibration_maps.get(component_type, {})
            if not calibration_map:
                return raw_confidence  # No calibration map, return original
            
            # Find closest bin
            bin_values = sorted(calibration_map.keys())
            if not bin_values:
                return raw_confidence
            
            # Binary search for closest bin
            closest_bin = min(bin_values, key=lambda x: abs(x - raw_confidence))
            return calibration_map[closest_bin]
        
        else:
            logger.warning(f"Unknown calibration method: {method}")
            return raw_confidence
    
    def _apply_temperature(self, confidence: float, temperature: float) -> float:
        """
        Apply temperature scaling to a confidence score.
        
        Args:
            confidence: Raw confidence score
            temperature: Temperature parameter (higher = more uncertainty)
            
        Returns:
            Calibrated confidence score
        """
        if temperature <= 0:
            # Invalid temperature, return original
            return confidence
        
        # Convert confidence to logit
        epsilon = 1e-10  # To avoid log(0) or log(1)
        confidence_clipped = max(epsilon, min(1 - epsilon, confidence))
        logit = math.log(confidence_clipped / (1 - confidence_clipped))
        
        # Apply temperature scaling
        scaled_logit = logit / temperature
        
        # Convert back to probability
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        
        return calibrated 