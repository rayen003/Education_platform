"""
Confidence Management System.

This module provides a comprehensive system for managing confidence assessment,
prediction, calibration, and data collection for math services.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid
import hashlib
import difflib

from app.math_services.metrics.confidence import ConfidenceMetrics, ConfidenceLevel
from app.math_services.metrics.models import ConfidencePredictor, ConfidenceCalibrator
from app.math_services.metrics.calibration import CalibrationDataCollector

logger = logging.getLogger(__name__)

class ConfidenceManager:
    """
    Comprehensive system for managing confidence in math assessment.
    
    This class integrates:
    1. Real-time confidence assessment based on multiple factors
    2. ML-based confidence prediction using historical data
    3. Confidence calibration to improve reliability
    4. Data collection for continuous improvement
    """
    
    def __init__(self, 
                data_dir: Optional[str] = None,
                enable_ml: bool = True,
                calibration_method: str = "temperature"):
        """
        Initialize the confidence manager.
        
        Args:
            data_dir: Base directory for data storage (defaults to app/data)
            enable_ml: Whether to enable ML-based confidence prediction
            calibration_method: Method for calibration ('temperature' or 'isotonic')
        """
        # Set up data directories
        self.data_dir = data_dir or "app/data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create component directories
        cal_dir = os.path.join(self.data_dir, "calibration")
        model_dir = os.path.join(self.data_dir, "models")
        os.makedirs(cal_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.metrics = ConfidenceMetrics()
        self.data_collector = CalibrationDataCollector(data_dir=cal_dir)
        self.predictor = ConfidencePredictor(model_dir=model_dir) if enable_ml else None
        self.calibrator = ConfidenceCalibrator()
        
        # Configuration
        self.enable_ml = enable_ml
        self.calibration_method = calibration_method
        
        # Cache for problem IDs
        self.problem_id_cache = {}
        
        logger.info(f"Initialized ConfidenceManager with ML enabled: {enable_ml}")
    
    def assess_feedback_confidence(self, 
                                 state: Dict[str, Any], 
                                 verification_result: Optional[Dict[str, Any]] = None,
                                 model_uncertainty: Optional[float] = None) -> float:
        """
        Assess confidence in feedback given to the student.
        
        Args:
            state: The current problem state (dict or object)
            verification_result: Optional verification result from meta agent
            model_uncertainty: Optional model uncertainty estimate
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Ensure state is a dictionary (convert if needed)
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state

        # Use ML prediction if enabled
        if self.enable_ml and self.predictor:
            try:
                features = self._extract_feedback_features(state_dict, verification_result, model_uncertainty)
                raw_confidence = self.predictor.predict_confidence("feedback", features)
                
                # Apply calibration if configured
                if self.calibration_method and self.calibrator:
                    confidence = self.calibrator.calibrate_confidence(
                        "feedback", raw_confidence, method=self.calibration_method
                    )
                else:
                    confidence = raw_confidence
                    
                logger.debug(f"ML-based feedback confidence: {confidence:.3f}")
                return confidence
            except Exception as e:
                logger.warning(f"Failed to predict feedback confidence using ML: {e}")
                # Fall back to heuristic approach
        
        # Use heuristic approach as fallback
        base_confidence = self.metrics.assess_feedback_confidence(
            state_dict, verification_result, model_uncertainty
        )
        
        logger.debug(f"Heuristic feedback confidence: {base_confidence:.3f}")
        return base_confidence
    
    def assess_hint_confidence(self,
                             state: Dict[str, Any],
                             hint: str,
                             hint_number: int,
                             verification_result: Optional[Dict[str, Any]] = None) -> float:
        """
        Assess confidence in a hint provided to the student.
        
        Args:
            state: The current problem state (dict or object)
            hint: The hint text
            hint_number: Which hint this is (1, 2, 3, etc.)
            verification_result: Optional verification result from meta agent
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Ensure state is a dictionary (convert if needed)
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        # Use ML prediction if enabled
        if self.enable_ml and self.predictor:
            try:
                features = self._extract_hint_features(state_dict, hint, hint_number, verification_result)
                raw_confidence = self.predictor.predict_confidence("hints", features)
                
                # Apply calibration if configured
                if self.calibration_method and self.calibrator:
                    confidence = self.calibrator.calibrate_confidence(
                        "hints", raw_confidence, method=self.calibration_method
                    )
                else:
                    confidence = raw_confidence
                    
                logger.debug(f"ML-based hint confidence: {confidence:.3f}")
                return confidence
            except Exception as e:
                logger.warning(f"Failed to predict hint confidence using ML: {e}")
                # Fall back to heuristic approach
        
        # Use heuristic approach as fallback
        base_confidence = self.metrics.assess_hint_confidence(
            state_dict, hint, hint_number, verification_result
        )
        
        logger.debug(f"Heuristic hint confidence: {base_confidence:.3f}")
        return base_confidence
    
    def assess_analysis_confidence(self,
                                 state: Dict[str, Any],
                                 analysis_result: Dict[str, Any],
                                 verification_result: Optional[Dict[str, Any]] = None) -> float:
        """
        Assess confidence in analysis of student's calculation.
        
        Args:
            state: The current problem state (dict or object)
            analysis_result: Analysis results for the calculation
            verification_result: Optional verification result from meta agent
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Ensure state is a dictionary (convert if needed)
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        # Use ML prediction if enabled
        if self.enable_ml and self.predictor:
            try:
                features = self._extract_analysis_features(state_dict, analysis_result, verification_result)
                raw_confidence = self.predictor.predict_confidence("analysis", features)
                
                # Apply calibration if configured
                if self.calibration_method and self.calibrator:
                    confidence = self.calibrator.calibrate_confidence(
                        "analysis", raw_confidence, method=self.calibration_method
                    )
                else:
                    confidence = raw_confidence
                    
                logger.debug(f"ML-based analysis confidence: {confidence:.3f}")
                return confidence
            except Exception as e:
                logger.warning(f"Failed to predict analysis confidence using ML: {e}")
                # Fall back to heuristic approach
        
        # Use heuristic approach as fallback
        base_confidence = self.metrics.assess_analysis_confidence(
            state_dict, analysis_result, verification_result
        )
        
        logger.debug(f"Heuristic analysis confidence: {base_confidence:.3f}")
        return base_confidence
    
    def assess_chat_confidence(self,
                             state: Dict[str, Any],
                             follow_up_question: str,
                             response: str) -> float:
        """
        Assess confidence in a chat response to a follow-up question.
        
        Args:
            state: The current problem state (dict or object)
            follow_up_question: The student's follow-up question
            response: The response provided
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Ensure state is a dictionary (convert if needed)
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        # Use ML prediction if enabled
        if self.enable_ml and self.predictor:
            try:
                features = self._extract_chat_features(state_dict, follow_up_question, response)
                raw_confidence = self.predictor.predict_confidence("chat", features)
                
                # Apply calibration if configured
                if self.calibration_method and self.calibrator:
                    confidence = self.calibrator.calibrate_confidence(
                        "chat", raw_confidence, method=self.calibration_method
                    )
                else:
                    confidence = raw_confidence
                    
                logger.debug(f"ML-based chat confidence: {confidence:.3f}")
                return confidence
            except Exception as e:
                logger.warning(f"Failed to predict chat confidence using ML: {e}")
                # Fall back to heuristic approach
        
        # Use heuristic approach as fallback
        base_confidence = self.metrics.assess_chat_confidence(
            state_dict, follow_up_question, response
        )
        
        logger.debug(f"Heuristic chat confidence: {base_confidence:.3f}")
        return base_confidence
    
    def record_feedback_result(self,
                              state: Dict[str, Any],
                              predicted_confidence: float,
                              was_correct: bool) -> None:
        """
        Record a feedback interaction for training and calibration.
        
        Args:
            state: The current problem state (dict or object)
            predicted_confidence: The predicted confidence score
            was_correct: Whether the feedback was correct
        """
        # Ensure state is a dictionary (convert if needed)
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        # Get problem ID
        problem_id = self._get_problem_id(state_dict)
        
        # Extract information
        question = state_dict.get("question", "")
        student_answer = state_dict.get("student_answer", "")
        correct_answer = state_dict.get("correct_answer", "")
        
        # Record the interaction
        self.data_collector.record_feedback_interaction(
            problem_id=problem_id,
            question_text=question,
            predicted_confidence=predicted_confidence,
            was_correct=was_correct,
            student_answer=student_answer,
            correct_answer=correct_answer,
            metadata={
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.debug(f"Recorded feedback result - predicted: {predicted_confidence:.3f}, actual: {was_correct}")
    
    def record_hint_result(self,
                          state: Dict[str, Any],
                          hint: str,
                          hint_number: int,
                          predicted_confidence: float,
                          was_helpful: bool) -> None:
        """
        Record a hint interaction for training and calibration.
        
        Args:
            state: The current problem state (dict or object)
            hint: The hint text 
            hint_number: Which hint this is (1, 2, 3, etc.)
            predicted_confidence: The predicted confidence score
            was_helpful: Whether the hint was helpful
        """
        # Ensure state is a dictionary (convert if needed)
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        # Get problem ID
        problem_id = self._get_problem_id(state_dict)
        
        # Extract information
        question = state_dict.get("question", "")
        
        # Record the interaction
        self.data_collector.record_hint_interaction(
            problem_id=problem_id,
            question_text=question,
            hint_text=hint,
            hint_number=hint_number,
            predicted_confidence=predicted_confidence,
            was_helpful=was_helpful,
            metadata={
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.debug(f"Recorded hint result - predicted: {predicted_confidence:.3f}, actual: {was_helpful}")
    
    def record_analysis_result(self,
                             state: Dict[str, Any],
                             analysis_summary: str,
                             predicted_confidence: float,
                             was_correct: bool) -> None:
        """
        Record an analysis interaction for training and calibration.
        
        Args:
            state: The current problem state (dict or object)
            analysis_summary: Summary of the analysis
            predicted_confidence: The predicted confidence score
            was_correct: Whether the analysis was correct
        """
        # Ensure state is a dictionary (convert if needed)
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        # Get problem ID
        problem_id = self._get_problem_id(state_dict)
        
        # Extract information
        question = state_dict.get("question", "")
        
        # Record the interaction
        self.data_collector.record_analysis_interaction(
            problem_id=problem_id,
            question_text=question,
            analysis_summary=analysis_summary,
            predicted_confidence=predicted_confidence,
            was_correct=was_correct,
            metadata={
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.debug(f"Recorded analysis result - predicted: {predicted_confidence:.3f}, actual: {was_correct}")
    
    def record_chat_result(self,
                          state: Dict[str, Any],
                          follow_up_question: str,
                          response: str,
                          predicted_confidence: float,
                          was_helpful: bool) -> None:
        """
        Record a chat interaction for training and calibration.
        
        Args:
            state: The current problem state (dict or object)
            follow_up_question: The student's follow-up question
            response: The response provided
            predicted_confidence: The predicted confidence score
            was_helpful: Whether the response was helpful
        """
        # Ensure state is a dictionary (convert if needed)
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        # Get problem ID
        problem_id = self._get_problem_id(state_dict)
        
        # Extract information
        question = state_dict.get("question", "")
        
        # Record the interaction
        self.data_collector.record_chat_interaction(
            problem_id=problem_id,
            question_text=question,
            follow_up_question=follow_up_question,
            response=response,
            predicted_confidence=predicted_confidence,
            was_helpful=was_helpful,
            metadata={
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.debug(f"Recorded chat result - predicted: {predicted_confidence:.3f}, actual: {was_helpful}")
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train ML models using collected data.
        
        Returns:
            Dictionary of training results
        """
        if not self.enable_ml or not self.predictor:
            return {"error": "ML not enabled"}
        
        result = {}
        
        # Get training data from data collector
        training_data = self.data_collector.get_training_data()
        
        # Train models for each component type
        for component_type, data in training_data.items():
            if "error" in data:
                result[component_type] = {"error": data["error"]}
                continue
                
            if data["count"] < 10:
                result[component_type] = {"error": "Not enough data"}
                continue
            
            # Train model
            try:
                performance = self.predictor.train_linear_model(
                    component_type,
                    data["features"],
                    data["labels"]
                )
                result[component_type] = {
                    "success": True,
                    "samples": data["count"],
                    "performance": performance
                }
                
                logger.info(f"Trained model for {component_type} with {data['count']} samples")
            except Exception as e:
                result[component_type] = {"error": str(e)}
                logger.error(f"Error training model for {component_type}: {e}")
        
        return result
    
    def calibrate_models(self) -> Dict[str, Any]:
        """
        Calibrate confidence models using collected data.
        
        Returns:
            Dictionary of calibration results
        """
        if not self.calibrator:
            return {"error": "Calibrator not available"}
        
        result = {}
        
        # Get calibration data from data collector
        calibration_summary = self.data_collector.get_calibration_summary()
        
        # Calibrate for each component type
        for component_type, summary in calibration_summary.items():
            if "error" in summary:
                result[component_type] = {"error": summary["error"]}
                continue
            
            if summary.get("count", 0) < 10:
                result[component_type] = {"error": "Not enough data"}
                continue
            
            # Extract prediction/actual pairs from records
            try:
                records = self.data_collector.calibration_data.get(component_type, [])
                predictions = [r.get("predicted_confidence", 0.5) for r in records]
                actuals = [r.get("actual_outcome", 0.0) for r in records]
                
                # Fit temperature scaling
                temperature = self.calibrator.fit_temperature_scaling(
                    component_type, predictions, actuals
                )
                
                # Fit isotonic calibration (if enough data)
                if len(predictions) >= 50:
                    calibration_map = self.calibrator.fit_isotonic_calibration(
                        component_type, predictions, actuals
                    )
                    map_size = len(calibration_map)
                else:
                    map_size = 0
                
                result[component_type] = {
                    "success": True,
                    "samples": len(predictions),
                    "temperature": temperature,
                    "isotonic_bins": map_size
                }
                
                logger.info(f"Calibrated {component_type} with {len(predictions)} samples (T={temperature:.3f})")
            except Exception as e:
                result[component_type] = {"error": str(e)}
                logger.error(f"Error calibrating {component_type}: {e}")
        
        return result
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """
        Get calibration metrics for all components.
        
        Returns:
            Dictionary of calibration metrics
        """
        return self.data_collector.get_calibration_summary()
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Get model performance metrics.
        
        Returns:
            Dictionary of model metrics
        """
        if not self.enable_ml or not self.predictor:
            return {"error": "ML not enabled"}
        
        return self.predictor.get_model_summary()
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Convert a numerical confidence score to a descriptive level.
        
        Args:
            confidence: Numerical confidence score (0.0-1.0)
            
        Returns:
            Descriptive confidence level
        """
        level = self.metrics.get_confidence_level(confidence)
        return level.value
    
    def _get_problem_id(self, state: Dict[str, Any]) -> str:
        """
        Get a unique ID for the problem from the state.
        
        Args:
            state: The current problem state (dict or object)
            
        Returns:
            A unique problem ID
        """
        # Ensure state is a dictionary (convert if needed)
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        # Use an existing ID if available
        if "problem_id" in state_dict:
            return str(state_dict["problem_id"])
        
        # Otherwise generate one from question and timestamp
        question = state_dict.get("question", "")
        
        # Create a hash of the question for the ID
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        
        # Add a timestamp to make it unique
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        return f"prob_{question_hash}_{timestamp}"
    
    def _extract_feedback_features(self, 
                                 state: Dict[str, Any],
                                 verification_result: Optional[Dict[str, Any]] = None,
                                 model_uncertainty: Optional[float] = None) -> Dict[str, float]:
        """
        Extract features for feedback confidence prediction.
        
        Args:
            state: The problem state dictionary
            verification_result: Optional verification result
            model_uncertainty: Optional model uncertainty
            
        Returns:
            Dictionary of feature names to values
        """
        # Ensure state is a dictionary
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        features = {}
        
        # Question features
        question = state_dict.get("question", "")
        features["question_length"] = len(question) / 1000.0  # Normalize
        features["question_complexity"] = self.metrics._assess_question_complexity(question)
        
        # Answer features
        student_answer = state_dict.get("student_answer", "")
        correct_answer = state_dict.get("correct_answer", "")
        if student_answer and correct_answer:
            # Simple string similarity (0-1)
            features["answer_similarity"] = difflib.SequenceMatcher(None, student_answer, correct_answer).ratio()
        else:
            features["answer_similarity"] = 0.5
        
        # Verification features
        if verification_result:
            features["verification_confidence"] = verification_result.get("confidence", 0.5) / 100.0
            features["verification_valid"] = 1.0 if verification_result.get("is_valid", False) else 0.0
        else:
            features["verification_confidence"] = 0.5
            features["verification_valid"] = 0.5
        
        # Model uncertainty
        features["model_uncertainty"] = model_uncertainty if model_uncertainty is not None else 0.5
        
        return features
    
    def _extract_hint_features(self,
                             state: Dict[str, Any],
                             hint: str,
                             hint_number: int,
                             verification_result: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Extract features for hint confidence prediction.
        
        Args:
            state: The problem state dictionary
            hint: The hint text
            hint_number: Which hint this is
            verification_result: Optional verification result
            
        Returns:
            Dictionary of feature names to values
        """
        # Ensure state is a dictionary
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        features = {}
        
        # Question features
        question = state_dict.get("question", "")
        features["question_length"] = len(question) / 1000.0  # Normalize
        features["question_complexity"] = self.metrics._assess_question_complexity(question)
        
        # Hint features
        features["hint_length"] = len(hint) / 500.0  # Normalize
        features["hint_number"] = hint_number / 5.0  # Normalize (assuming max 5 hints)
        features["hint_specificity"] = self.metrics._assess_hint_specificity(hint)
        
        # Progressive hints should have lower confidence
        features["hint_progression"] = hint_number / 10.0  # Later hints get higher values
        
        # Verification features
        if verification_result:
            features["verification_confidence"] = verification_result.get("confidence", 0.5) / 100.0
            features["verification_valid"] = 1.0 if verification_result.get("is_valid", False) else 0.0
        else:
            features["verification_confidence"] = 0.5
            features["verification_valid"] = 0.5
        
        return features
    
    def _extract_analysis_features(self,
                                 state: Dict[str, Any],
                                 analysis_result: Dict[str, Any],
                                 verification_result: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Extract features for analysis confidence prediction.
        
        Args:
            state: The problem state dictionary
            analysis_result: The analysis result
            verification_result: Optional verification result
            
        Returns:
            Dictionary of feature names to values
        """
        # Ensure state is a dictionary
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        features = {}
        
        # Question features
        question = state_dict.get("question", "")
        features["question_length"] = len(question) / 1000.0  # Normalize
        features["question_complexity"] = self.metrics._assess_question_complexity(question)
        
        # Analysis features
        features["explanation_coherence"] = self.metrics._assess_explanation_coherence(analysis_result)
        features["is_correct"] = 1.0 if analysis_result.get("is_correct", False) else 0.0
        
        # Error type features
        error_type = analysis_result.get("error_type", "")
        features["has_error_type"] = 1.0 if error_type else 0.0
        
        # Verification features
        if verification_result:
            features["verification_confidence"] = verification_result.get("confidence", 0.5) / 100.0
            features["verification_valid"] = 1.0 if verification_result.get("is_valid", False) else 0.0
        else:
            features["verification_confidence"] = 0.5
            features["verification_valid"] = 0.5
        
        return features
    
    def _extract_chat_features(self,
                             state: Dict[str, Any],
                             follow_up_question: str,
                             response: str) -> Dict[str, float]:
        """
        Extract features for chat confidence prediction.
        
        Args:
            state: The problem state dictionary
            follow_up_question: The student's follow-up question
            response: The response provided
            
        Returns:
            Dictionary of feature names to values
        """
        # Ensure state is a dictionary
        if hasattr(state, 'to_dict') and callable(getattr(state, 'to_dict')):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        features = {}
        
        # Question features
        question = state_dict.get("question", "")
        features["question_length"] = len(question) / 1000.0  # Normalize
        features["question_complexity"] = self.metrics._assess_question_complexity(question)
        
        # Follow-up question features
        features["follow_up_length"] = len(follow_up_question) / 500.0  # Normalize
        features["follow_up_specificity"] = self.metrics._assess_question_specificity(follow_up_question)
        
        # Response features
        features["response_length"] = len(response) / 1000.0  # Normalize
        features["answer_clarity"] = self.metrics._assess_answer_clarity(response)
        features["context_relevance"] = self.metrics._assess_context_relevance(state_dict, follow_up_question, response)
        
        # Chat history features
        chat_history = state_dict.get("chat_history", [])
        features["chat_history_length"] = len(chat_history) / 10.0  # Normalize
        
        return features 