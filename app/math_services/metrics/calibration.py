"""
Confidence Calibration Module.

This module collects and manages data for calibrating confidence metrics based on
actual performance across different types of math problems and student interactions.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import statistics
import pickle
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class CalibrationDataCollector:
    """
    Collects and manages calibration data for confidence metrics.
    
    This class provides tools for:
    1. Recording student interactions
    2. Tracking confidence predictions vs. actual outcomes
    3. Saving and loading calibration datasets
    4. Summarizing calibration performance
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the calibration data collector.
        
        Args:
            data_dir: Directory to store calibration data (defaults to app/data/calibration)
        """
        self.data_dir = data_dir or "app/data/calibration"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Main calibration dataset
        self.calibration_data = {
            "feedback": [],
            "hints": [],
            "analysis": [],
            "chat": []
        }
        
        # Problem characteristics dataset
        self.problem_data = {}
        
        # Lock for thread safety when updating data
        self.lock = threading.Lock()
        
        # Load existing data if available
        self._load_data()
        
        # Make feature weights configurable rather than hardcoded
        self.feature_weights = {
            "feedback": {
                "verification_result": 0.5,
                "answer_proximity": 0.2,
                "question_complexity": 0.15,
                "model_uncertainty": 0.15,
            },
            # similar for other components
        }
        
        logger.info(f"Initialized CalibrationDataCollector with data dir: {self.data_dir}")
    
    def record_feedback_interaction(self, 
                                   problem_id: str,
                                   question_text: str,
                                   predicted_confidence: float,
                                   was_correct: bool,
                                   student_answer: str,
                                   correct_answer: str,
                                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a feedback interaction for calibration.
        
        Args:
            problem_id: Unique ID for the problem
            question_text: The question text
            predicted_confidence: The confidence score predicted by the system
            was_correct: Whether the feedback assessment was correct
            student_answer: The student's answer
            correct_answer: The correct answer
            metadata: Additional metadata about the interaction
        """
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Create the record
            record = {
                "problem_id": problem_id,
                "timestamp": timestamp,
                "predicted_confidence": predicted_confidence,
                "actual_outcome": 1.0 if was_correct else 0.0,
                "student_answer": student_answer,
                "correct_answer": correct_answer,
                "metadata": metadata or {}
            }
            
            # Add to calibration data
            self.calibration_data["feedback"].append(record)
            
            # Update problem data if not already present
            if problem_id not in self.problem_data:
                self.problem_data[problem_id] = {
                    "id": problem_id,
                    "question_text": question_text,
                    "correct_answer": correct_answer,
                    "interaction_count": 0,
                    "first_seen": timestamp,
                    "characteristics": self._extract_problem_characteristics(question_text)
                }
            
            # Update interaction count
            self.problem_data[problem_id]["interaction_count"] += 1
            self.problem_data[problem_id]["last_interaction"] = timestamp
            
            # Save data periodically (every 10 records)
            if len(self.calibration_data["feedback"]) % 10 == 0:
                self._save_data()
                
            logger.debug(f"Recorded feedback interaction for problem {problem_id}")
    
    def record_hint_interaction(self,
                               problem_id: str,
                               question_text: str,
                               hint_text: str,
                               hint_number: int,
                               predicted_confidence: float,
                               was_helpful: bool,
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a hint interaction for calibration.
        
        Args:
            problem_id: Unique ID for the problem
            question_text: The question text
            hint_text: The hint provided
            hint_number: Which hint this was (1st, 2nd, etc.)
            predicted_confidence: The confidence score predicted by the system
            was_helpful: Whether the hint was helpful (based on subsequent student success)
            metadata: Additional metadata about the interaction
        """
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Create the record
            record = {
                "problem_id": problem_id,
                "timestamp": timestamp,
                "hint_text": hint_text,
                "hint_number": hint_number,
                "predicted_confidence": predicted_confidence,
                "actual_outcome": 1.0 if was_helpful else 0.0,
                "metadata": metadata or {}
            }
            
            # Add to calibration data
            self.calibration_data["hints"].append(record)
            
            # Update problem data if not already present
            if problem_id not in self.problem_data:
                self.problem_data[problem_id] = {
                    "id": problem_id,
                    "question_text": question_text,
                    "interaction_count": 0,
                    "first_seen": timestamp,
                    "characteristics": self._extract_problem_characteristics(question_text)
                }
            
            # Update interaction count
            self.problem_data[problem_id]["interaction_count"] += 1
            self.problem_data[problem_id]["last_interaction"] = timestamp
            
            # Save data periodically
            if len(self.calibration_data["hints"]) % 10 == 0:
                self._save_data()
                
            logger.debug(f"Recorded hint interaction for problem {problem_id}")
    
    def record_analysis_interaction(self,
                                   problem_id: str,
                                   question_text: str,
                                   analysis_summary: str,
                                   predicted_confidence: float,
                                   was_correct: bool,
                                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an analysis interaction for calibration.
        
        Args:
            problem_id: Unique ID for the problem
            question_text: The question text
            analysis_summary: Summary of the analysis provided
            predicted_confidence: The confidence score predicted by the system
            was_correct: Whether the analysis was correct
            metadata: Additional metadata about the interaction
        """
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Create the record
            record = {
                "problem_id": problem_id,
                "timestamp": timestamp,
                "analysis_summary": analysis_summary,
                "predicted_confidence": predicted_confidence,
                "actual_outcome": 1.0 if was_correct else 0.0,
                "metadata": metadata or {}
            }
            
            # Add to calibration data
            self.calibration_data["analysis"].append(record)
            
            # Update problem data if not already present
            if problem_id not in self.problem_data:
                self.problem_data[problem_id] = {
                    "id": problem_id,
                    "question_text": question_text,
                    "interaction_count": 0,
                    "first_seen": timestamp,
                    "characteristics": self._extract_problem_characteristics(question_text)
                }
            
            # Update interaction count
            self.problem_data[problem_id]["interaction_count"] += 1
            self.problem_data[problem_id]["last_interaction"] = timestamp
            
            # Save data periodically
            if len(self.calibration_data["analysis"]) % 10 == 0:
                self._save_data()
                
            logger.debug(f"Recorded analysis interaction for problem {problem_id}")
    
    def record_chat_interaction(self,
                               problem_id: str,
                               question_text: str,
                               follow_up_question: str,
                               response: str,
                               predicted_confidence: float,
                               was_helpful: bool,
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a chat interaction for calibration.
        
        Args:
            problem_id: Unique ID for the problem
            question_text: The question text
            follow_up_question: The student's follow-up question
            response: The system's response
            predicted_confidence: The confidence score predicted by the system
            was_helpful: Whether the response was helpful (based on feedback or subsequent success)
            metadata: Additional metadata about the interaction
        """
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Create the record
            record = {
                "problem_id": problem_id,
                "timestamp": timestamp,
                "follow_up_question": follow_up_question,
                "response": response,
                "predicted_confidence": predicted_confidence,
                "actual_outcome": 1.0 if was_helpful else 0.0,
                "metadata": metadata or {}
            }
            
            # Add to calibration data
            self.calibration_data["chat"].append(record)
            
            # Update problem data if not already present
            if problem_id not in self.problem_data:
                self.problem_data[problem_id] = {
                    "id": problem_id,
                    "question_text": question_text,
                    "interaction_count": 0,
                    "first_seen": timestamp,
                    "characteristics": self._extract_problem_characteristics(question_text)
                }
            
            # Update interaction count
            self.problem_data[problem_id]["interaction_count"] += 1
            self.problem_data[problem_id]["last_interaction"] = timestamp
            
            # Save data periodically
            if len(self.calibration_data["chat"]) % 10 == 0:
                self._save_data()
                
            logger.debug(f"Recorded chat interaction for problem {problem_id}")
    
    def get_calibration_summary(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of calibration performance.
        
        Args:
            component_type: Optional component type to get summary for
            
        Returns:
            Dictionary of calibration summary metrics
        """
        with self.lock:
            result = {}
            
            if component_type and component_type in self.calibration_data:
                result[component_type] = self._calculate_summary(component_type)
            else:
                # Calculate for all component types
                for comp_type in self.calibration_data:
                    result[comp_type] = self._calculate_summary(comp_type)
            
            return result
    
    def get_training_data(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get training data for machine learning models.
        
        Args:
            component_type: Optional component type to get data for
            
        Returns:
            Dictionary with features and labels for training
        """
        with self.lock:
            result = {}
            
            if component_type and component_type in self.calibration_data:
                result[component_type] = self._prepare_training_data(component_type)
            else:
                # Prepare for all component types
                for comp_type in self.calibration_data:
                    result[comp_type] = self._prepare_training_data(comp_type)
            
            return result
    
    def export_data(self, export_dir: Optional[str] = None) -> str:
        """
        Export all calibration data to JSON files.
        
        Args:
            export_dir: Directory to export to (defaults to data_dir/exports)
            
        Returns:
            Path to the export directory
        """
        with self.lock:
            # Set up export directory
            if export_dir:
                export_path = export_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = os.path.join(self.data_dir, "exports", timestamp)
            
            os.makedirs(export_path, exist_ok=True)
            
            # Export calibration data
            for component_type, data in self.calibration_data.items():
                file_path = os.path.join(export_path, f"{component_type}_calibration.json")
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Export problem data
            problem_file_path = os.path.join(export_path, "problems.json")
            with open(problem_file_path, 'w') as f:
                json.dump(self.problem_data, f, indent=2)
            
            logger.info(f"Exported calibration data to {export_path}")
            return export_path
    
    def _save_data(self) -> None:
        """Save calibration data to disk."""
        try:
            # Save calibration data
            calibration_path = os.path.join(self.data_dir, "calibration_data.json")
            with open(calibration_path, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
            
            # Save problem data
            problem_path = os.path.join(self.data_dir, "problem_data.json")
            with open(problem_path, 'w') as f:
                json.dump(self.problem_data, f, indent=2)
            
            logger.debug("Saved calibration data")
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
    
    def _load_data(self) -> None:
        """Load calibration data from disk if available."""
        try:
            # Load calibration data
            calibration_path = os.path.join(self.data_dir, "calibration_data.json")
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r') as f:
                    self.calibration_data = json.load(f)
            
            # Load problem data
            problem_path = os.path.join(self.data_dir, "problem_data.json")
            if os.path.exists(problem_path):
                with open(problem_path, 'r') as f:
                    self.problem_data = json.load(f)
            
            logger.info("Loaded existing calibration data")
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
    
    def _calculate_summary(self, component_type: str) -> Dict[str, Any]:
        """
        Calculate summary metrics for a component type.
        
        Args:
            component_type: The component type
            
        Returns:
            Dictionary of summary metrics
        """
        records = self.calibration_data.get(component_type, [])
        
        if not records:
            return {"error": "No calibration data available"}
        
        # Extract predictions and actuals
        predictions = [r.get("predicted_confidence", 0) for r in records]
        actuals = [r.get("actual_outcome", 0) for r in records]
        
        # Calculate summary metrics
        metrics = {
            "count": len(records),
            "mean_prediction": statistics.mean(predictions),
            "mean_actual": statistics.mean(actuals),
            "min_prediction": min(predictions),
            "max_prediction": max(predictions),
        }
        
        # Calculate calibration error
        if len(predictions) > 1:
            # Mean squared error
            mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
            metrics["calibration_mse"] = mse
            
            # Calculate correlation
            try:
                correlation = statistics.correlation(predictions, actuals)
                metrics["correlation"] = correlation
            except:
                metrics["correlation"] = 0.0
        
        # Calculate calibration by confidence buckets
        buckets = {}
        for p, a in zip(predictions, actuals):
            bucket = round(p * 10) / 10  # Round to nearest 0.1
            if bucket not in buckets:
                buckets[bucket] = {"count": 0, "correct": 0}
            buckets[bucket]["count"] += 1
            buckets[bucket]["correct"] += a
        
        # Calculate accuracy per bucket
        for bucket, data in buckets.items():
            data["accuracy"] = data["correct"] / data["count"]
            data["calibration_error"] = abs(bucket - data["accuracy"])
        
        metrics["buckets"] = buckets
        
        # Recent trend (last 50 interactions)
        if len(records) >= 50:
            recent_records = records[-50:]
            recent_predictions = [r.get("predicted_confidence", 0) for r in recent_records]
            recent_actuals = [r.get("actual_outcome", 0) for r in recent_records]
            
            metrics["recent"] = {
                "mean_prediction": statistics.mean(recent_predictions),
                "mean_actual": statistics.mean(recent_actuals),
                "calibration_mse": sum((p - a) ** 2 for p, a in zip(recent_predictions, recent_actuals)) / len(recent_predictions)
            }
        
        return metrics
    
    def _prepare_training_data(self, component_type: str) -> Dict[str, Any]:
        """
        Prepare training data for machine learning models.
        
        Args:
            component_type: The component type
            
        Returns:
            Dictionary with features and labels
        """
        records = self.calibration_data.get(component_type, [])
        
        if not records:
            return {"error": "No training data available"}
        
        features = []
        labels = []
        
        for record in records:
            # Skip if missing problem_id
            if "problem_id" not in record:
                continue
                
            # Get problem characteristics
            problem_id = record["problem_id"]
            if problem_id in self.problem_data:
                problem_chars = self.problem_data[problem_id].get("characteristics", {})
                
                # Combine problem characteristics with record-specific data
                feature_vector = {
                    # Problem characteristics
                    "complexity": problem_chars.get("complexity", 0.5),
                    "symbol_count": problem_chars.get("symbol_count", 0),
                    "keyword_count": problem_chars.get("keyword_count", 0),
                    "length": problem_chars.get("length", 0),
                    
                    # Interaction-specific features
                    "predicted_confidence": record.get("predicted_confidence", 0.5)
                }
                
                # Add component-specific features
                if component_type == "hints":
                    feature_vector["hint_number"] = record.get("hint_number", 1)
                
                features.append(feature_vector)
                labels.append(record.get("actual_outcome", 0))
        
        return {
            "features": features,
            "labels": labels,
            "count": len(features)
        }
    
    def _extract_problem_characteristics(self, question_text: str) -> Dict[str, Any]:
        """
        Extract characteristics from a question for feature engineering.
        
        Args:
            question_text: The question text
            
        Returns:
            Dictionary of problem characteristics
        """
        import re
        
        # 1. Length-based complexity
        length = len(question_text)
        length_norm = min(1.0, length / 300)  # Normalize to 0-1
        
        # 2. Keyword-based complexity
        # Advanced math keywords
        advanced_keywords = [
            "calculus", "derivative", "integral", "differential", "equations", "theorem",
            "prove", "proof", "vector", "matrix", "matrices", "optimization", "probability",
            "statistics", "hypothesis", "logarithm", "exponential", "trigonometric"
        ]
        
        # Medium complexity keywords
        medium_keywords = [
            "algebra", "function", "equation", "solve", "simplify", "factor", "evaluate",
            "graph", "system", "polynomial", "quadratic", "fraction", "percentage", "ratio"
        ]
        
        # Count keyword occurrences
        advanced_count = sum(1 for keyword in advanced_keywords if keyword.lower() in question_text.lower())
        medium_count = sum(1 for keyword in medium_keywords if keyword.lower() in question_text.lower())
        
        # 3. Symbol-based complexity
        symbols = ["∫", "∂", "Σ", "π", "θ", "√", "∞", "≠", "≤", "≥", "±", "→", "∈", "∀", "∃"]
        symbol_count = sum(1 for symbol in symbols if symbol in question_text)
        
        # 4. Formula complexity
        formula_pattern = r"[a-zA-Z]+\s*[=+\-*/^]\s*[a-zA-Z0-9]+"
        formula_count = len(re.findall(formula_pattern, question_text))
        
        # 5. Overall complexity score
        complexity_score = (
            0.3 * length_norm +
            0.4 * (min(1.0, (advanced_count * 0.2) + (medium_count * 0.1))) +
            0.15 * min(1.0, symbol_count * 0.25) +
            0.15 * min(1.0, formula_count * 0.15)
        )
        
        return {
            "length": length,
            "advanced_keyword_count": advanced_count,
            "medium_keyword_count": medium_count,
            "keyword_count": advanced_count + medium_count,
            "symbol_count": symbol_count,
            "formula_count": formula_count,
            "complexity": min(1.0, complexity_score)
        }

    def _assess_question_complexity(self, question: str) -> float:
        # Hardcoded complexity indicators
        advanced_keywords = ["integral", "derivative", "limit", ...]

        # 1. Length-based complexity
        length = len(question)
        length_norm = min(1.0, length / 300)  # Normalize to 0-1
        
        # 2. Keyword-based complexity
        # Advanced math keywords
        advanced_keywords = [
            "calculus", "derivative", "integral", "differential", "equations", "theorem",
            "prove", "proof", "vector", "matrix", "matrices", "optimization", "probability",
            "statistics", "hypothesis", "logarithm", "exponential", "trigonometric"
        ]
        
        # Medium complexity keywords
        medium_keywords = [
            "algebra", "function", "equation", "solve", "simplify", "factor", "evaluate",
            "graph", "system", "polynomial", "quadratic", "fraction", "percentage", "ratio"
        ]
        
        # Count keyword occurrences
        advanced_count = sum(1 for keyword in advanced_keywords if keyword.lower() in question.lower())
        medium_count = sum(1 for keyword in medium_keywords if keyword.lower() in question.lower())
        
        # 3. Symbol-based complexity
        symbols = ["∫", "∂", "Σ", "π", "θ", "√", "∞", "≠", "≤", "≥", "±", "→", "∈", "∀", "∃"]
        symbol_count = sum(1 for symbol in symbols if symbol in question)
        
        # 4. Formula complexity
        formula_pattern = r"[a-zA-Z]+\s*[=+\-*/^]\s*[a-zA-Z0-9]+"
        formula_count = len(re.findall(formula_pattern, question))
        
        # 5. Overall complexity score
        complexity_score = (
            0.3 * length_norm +
            0.4 * (min(1.0, (advanced_count * 0.2) + (medium_count * 0.1))) +
            0.15 * min(1.0, symbol_count * 0.25) +
            0.15 * min(1.0, formula_count * 0.15)
        )
        
        return min(1.0, complexity_score) 