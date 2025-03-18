"""
Confidence Assessment Module.

This module provides advanced confidence assessment capabilities for math services,
allowing for more accurate estimation of confidence in various operations.
"""

import logging
import re
import math
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import statistics
from enum import Enum

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Standardized confidence levels."""
    VERY_LOW = "very_low"     # 0.0-0.2: Highly uncertain
    LOW = "low"               # 0.2-0.4: Significant uncertainty
    MEDIUM = "medium"         # 0.4-0.6: Moderate confidence
    HIGH = "high"             # 0.6-0.8: Good confidence
    VERY_HIGH = "very_high"   # 0.8-1.0: Strong confidence

class ConfidenceMetrics:
    """Handles confidence assessment and calibration for math services."""
    
    def __init__(self, calibration_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the confidence metrics system.
        
        Args:
            calibration_data: Optional pre-existing calibration data
        """
        # Historical calibration data - maps confidence predictions to actual outcomes
        self.calibration_data = calibration_data or {
            "feedback": {"predictions": [], "actuals": []},
            "hints": {"predictions": [], "actuals": []},
            "analysis": {"predictions": [], "actuals": []},
            "chat": {"predictions": [], "actuals": []}
        }
        
        # Feature weights for different confidence components
        self.feature_weights = {
            # Weights for feedback confidence
            "feedback": {
                "verification_result": 0.5,    # Weight for verification results
                "answer_proximity": 0.2,       # Weight for how close student answer is to correct
                "question_complexity": 0.15,   # Weight for problem complexity
                "model_uncertainty": 0.15,     # Weight for model's reported uncertainty
            },
            
            # Weights for hint confidence
            "hints": {
                "hint_specificity": 0.3,       # More specific hints have lower confidence
                "hint_count": 0.2,             # More hints = lower confidence in later hints
                "verification_result": 0.3,    # Weight for verification results
                "question_complexity": 0.2,    # Weight for problem complexity
            },
            
            # Weights for analysis confidence
            "analysis": {
                "explanation_coherence": 0.25, # How coherent/structured is the explanation
                "verification_result": 0.4,    # Weight for verification results
                "consistency_check": 0.2,      # Consistency of analysis with solution
                "question_complexity": 0.15,   # Weight for problem complexity
            },
            
            # Weights for chat response confidence
            "chat": {
                "context_relevance": 0.3,      # How relevant is the response to context
                "answer_clarity": 0.2,         # How clear/precise is the answer
                "question_specificity": 0.3,   # How specific is the question
                "conversation_history": 0.2,   # How much context from history
            }
        }
        
        logger.info("Initialized ConfidenceMetrics system")
    
    def get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """
        Convert a numerical confidence score to a standardized level.
        
        Args:
            confidence_score: Numerical confidence score (0.0-1.0)
            
        Returns:
            Standardized confidence level
        """
        if confidence_score < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif confidence_score < 0.4:
            return ConfidenceLevel.LOW
        elif confidence_score < 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def assess_feedback_confidence(self, 
                                 state: Dict[str, Any], 
                                 verification_result: Optional[Dict[str, Any]] = None,
                                 model_uncertainty: Optional[float] = None) -> float:
        """
        Assess confidence in feedback given to the student.
        
        Args:
            state: The current problem state
            verification_result: Optional verification result from meta agent
            model_uncertainty: Optional model uncertainty estimate
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Extract features for confidence estimation
        question_complexity = self._assess_question_complexity(state.get("question", ""))
        
        # Get verification confidence if available
        verification_confidence = 0.0
        if verification_result and isinstance(verification_result, dict):
            verification_confidence = verification_result.get("confidence", 0.0)
            if isinstance(verification_confidence, (int, float)) and verification_confidence > 1.0:
                verification_confidence /= 100.0  # Normalize to 0-1 if given as percentage
        
        # Get proximity score if available, with safe handling for None
        proximity_value = state.get("proximity_score", None)
        proximity_score = 0.5  # Default middle value
        if proximity_value is not None:
            proximity_score = float(proximity_value) / 10.0  # Normalize to 0-1
        
        # Use model's uncertainty estimate if provided
        model_uncertainty_score = model_uncertainty if model_uncertainty is not None else 0.7
        
        # Apply feature weights
        weights = self.feature_weights.get("feedback", {})
        
        weighted_sum = (
            weights.get("verification_result", 0.5) * verification_confidence +
            weights.get("answer_proximity", 0.2) * proximity_score +
            weights.get("question_complexity", 0.15) * (1.0 - question_complexity) +  # Inverse of complexity
            weights.get("model_uncertainty", 0.15) * model_uncertainty_score
        )
        
        # Normalize the final score to 0-1
        total_weight = sum(weights.values())
        if total_weight > 0:
            confidence = weighted_sum / total_weight
        else:
            confidence = 0.5  # Default to middle confidence if no weights
        
        return max(0.0, min(1.0, confidence))  # Ensure in range 0-1
    
    def assess_hint_confidence(self,
                              state: Dict[str, Any],
                              hint: str,
                              hint_number: int,
                              verification_result: Optional[Dict[str, Any]] = None) -> float:
        """
        Assess confidence in a hint provided to the student.
        
        Args:
            state: The current problem state
            hint: The hint text
            hint_number: Which hint this is (1, 2, 3, etc.)
            verification_result: Optional verification result from meta agent
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Extract features for confidence estimation
        question_complexity = self._assess_question_complexity(state.get("question", ""))
        hint_specificity = self._assess_hint_specificity(hint)
        
        # Progressive hints have decreasing confidence
        # First hint: high confidence, later hints: lower confidence
        hint_position_factor = max(0.0, 1.0 - (hint_number - 1) * 0.15)
        
        # Get verification confidence if available
        verification_confidence = 0.0
        if verification_result and isinstance(verification_result, dict):
            verification_confidence = verification_result.get("confidence", 0.0)
            if isinstance(verification_confidence, (int, float)) and verification_confidence > 1.0:
                verification_confidence /= 100.0  # Normalize to 0-1 if given as percentage
        
        # Apply feature weights
        weights = self.feature_weights.get("hints", {})
        
        weighted_sum = (
            weights.get("verification_result", 0.4) * verification_confidence +
            weights.get("hint_specificity", 0.3) * hint_specificity +
            weights.get("hint_position", 0.2) * hint_position_factor +
            weights.get("question_complexity", 0.1) * (1.0 - question_complexity)  # Inverse of complexity
        )
        
        # Normalize the final score to 0-1
        total_weight = sum(weights.values())
        if total_weight > 0:
            confidence = weighted_sum / total_weight
        else:
            confidence = 0.5  # Default to middle confidence if no weights
        
        return max(0.0, min(1.0, confidence))  # Ensure in range 0-1
    
    def assess_analysis_confidence(self,
                                  state: Dict[str, Any],
                                  analysis_result: Dict[str, Any],
                                  verification_result: Optional[Dict[str, Any]] = None) -> float:
        """
        Assess confidence in analysis of student's calculation.
        
        Args:
            state: The current problem state
            analysis_result: Analysis results for the calculation
            verification_result: Optional verification result from meta agent
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Extract features for confidence estimation
        question_complexity = self._assess_question_complexity(state.get("question", ""))
        
        # Get verification confidence if available
        verification_confidence = 0.0
        if verification_result and isinstance(verification_result, dict):
            verification_confidence = verification_result.get("confidence", 0.0)
            if isinstance(verification_confidence, (int, float)) and verification_confidence > 1.0:
                verification_confidence /= 100.0  # Normalize to 0-1 if given as percentage
        
        # Assess coherence of analysis explanation
        explanation_coherence = self._assess_explanation_coherence(analysis_result)
        
        # Apply feature weights
        weights = self.feature_weights.get("analysis", {})
        
        weighted_sum = (
            weights.get("verification_result", 0.6) * verification_confidence +
            weights.get("explanation_coherence", 0.25) * explanation_coherence +
            weights.get("question_complexity", 0.15) * (1.0 - question_complexity)  # Inverse of complexity
        )
        
        # Normalize the final score to 0-1
        total_weight = sum(weights.values())
        if total_weight > 0:
            confidence = weighted_sum / total_weight
        else:
            confidence = 0.5  # Default to middle confidence if no weights
        
        return max(0.0, min(1.0, confidence))  # Ensure in range 0-1
    
    def assess_chat_confidence(self,
                              state: Dict[str, Any],
                              follow_up_question: str,
                              response: str) -> float:
        """
        Assess confidence in a chat response to a follow-up question.
        
        Args:
            state: The current problem state
            follow_up_question: The student's follow-up question
            response: The response provided
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Extract features for confidence estimation
        question_complexity = self._assess_question_complexity(state.get("question", ""))
        question_specificity = self._assess_question_specificity(follow_up_question)
        answer_clarity = self._assess_answer_clarity(response)
        context_relevance = self._assess_context_relevance(state, follow_up_question, response)
        
        # Apply feature weights
        weights = self.feature_weights.get("chat", {})
        
        weighted_sum = (
            weights.get("answer_clarity", 0.35) * answer_clarity +
            weights.get("context_relevance", 0.35) * context_relevance +
            weights.get("question_specificity", 0.2) * question_specificity +
            weights.get("question_complexity", 0.1) * (1.0 - question_complexity)  # Inverse of complexity
        )
        
        # Normalize the final score to 0-1
        total_weight = sum(weights.values())
        if total_weight > 0:
            confidence = weighted_sum / total_weight
        else:
            confidence = 0.5  # Default to middle confidence if no weights
        
        return max(0.0, min(1.0, confidence))  # Ensure in range 0-1
    
    def calibrate_confidence(self, 
                           prediction: float, 
                           actual: bool, 
                           component_type: str) -> None:
        """
        Update calibration data with a new prediction/actual pair.
        
        Args:
            prediction: The confidence prediction (0.0-1.0)
            actual: Whether the prediction was actually correct
            component_type: The type of component (feedback, hints, analysis, chat)
        """
        if component_type not in self.calibration_data:
            self.calibration_data[component_type] = {"predictions": [], "actuals": []}
            
        self.calibration_data[component_type]["predictions"].append(prediction)
        self.calibration_data[component_type]["actuals"].append(1.0 if actual else 0.0)
        
        logger.info(f"Added calibration data for {component_type}: prediction={prediction:.2f}, actual={actual}")
    
    def get_calibration_metrics(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get calibration metrics for the specified component type or all components.
        
        Args:
            component_type: Optional specific component type to get metrics for
            
        Returns:
            Dictionary of calibration metrics
        """
        result = {}
        
        if component_type and component_type in self.calibration_data:
            result[component_type] = self._calculate_calibration_metrics(component_type)
        else:
            # Calculate for all component types
            for comp_type in self.calibration_data:
                result[comp_type] = self._calculate_calibration_metrics(comp_type)
        
        return result
    
    def _calculate_calibration_metrics(self, component_type: str) -> Dict[str, Any]:
        """
        Calculate calibration metrics for a component type.
        
        Args:
            component_type: The component type to calculate metrics for
            
        Returns:
            Dictionary of calibration metrics
        """
        data = self.calibration_data[component_type]
        predictions = data["predictions"]
        actuals = data["actuals"]
        
        if not predictions:
            return {"error": "No calibration data available"}
        
        # Calculate calibration metrics
        metrics = {
            "count": len(predictions),
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
            bucket = int(p * 10) / 10  # Round to nearest 0.1
            if bucket not in buckets:
                buckets[bucket] = {"count": 0, "correct": 0}
            buckets[bucket]["count"] += 1
            buckets[bucket]["correct"] += a
        
        # Calculate accuracy per bucket
        for bucket, data in buckets.items():
            data["accuracy"] = data["correct"] / data["count"]
            data["calibration_error"] = abs(bucket - data["accuracy"])
        
        metrics["buckets"] = buckets
        
        return metrics
    
    def adjust_weights(self, component_type: str, new_weights: Dict[str, float]) -> None:
        """
        Adjust feature weights for a component type.
        
        Args:
            component_type: The component type to adjust weights for
            new_weights: Dictionary of new weights
        """
        if component_type not in self.feature_weights:
            logger.error(f"Unknown component type: {component_type}")
            return
            
        # Validate weights
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights do not sum to 1.0: {total}")
            # Normalize weights
            for k in new_weights:
                new_weights[k] /= total
        
        # Update weights
        for k, v in new_weights.items():
            if k in self.feature_weights[component_type]:
                self.feature_weights[component_type][k] = v
            else:
                logger.warning(f"Unknown weight component: {k}")
        
        logger.info(f"Updated weights for {component_type}: {self.feature_weights[component_type]}")
    
    def _assess_question_complexity(self, question: str) -> float:
        """
        Assess the complexity of a math question.
        
        Args:
            question: The question text
            
        Returns:
            Complexity score from 0.0 (simple) to 1.0 (complex)
        """
        # Simple complexity heuristics
        
        # 1. Length-based complexity: longer questions tend to be more complex
        length_score = min(1.0, len(question) / 300)  # Normalize, cap at 1.0
        
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
        
        # Calculate keyword complexity score
        keyword_score = min(1.0, (advanced_count * 0.2) + (medium_count * 0.1))
        
        # 3. Symbol-based complexity: more symbols = more complex
        symbols = ["∫", "∂", "Σ", "π", "θ", "√", "∞", "≠", "≤", "≥", "±", "→", "∈", "∀", "∃"]
        symbol_count = sum(1 for symbol in symbols if symbol in question)
        symbol_score = min(1.0, symbol_count * 0.25)
        
        # 4. Formula complexity (detected by pattern of variables and operations)
        formula_pattern = r"[a-zA-Z]+\s*[=+\-*/^]\s*[a-zA-Z0-9]+"
        formula_count = len(re.findall(formula_pattern, question))
        formula_score = min(1.0, formula_count * 0.15)
        
        # Combine scores with weights
        complexity_score = (
            0.3 * length_score +
            0.4 * keyword_score +
            0.15 * symbol_score +
            0.15 * formula_score
        )
        
        return min(1.0, complexity_score)
    
    def _assess_hint_specificity(self, hint: str) -> float:
        """
        Assess how specific a hint is (more specific hints should have lower confidence).
        
        Args:
            hint: The hint text
            
        Returns:
            Specificity score from 0.0 (general) to 1.0 (very specific)
        """
        # 1. Length-based specificity: longer hints tend to be more specific
        length_score = min(1.0, len(hint) / 150)  # Normalize, cap at 1.0
        
        # 2. Keyword-based specificity
        specific_keywords = [
            "exact", "precisely", "specifically", "formula", "equation", "step", "value",
            "calculate", "compute", "plug", "substitute", "solve", "exactly", "directly"
        ]
        
        # Count keyword occurrences
        specific_count = sum(1 for keyword in specific_keywords if keyword.lower() in hint.lower())
        keyword_score = min(1.0, specific_count * 0.15)
        
        # 3. Number presence: hints with numbers are more specific
        number_pattern = r'\b\d+\.?\d*\b'
        number_count = len(re.findall(number_pattern, hint))
        number_score = min(1.0, number_count * 0.2)
        
        # 4. Question format: hints phrased as questions are less specific
        question_pattern = r'\?'
        question_count = len(re.findall(question_pattern, hint))
        question_score = max(0.0, 1.0 - (question_count * 0.3))  # More questions = less specific
        
        # Combine scores with weights
        specificity_score = (
            0.3 * length_score +
            0.3 * keyword_score +
            0.25 * number_score +
            0.15 * question_score
        )
        
        return min(1.0, specificity_score)
    
    def _assess_explanation_coherence(self, analysis_result: Dict[str, Any]) -> float:
        """
        Assess the coherence of an explanation.
        
        Args:
            analysis_result: The analysis result
            
        Returns:
            Coherence score from 0.0 (incoherent) to 1.0 (very coherent)
        """
        # Extract explanation text
        explanation = ""
        if "explanation" in analysis_result:
            explanation = analysis_result["explanation"]
        elif "details" in analysis_result:
            explanation = analysis_result["details"]
        elif "analysis" in analysis_result:
            explanation = analysis_result["analysis"]
        
        # Short explanations are penalized
        if len(explanation) < 50:
            return 0.5
        
        # 1. Structure indicators
        structure_keywords = [
            "first", "second", "third", "next", "then", "finally", "step", "because",
            "therefore", "thus", "however", "instead", "alternatively", "consequently"
        ]
        structure_count = sum(1 for keyword in structure_keywords if keyword.lower() in explanation.lower())
        structure_score = min(1.0, structure_count * 0.15)
        
        # 2. Paragraph structure
        paragraphs = explanation.split("\n\n")
        paragraph_score = min(1.0, len(paragraphs) * 0.2)
        
        # 3. Sentence flow (approximate by looking at conjunctions)
        conjunctions = ["and", "but", "or", "so", "because", "however", "therefore"]
        conjunction_count = sum(1 for conj in conjunctions if f" {conj} " in explanation.lower())
        flow_score = min(1.0, conjunction_count * 0.1)
        
        # 4. Math terms - more math terms indicates better coherence for math explanations
        math_terms = [
            "equation", "formula", "value", "calculate", "solve", "result", "solution",
            "step", "method", "approach", "simplify", "answer", "problem"
        ]
        math_term_count = sum(1 for term in math_terms if term.lower() in explanation.lower())
        math_score = min(1.0, math_term_count * 0.1)
        
        # Combine scores with weights
        coherence_score = (
            0.4 * structure_score +
            0.2 * paragraph_score +
            0.2 * flow_score +
            0.2 * math_score
        )
        
        return min(1.0, coherence_score)
    
    def _assess_context_relevance(self, state: Dict[str, Any], question: str, response: str) -> float:
        """
        Assess the relevance of a chat response to the context.
        
        Args:
            state: The current problem state
            question: The follow-up question
            response: The generated response
            
        Returns:
            Relevance score from 0.0 (irrelevant) to 1.0 (very relevant)
        """
        # Extract key context elements
        problem = state.get("question", "")
        student_answer = state.get("student_answer", "")
        
        # 1. Context keywords from problem
        # Extract important words from the problem (nouns, numbers, etc.)
        problem_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', problem.lower()))
        problem_numbers = set(re.findall(r'\b\d+\.?\d*\b', problem))
        
        # Count occurrences in response
        problem_word_matches = sum(1 for word in problem_words if word in response.lower())
        problem_number_matches = sum(1 for num in problem_numbers if num in response)
        
        problem_score = min(1.0, (
            (problem_word_matches / max(1, len(problem_words)) * 0.6) +
            (problem_number_matches / max(1, len(problem_numbers)) * 0.4)
        ))
        
        # 2. Question keywords
        question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
        question_matches = sum(1 for word in question_words if word in response.lower())
        question_score = min(1.0, question_matches / max(1, len(question_words)))
        
        # 3. Answer reference
        answer_reference_score = 0.0
        if student_answer and len(student_answer) > 0:
            if student_answer.lower() in response.lower():
                answer_reference_score = 0.8
            else:
                # Check for partial references
                answer_parts = re.findall(r'\b[a-zA-Z0-9]{2,}\b', student_answer.lower())
                matches = sum(1 for part in answer_parts if part in response.lower())
                answer_reference_score = min(0.8, matches / max(1, len(answer_parts)))
        
        # Combine scores with weights
        relevance_score = (
            0.4 * problem_score +
            0.5 * question_score +
            0.1 * answer_reference_score
        )
        
        return min(1.0, relevance_score)
    
    def _assess_answer_clarity(self, response: str) -> float:
        """
        Assess the clarity of a response.
        
        Args:
            response: The generated response
            
        Returns:
            Clarity score from 0.0 (unclear) to 1.0 (very clear)
        """
        # 1. Sentence structure
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Penalize very long sentences
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        sentence_score = max(0.0, min(1.0, 2.0 - (avg_sentence_length / 20)))
        
        # 2. Explanation markers
        explanation_markers = [
            "because", "therefore", "thus", "for example", "such as", "specifically",
            "in other words", "to clarify", "this means", "to illustrate", "in essence"
        ]
        marker_count = sum(1 for marker in explanation_markers if marker.lower() in response.lower())
        marker_score = min(1.0, marker_count * 0.2)
        
        # 3. Paragraph structure
        paragraphs = response.split("\n\n")
        paragraph_score = min(1.0, len(paragraphs) * 0.25)
        
        # 4. Specificity (numbers, formulas, specific terms)
        specificity_patterns = [
            r'\b\d+\.?\d*\b',  # Numbers
            r'[a-zA-Z]+\s*[=+\-*/^]\s*[a-zA-Z0-9]+',  # Formulas
            r'\b[A-Z][a-z]+\'s\s+[Tt]heorem\b',  # Named theorems
        ]
        
        specificity_count = sum(len(re.findall(pattern, response)) for pattern in specificity_patterns)
        specificity_score = min(1.0, specificity_count * 0.15)
        
        # Combine scores with weights
        clarity_score = (
            0.3 * sentence_score +
            0.3 * marker_score +
            0.2 * paragraph_score +
            0.2 * specificity_score
        )
        
        return min(1.0, clarity_score)
    
    def _assess_question_specificity(self, question: str) -> float:
        """
        Assess the specificity of a follow-up question.
        
        Args:
            question: The follow-up question
            
        Returns:
            Specificity score from 0.0 (vague) to 1.0 (very specific)
        """
        # 1. Question length (longer questions tend to be more specific)
        words = question.split()
        length_score = min(1.0, len(words) / 15)  # Normalize, cap at 1.0
        
        # 2. Specific question markers
        specific_markers = [
            "specifically", "exactly", "precisely", "steps?", "how do", "why does",
            "explain", "clarify", "what is the", "where did", "when would"
        ]
        marker_count = sum(1 for marker in specific_markers if re.search(r'\b' + marker + r'\b', question.lower()))
        marker_score = min(1.0, marker_count * 0.25)
        
        # 3. Numbers and symbols (more specific questions often contain these)
        number_pattern = r'\b\d+\.?\d*\b'
        number_count = len(re.findall(number_pattern, question))
        
        symbol_pattern = r'[+\-*/=<>]'
        symbol_count = len(re.findall(symbol_pattern, question))
        
        technical_score = min(1.0, (number_count + symbol_count) * 0.2)
        
        # 4. Vague question patterns - reduce score for vague questions
        vague_patterns = [
            r'\bhelp\b',
            r'\bexplain more\b',
            r'\bdon\'t understand\b',
            r'\bconfused\b',
            r'\bwhat do you mean\b'
        ]
        
        vague_count = sum(len(re.findall(pattern, question.lower())) for pattern in vague_patterns)
        vague_penalty = max(0.0, min(0.5, vague_count * 0.25))
        
        # Combine scores with weights and apply vague penalty
        specificity_score = (
            0.4 * length_score +
            0.4 * marker_score +
            0.2 * technical_score
        ) * (1.0 - vague_penalty)
        
        return min(1.0, specificity_score)
    
    def save_calibration_data(self, file_path: str) -> None:
        """
        Save calibration data to a file.
        
        Args:
            file_path: Path to save the data to
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
            logger.info(f"Saved calibration data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
    
    def load_calibration_data(self, file_path: str) -> bool:
        """
        Load calibration data from a file.
        
        Args:
            file_path: Path to load the data from
            
        Returns:
            Whether the load was successful
        """
        try:
            with open(file_path, 'r') as f:
                self.calibration_data = json.load(f)
            logger.info(f"Loaded calibration data from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return False
            
    def train_simple_model(self) -> Dict[str, Any]:
        """
        Train a simple regression model for confidence calibration.
        This is a basic implementation that could be replaced with a more sophisticated ML model.
        
        Returns:
            Dictionary of model parameters by component type
        """
        models = {}
        
        for component_type, data in self.calibration_data.items():
            predictions = data["predictions"]
            actuals = data["actuals"]
            
            if len(predictions) < 10:
                # Not enough data to train
                logger.warning(f"Not enough data to train model for {component_type}")
                models[component_type] = {"bias": 0.0, "scale": 1.0}
                continue
            
            # Simple linear calibration: y = ax + b
            # We want to find a and b such that a*prediction + b better matches the actual values
            
            # First compute means
            mean_pred = statistics.mean(predictions)
            mean_actual = statistics.mean(actuals)
            
            # Then compute scale (a) using covariance and variance
            numerator = sum((p - mean_pred) * (a - mean_actual) for p, a in zip(predictions, actuals))
            denominator = sum((p - mean_pred) ** 2 for p in predictions)
            
            if denominator == 0:
                # Avoid division by zero
                scale = 1.0
            else:
                scale = numerator / denominator
            
            # Compute bias (b) using means and scale
            bias = mean_actual - (scale * mean_pred)
            
            models[component_type] = {"bias": bias, "scale": scale}
            logger.info(f"Trained model for {component_type}: bias={bias:.3f}, scale={scale:.3f}")
        
        return models
    
    def apply_calibration_model(self, 
                               raw_confidence: float, 
                               component_type: str, 
                               model_params: Optional[Dict[str, float]] = None) -> float:
        """
        Apply a calibration model to adjust a raw confidence score.
        
        Args:
            raw_confidence: The uncalibrated confidence score
            component_type: The type of component
            model_params: Optional model parameters to use (uses trained ones if None)
            
        Returns:
            Calibrated confidence score
        """
        if model_params is None:
            # Train a model using current calibration data
            model_params = self.train_simple_model().get(component_type, {"bias": 0.0, "scale": 1.0})
        
        # Apply linear calibration: calibrated = scale * raw + bias
        bias = model_params.get("bias", 0.0)
        scale = model_params.get("scale", 1.0)
        
        calibrated = (scale * raw_confidence) + bias
        
        # Ensure the result is between 0 and 1
        calibrated = max(0.0, min(1.0, calibrated))
        
        logger.debug(f"Calibrated {component_type} confidence: {raw_confidence:.3f} -> {calibrated:.3f}")
        return calibrated 