#!/usr/bin/env python3
"""
Advanced Confidence Metrics Demonstration.

This script demonstrates the advanced confidence assessment features of the math services,
including real confidence metrics, ML-based confidence prediction, and calibration.
"""

import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, Any, List
import random
from datetime import datetime

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("confidence_demo")

# Import the necessary modules
from app.math_services.metrics import (
    ConfidenceManager,
    ConfidenceLevel
)

def create_simulated_problem_state(problem_type: str = "algebra") -> Dict[str, Any]:
    """
    Create a simulated problem state for demonstration.
    
    Args:
        problem_type: Type of math problem
        
    Returns:
        Dictionary with simulated problem state
    """
    # Define problem templates
    templates = {
        "algebra": {
            "question": "Solve for x: {coefficient}x + {term} = {result}",
            "answer": "{solution}",
            "complexity": 0.4
        },
        "calculus": {
            "question": "Find the derivative of f(x) = {expression} with respect to x.",
            "answer": "{solution}",
            "complexity": 0.7
        },
        "statistics": {
            "question": "The mean of a normal distribution is μ = {mean} and the standard deviation is σ = {std_dev}. What is the probability that a randomly selected observation is less than {threshold}?",
            "answer": "{solution}",
            "complexity": 0.8
        }
    }
    
    # Get template
    template = templates.get(problem_type, templates["algebra"])
    
    # Generate parameters
    if problem_type == "algebra":
        coefficient = random.randint(2, 10)
        term = random.randint(-20, 20)
        solution = random.randint(-10, 10)
        result = coefficient * solution + term
        
        question = template["question"].format(
            coefficient=coefficient,
            term=term,
            result=result
        )
        correct_answer = str(solution)
        
    elif problem_type == "calculus":
        expressions = [
            "3x^2 + 2x - 5",
            "sin(x) + cos(x)",
            "e^x * ln(x)",
            "x^3 - 4x^2 + 7x - 2"
        ]
        solutions = [
            "6x + 2",
            "cos(x) - sin(x)",
            "e^x * (1/x + ln(x))",
            "3x^2 - 8x + 7"
        ]
        
        idx = random.randint(0, len(expressions) - 1)
        expression = expressions[idx]
        solution = solutions[idx]
        
        question = template["question"].format(expression=expression)
        correct_answer = solution
        
    elif problem_type == "statistics":
        mean = random.randint(50, 150)
        std_dev = random.randint(5, 20)
        threshold = mean + (random.choice([-2, -1, 0, 1, 2]) * std_dev)
        
        # Simplify for demo - we don't need exact solutions
        if threshold < mean:
            solution = f"Less than 0.5, approximately {random.uniform(0.02, 0.48):.4f}"
        elif threshold == mean:
            solution = "Exactly 0.5"
        else:
            solution = f"Greater than 0.5, approximately {random.uniform(0.52, 0.98):.4f}"
        
        question = template["question"].format(
            mean=mean,
            std_dev=std_dev,
            threshold=threshold
        )
        correct_answer = solution
    
    # Generate a simulated student answer (sometimes correct, sometimes close, sometimes wrong)
    answer_type = random.choices(
        ["correct", "close", "wrong"],
        weights=[0.3, 0.4, 0.3]
    )[0]
    
    if answer_type == "correct":
        student_answer = correct_answer
        proximity_score = 10.0
        is_correct = True
    elif answer_type == "close":
        if problem_type == "algebra":
            student_answer = str(int(solution) + random.choice([-1, 1]))
        else:
            student_answer = correct_answer + " + minor error"
        proximity_score = random.uniform(6.0, 8.5)
        is_correct = False
    else:
        if problem_type == "algebra":
            student_answer = str(int(solution) + random.randint(-10, 10))
        else:
            student_answer = "Incorrect approach: " + correct_answer.split(" ")[0]
        proximity_score = random.uniform(1.0, 5.5)
        is_correct = False
    
    # Create the state
    state = {
        "question": question,
        "student_answer": student_answer,
        "correct_answer": correct_answer,
        "proximity_score": proximity_score,
        "is_correct": is_correct,
        "problem_id": f"{problem_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "chat_history": []
    }
    
    return state

def simulate_verification_result(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate a verification result.
    
    Args:
        state: The problem state
        
    Returns:
        Simulated verification result
    """
    is_correct = state.get("is_correct", False)
    proximity = state.get("proximity_score", 5.0) / 10.0
    
    # Verification is usually accurate but not perfect
    if random.random() < 0.9:
        verified_correct = is_correct
    else:
        verified_correct = not is_correct
    
    # Generate confidence based on proximity and correctness match
    base_confidence = 0.7
    correctness_bonus = 0.1 if verified_correct == is_correct else -0.2
    proximity_factor = (proximity - 0.5) * 0.4
    
    confidence = min(0.95, max(0.4, base_confidence + correctness_bonus + proximity_factor))
    
    return {
        "is_correct": verified_correct,
        "confidence": confidence,
        "details": {
            "verification_method": "symbolic",
            "match_score": proximity * 100
        }
    }

def simulate_hint(problem_type: str, hint_number: int) -> str:
    """
    Simulate a hint for the problem.
    
    Args:
        problem_type: Type of problem
        hint_number: Which hint this is (1st, 2nd, etc.)
        
    Returns:
        Simulated hint text
    """
    if problem_type == "algebra":
        hints = [
            "First, isolate the terms with x on one side of the equation.",
            "You need to divide both sides by the coefficient of x to solve for x.",
            "Remember to carefully handle the signs when moving terms across the equals sign.",
            "The final form should be x = (result - term) / coefficient."
        ]
    elif problem_type == "calculus":
        hints = [
            "Recall the basic differentiation rules for each term.",
            "For composite functions like e^x * ln(x), use the product rule.",
            "The product rule states: d/dx[f(x)*g(x)] = f'(x)*g(x) + f(x)*g'(x).",
            "Don't forget that the derivative of e^x is e^x and the derivative of ln(x) is 1/x."
        ]
    elif problem_type == "statistics":
        hints = [
            "For a normal distribution, you need to convert to a standard normal (Z-score).",
            "The formula for Z-score is Z = (x - μ) / σ.",
            "Once you have the Z-score, use a standard normal table or the appropriate function to find the probability.",
            "For values below the mean, the probability will be less than 0.5."
        ]
    else:
        hints = [
            "Think about the fundamental concepts involved in this problem.",
            "Try breaking the problem into smaller, more manageable parts.",
            "Consider what formulas or methods would be applicable here.",
            "Double-check your calculation for any arithmetic errors."
        ]
    
    if hint_number <= len(hints):
        return hints[hint_number - 1]
    else:
        return f"Hint {hint_number}: Look carefully at your previous work and correct any errors."

def demonstrate_real_confidence_metrics() -> None:
    """Demonstrate real confidence metrics in action."""
    logger.info("=== DEMONSTRATING REAL CONFIDENCE METRICS ===")
    
    # Initialize the confidence manager
    confidence_manager = ConfidenceManager(enable_ml=False)
    
    # Generate example problems
    problem_types = ["algebra", "calculus", "statistics"]
    
    for problem_type in problem_types:
        logger.info(f"\n--- {problem_type.upper()} PROBLEM DEMONSTRATION ---")
        
        # Create problem state
        state = create_simulated_problem_state(problem_type)
        
        logger.info(f"Problem: {state['question']}")
        logger.info(f"Student answer: {state['student_answer']}")
        logger.info(f"Correct answer: {state['correct_answer']}")
        logger.info(f"Is correct: {state['is_correct']}")
        logger.info(f"Proximity score: {state['proximity_score']:.1f}/10.0")
        
        # Simulate verification
        verification = simulate_verification_result(state)
        
        # 1. Assess feedback confidence
        feedback_confidence = confidence_manager.assess_feedback_confidence(
            state, verification, model_uncertainty=0.2
        )
        
        confidence_level = confidence_manager.get_confidence_level(feedback_confidence)
        logger.info(f"Feedback confidence: {feedback_confidence:.2f} ({confidence_level})")
        
        # Record feedback result (sometimes incorrectly to simulate real-world conditions)
        was_correctly_assessed = random.random() < 0.9
        confidence_manager.record_feedback_result(state, feedback_confidence, was_correctly_assessed)
        
        # 2. Provide hints with confidence scoring
        logger.info("\nGenerating hints with confidence scores:")
        for hint_num in range(1, 4):
            hint = simulate_hint(problem_type, hint_num)
            
            hint_confidence = confidence_manager.assess_hint_confidence(
                state, hint, hint_num, verification
            )
            
            confidence_level = confidence_manager.get_confidence_level(hint_confidence)
            logger.info(f"Hint {hint_num}: {hint}")
            logger.info(f"  Confidence: {hint_confidence:.2f} ({confidence_level})")
            
            # Record hint result
            was_helpful = random.random() < 0.8  # 80% chance hint was helpful
            confidence_manager.record_hint_result(state, hint, hint_num, hint_confidence, was_helpful)
        
        # 3. Analysis confidence
        analysis_result = {
            "is_correct": verification["is_correct"],
            "explanation": f"The analysis shows that the student's answer is {'correct' if verification['is_correct'] else 'incorrect'}. " +
                           f"The expected answer is {state['correct_answer']}. " +
                           f"The student's approach {'was valid' if state['proximity_score'] > 7 else 'had some issues'}."
        }
        
        analysis_confidence = confidence_manager.assess_analysis_confidence(
            state, analysis_result, verification
        )
        
        confidence_level = confidence_manager.get_confidence_level(analysis_confidence)
        logger.info(f"\nAnalysis confidence: {analysis_confidence:.2f} ({confidence_level})")
        logger.info(f"Analysis: {analysis_result['explanation']}")
        
        # Record analysis result
        confidence_manager.record_analysis_result(
            state, 
            analysis_result["explanation"], 
            analysis_confidence, 
            random.random() < 0.9  # 90% chance analysis was correct
        )
        
        # 4. Chat confidence
        follow_up_question = f"Why is my answer of {state['student_answer']} {'incorrect' if not state['is_correct'] else 'correct'}?"
        
        chat_response = (
            f"Your answer of {state['student_answer']} is {'correct' if state['is_correct'] else 'incorrect'} " +
            f"because {'it matches the expected answer' if state['is_correct'] else 'the correct answer is ' + state['correct_answer']}. " +
            f"{'Great job!' if state['is_correct'] else 'You might want to review the steps involved in solving this problem.'}"
        )
        
        chat_confidence = confidence_manager.assess_chat_confidence(
            state, follow_up_question, chat_response
        )
        
        confidence_level = confidence_manager.get_confidence_level(chat_confidence)
        logger.info(f"\nStudent follow-up: {follow_up_question}")
        logger.info(f"Response: {chat_response}")
        logger.info(f"Chat confidence: {chat_confidence:.2f} ({confidence_level})")
        
        # Record chat result
        confidence_manager.record_chat_result(
            state,
            follow_up_question,
            chat_response,
            chat_confidence,
            random.random() < 0.85  # 85% chance chat was helpful
        )

def demonstrate_ml_confidence_prediction() -> None:
    """Demonstrate ML-based confidence prediction."""
    logger.info("\n=== DEMONSTRATING ML-BASED CONFIDENCE PREDICTION ===")
    
    # Initialize the confidence manager with ML enabled
    confidence_manager = ConfidenceManager(enable_ml=True)
    
    # Generate training data
    logger.info("Generating training data...")
    problem_types = ["algebra", "calculus", "statistics"]
    
    # Generate 50 problems for training
    for _ in range(50):
        problem_type = random.choice(problem_types)
        state = create_simulated_problem_state(problem_type)
        verification = simulate_verification_result(state)
        
        # Record feedback
        feedback_confidence = confidence_manager.assess_feedback_confidence(
            state, verification, model_uncertainty=0.2
        )
        confidence_manager.record_feedback_result(
            state, feedback_confidence, random.random() < 0.9
        )
        
        # Record hints
        for hint_num in range(1, 4):
            hint = simulate_hint(problem_type, hint_num)
            hint_confidence = confidence_manager.assess_hint_confidence(
                state, hint, hint_num, verification
            )
            confidence_manager.record_hint_result(
                state, hint, hint_num, hint_confidence, random.random() < 0.8
            )
        
        # Record analysis
        analysis_result = {
            "is_correct": verification["is_correct"],
            "explanation": f"Analysis of student answer: {state['student_answer']}"
        }
        analysis_confidence = confidence_manager.assess_analysis_confidence(
            state, analysis_result, verification
        )
        confidence_manager.record_analysis_result(
            state, analysis_result["explanation"], analysis_confidence, random.random() < 0.9
        )
        
        # Record chat
        follow_up = "Why is this answer wrong?"
        response = "Your answer has the following issues..."
        chat_confidence = confidence_manager.assess_chat_confidence(
            state, follow_up, response
        )
        confidence_manager.record_chat_result(
            state, follow_up, response, chat_confidence, random.random() < 0.85
        )
    
    # Train models
    logger.info("Training ML models...")
    training_results = confidence_manager.train_models()
    
    # Log training results
    for component, result in training_results.items():
        if "error" in result:
            logger.info(f"Error training {component} model: {result['error']}")
        else:
            logger.info(f"Trained {component} model with {result['samples']} samples")
            if "performance" in result:
                perf = result["performance"]
                logger.info(f"  MSE: {perf.get('mse', 'N/A')}")
                logger.info(f"  MAE: {perf.get('mae', 'N/A')}")
    
    # Demonstrate prediction
    logger.info("\nDemonstrating ML prediction...")
    
    # Generate a new problem for demonstration
    state = create_simulated_problem_state(random.choice(problem_types))
    verification = simulate_verification_result(state)
    
    logger.info(f"Problem: {state['question']}")
    logger.info(f"Student answer: {state['student_answer']}")
    
    # Assess feedback confidence
    feedback_confidence = confidence_manager.assess_feedback_confidence(
        state, verification, model_uncertainty=0.2
    )
    
    confidence_level = confidence_manager.get_confidence_level(feedback_confidence)
    logger.info(f"ML-based feedback confidence: {feedback_confidence:.2f} ({confidence_level})")
    
    # Show feature importance
    if confidence_manager.predictor:
        logger.info("\nFeature importance for feedback confidence:")
        importance = confidence_manager.predictor.get_feature_importance("feedback")
        for feature, score in importance.items():
            logger.info(f"  {feature}: {score:.3f}")

def demonstrate_confidence_calibration() -> None:
    """Demonstrate confidence calibration."""
    logger.info("\n=== DEMONSTRATING CONFIDENCE CALIBRATION ===")
    
    # Initialize the confidence manager
    confidence_manager = ConfidenceManager(enable_ml=True)
    
    # Generate calibration data
    logger.info("Generating calibration data...")
    problem_types = ["algebra", "calculus", "statistics"]
    
    # Generate 100 problems for calibration
    for _ in range(100):
        problem_type = random.choice(problem_types)
        state = create_simulated_problem_state(problem_type)
        verification = simulate_verification_result(state)
        
        # Record feedback with deliberately biased confidence
        feedback_confidence = random.uniform(0.7, 0.9)  # Deliberately overconfident
        was_correct = random.random() < 0.6  # Only correct 60% of the time
        confidence_manager.record_feedback_result(state, feedback_confidence, was_correct)
        
        # Record hints with deliberate bias
        for hint_num in range(1, 4):
            hint = simulate_hint(problem_type, hint_num)
            hint_confidence = random.uniform(0.6, 0.8)  # Deliberately overconfident
            was_helpful = random.random() < 0.5  # Only helpful 50% of the time
            confidence_manager.record_hint_result(state, hint, hint_num, hint_confidence, was_helpful)
    
    # Calibrate the models
    logger.info("Calibrating confidence models...")
    calibration_results = confidence_manager.calibrate_models()
    
    # Log calibration results
    for component, result in calibration_results.items():
        if "error" in result:
            logger.info(f"Error calibrating {component}: {result['error']}")
        else:
            logger.info(f"Calibrated {component} with {result['samples']} samples")
            logger.info(f"  Temperature: {result['temperature']:.3f}")
            logger.info(f"  Isotonic bins: {result['isotonic_bins']}")
    
    # Demonstrate calibration
    logger.info("\nDemonstrating calibration effects...")
    
    # Compare calibrated vs. uncalibrated confidence
    state = create_simulated_problem_state(random.choice(problem_types))
    verification = simulate_verification_result(state)
    
    # Raw confidence (uncalibrated)
    raw_confidence = confidence_manager.metrics.assess_feedback_confidence(
        state, verification, model_uncertainty=0.2
    )
    
    # Calibrated confidence
    calibrated_confidence = confidence_manager.assess_feedback_confidence(
        state, verification, model_uncertainty=0.2
    )
    
    logger.info(f"Problem: {state['question']}")
    logger.info(f"Uncalibrated confidence: {raw_confidence:.3f}")
    logger.info(f"Calibrated confidence: {calibrated_confidence:.3f}")
    
    # Get calibration metrics
    logger.info("\nCalibration metrics:")
    metrics = confidence_manager.get_calibration_metrics()
    
    for component, data in metrics.items():
        if "error" not in data:
            logger.info(f"{component} calibration:")
            logger.info(f"  Mean prediction: {data.get('mean_prediction', 'N/A'):.3f}")
            logger.info(f"  Mean actual: {data.get('mean_actual', 'N/A'):.3f}")
            if "calibration_mse" in data:
                logger.info(f"  Calibration MSE: {data['calibration_mse']:.3f}")

def main():
    """Main demo function."""
    # Ensure data directories exist
    os.makedirs("app/data/calibration", exist_ok=True)
    os.makedirs("app/data/models", exist_ok=True)
    
    logger.info("Starting Advanced Confidence Metrics Demo")
    
    # Part 1: Real confidence metrics
    demonstrate_real_confidence_metrics()
    
    # Part 2: ML-based prediction
    demonstrate_ml_confidence_prediction()
    
    # Part 3: Calibration
    demonstrate_confidence_calibration()
    
    logger.info("\nDemo complete!")

if __name__ == "__main__":
    main() 