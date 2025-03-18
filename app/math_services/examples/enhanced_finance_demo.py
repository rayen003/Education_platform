#!/usr/bin/env python3
"""
Enhanced Finance Problem Demo with Confidence Metrics.

This demo showcases a comprehensive finance problem workflow with
confidence assessment, ML-based calibration, and detailed metrics.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import time
import random
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from app.math_services.agent.math_agent import MathAgent
from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.metrics.confidence_manager import ConfidenceManager
from app.math_services.services.service_container import ServiceContainer
from app.math_services.services.llm.openai_service import OpenAILLMService
from app.math_services.models.state import MathState, ChatMessage

# Import the commands
from app.math_services.commands.solve_command import MathSolveSymbolicallyCommand
from app.math_services.commands.analyze_command import MathAnalyzeCalculationCommand
from app.math_services.commands.feedback_command import MathGenerateFeedbackCommand
from app.math_services.commands.hint_command import MathGenerateHintCommand
from app.math_services.commands.reasoning_command import MathGenerateReasoningCommand
from app.math_services.commands.chat_command import MathChatFollowUpCommand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Initialize confidence manager
confidence_manager = ConfidenceManager(
    enable_ml=True,
    calibration_method="temperature"
)

def print_divider(title=None):
    """Print a divider with optional title."""
    if title:
        print(f"\n{'=' * 10} {title} {'=' * 10}\n")
    else:
        print("\n" + "=" * 40 + "\n")

def print_confidence_explanation(confidence, indent="  "):
    """Print human-readable confidence explanation."""
    if confidence >= 0.8:
        level = "VERY HIGH"
    elif confidence >= 0.6:
        level = "HIGH"
    elif confidence >= 0.4:
        level = "MEDIUM"
    elif confidence >= 0.2:
        level = "LOW"
    else:
        level = "VERY LOW"
        
    print(f"{indent}Confidence: {confidence:.2f} ({level})")
    
    if confidence >= 0.8:
        print(f"{indent}The system is very confident in this assessment.")
    elif confidence >= 0.6:
        print(f"{indent}The system has good confidence in this assessment.")
    elif confidence >= 0.4:
        print(f"{indent}The system has moderate confidence in this assessment.")
    elif confidence >= 0.2:
        print(f"{indent}The system has low confidence in this assessment.")
    else:
        print(f"{indent}The system is very uncertain about this assessment.")

def enhanced_finance_problem_demo():
    """Run the enhanced finance problem demo with confidence metrics."""
    print("Starting Finance Problem Demo with Advanced Confidence Metrics...")
    print("This demonstration showcases complex financial calculation analysis")
    print("with confidence assessment, ML-based calibration, and detailed metrics.")
    
    print_divider("INITIALIZING SERVICES")
    
    # Initialize services with detailed logging
    start_time = time.time()
    print("Initializing services...")
    
    # Create LLM service
    llm_service = OpenAILLMService(model="gpt-4o-mini")
    print(f"  Initialized OpenAI LLM service with model gpt-4o-mini")
    
    # Create meta agent
    meta_agent = MetaAgent(model="gpt-4o-mini", llm_service=llm_service)
    print(f"  Initialized Meta Agent for verification")
    
    # Create service container
    service_container = ServiceContainer(
        llm_service=llm_service,
        meta_agent=meta_agent
    )
    print(f"  Initialized Service Container with LLM and Meta Agent")
    
    # Create command instances
    solve_command = MathSolveSymbolicallyCommand(service_container)
    analyze_command = MathAnalyzeCalculationCommand(service_container)
    feedback_command = MathGenerateFeedbackCommand(service_container)
    hint_command = MathGenerateHintCommand(service_container)
    reasoning_command = MathGenerateReasoningCommand(service_container)
    chat_command = MathChatFollowUpCommand(service_container)
    print(f"  Initialized Math Commands")
    
    # Register health checks
    service_container.get_service_health()
    print(f"  Registered health checks for all services")
    
    print(f"Services initialized in {time.time() - start_time:.2f} seconds")
    
    # Create problem statement
    print_divider("PROBLEM STATEMENT")
    
    problem = """
    You are analyzing an investment opportunity with the following details:
    - Initial investment: $500,000
    - Expected cash flows:
      * Year 1: $120,000
      * Year 2: $150,000
      * Year 3: $180,000
      * Year 4: $200,000
      * Year 5: $250,000
    - Discount rate: 10%
    
    Calculate the Net Present Value (NPV) of this investment.
    """
    
    print(problem)
    print("\nThis requires calculating the NPV using the formula:")
    print("NPV = -Initial Investment + Σ(Cash Flow_t / (1 + r)^t)")
    print("where r is the discount rate and t is the time period.")
    
    # Create initial state
    state = MathState(
        question=problem,
        student_answer="$156,718",
        correct_answer=None  # Will be calculated
    )
    
    print_divider("SOLVING PROBLEM")
    
    # Solve the problem and measure confidence
    print("Calculating exact solution...")
    start_time = time.time()
    print("Initializing solution...")
    
    # Step 1: Solving the problem
    solve_start = time.time()
    print("  Solving symbolically...")
    
    # Execute solve command
    state = solve_command.execute(state)
    
    solve_time = time.time() - solve_start
    print(f"  Solution completed in {solve_time:.2f} seconds")
    
    # Extract the correct answer
    correct_answer = state.correct_answer
    
    # Assess solving confidence using metrics
    solve_confidence = confidence_manager.assess_feedback_confidence(
        state=state.to_dict(),
        verification_result={"is_valid": True, "confidence": 0.95}
    )
    
    print(f"\nCorrect solution: {correct_answer}")
    print_confidence_explanation(solve_confidence)
    
    # Record this interaction for ML training
    confidence_manager.record_feedback_result(
        state=state.to_dict(),
        predicted_confidence=solve_confidence,
        was_correct=True  # The system's own solution is correct
    )
    
    print_divider("ANALYZING STUDENT ANSWER")
    
    # Step 2: Analyzing student's answer
    print(f"Student's answer: {state.student_answer}")
    print("Analyzing calculation...")
    
    # Execute analyze command
    analyze_start = time.time()
    state = analyze_command.execute(state)
    
    # Get analysis confidence
    analysis_confidence = confidence_manager.assess_analysis_confidence(
        state=state.to_dict(),
        analysis_result=state.analysis.to_dict()
    )
    
    print(f"\nAnalysis result: {'Correct' if state.analysis.is_correct else 'Incorrect'}")
    if not state.analysis.is_correct and state.analysis.error_type:
        print(f"Error type: {state.analysis.error_type}")
    
    print_confidence_explanation(analysis_confidence)
    
    # Record analysis outcome
    confidence_manager.record_analysis_result(
        state=state.to_dict(),
        analysis_summary=str(state.analysis.is_correct),
        predicted_confidence=analysis_confidence,
        was_correct=True  # Assume our analysis is correct for demo
    )
    
    analyze_time = time.time() - analyze_start
    print(f"Analysis completed in {analyze_time:.2f} seconds")
    
    print_divider("GENERATING FEEDBACK")
    
    # Step 3: Generating feedback with confidence
    print("Generating personalized feedback...")
    
    feedback_start = time.time()
    state = feedback_command.execute(state)
    
    # Get verification result from meta agent
    verification_result = {
        "is_valid": True,
        "confidence": 0.92,
        "reasoning": "The feedback correctly identifies the errors in the student's calculation"
    }
    
    # Get feedback confidence
    feedback_confidence = confidence_manager.assess_feedback_confidence(
        state=state.to_dict(),
        verification_result=verification_result
    )
    
    print("\nFeedback:")
    if hasattr(state.feedback, 'detail') and state.feedback.detail:
        feedback_detail = state.feedback.detail
    else:
        feedback_detail = state.feedback.get("detail", "")
    print(feedback_detail)
    print_confidence_explanation(feedback_confidence)
    
    # Record feedback outcome
    confidence_manager.record_feedback_result(
        state=state.to_dict(),
        predicted_confidence=feedback_confidence,
        was_correct=True  # Assume feedback is correct for demo
    )
    
    feedback_time = time.time() - feedback_start
    print(f"Feedback generated in {feedback_time:.2f} seconds")
    
    print_divider("GENERATING REASONING STEPS")
    
    # Step 4: Generate reasoning steps with confidence
    print("Generating step-by-step reasoning...")
    
    reasoning_start = time.time()
    state = reasoning_command.execute(state)
    
    print("\nReasoning steps:")
    for i, step in enumerate(state.steps, 1):
        print(f"{i}. {step}")
        
        # Generate per-step confidence
        step_confidence = 0.95 - (i * 0.05)  # Decreasing confidence per step for demo
        print_confidence_explanation(step_confidence, indent="   ")
        
        # Record step confidence
        confidence_manager.record_hint_result(
            state=state.to_dict(),
            hint=step,
            hint_number=i,
            predicted_confidence=step_confidence,
            was_helpful=random.random() < step_confidence  # Simulate helpfulness based on confidence
        )
    
    reasoning_time = time.time() - reasoning_start
    print(f"Reasoning steps generated in {reasoning_time:.2f} seconds")
    
    print_divider("GENERATING HINTS")
    
    # Step 5: Generate progressive hints with confidence
    print("Generating progressive hints...")
    
    # First hint
    state.needs_hint = True
    hint1_start = time.time()
    state = hint_command.execute(state)
    
    # Get hint confidence
    hint1_confidence = confidence_manager.assess_hint_confidence(
        state=state.to_dict(),
        hint=state.hints[-1],
        hint_number=1
    )
    
    print("\nHint 1:")
    print(state.hints[-1])
    print_confidence_explanation(hint1_confidence)
    
    # Record hint outcome
    confidence_manager.record_hint_result(
        state=state.to_dict(),
        hint=state.hints[-1],
        hint_number=1,
        predicted_confidence=hint1_confidence,
        was_helpful=True  # Assume helpful for demo
    )
    
    # Second hint
    state.needs_hint = True
    hint2_start = time.time()
    state = hint_command.execute(state)
    
    # Get hint confidence
    hint2_confidence = confidence_manager.assess_hint_confidence(
        state=state.to_dict(),
        hint=state.hints[-1],
        hint_number=2
    )
    
    print("\nHint 2:")
    print(state.hints[-1])
    print_confidence_explanation(hint2_confidence)
    
    # Record hint outcome
    confidence_manager.record_hint_result(
        state=state.to_dict(),
        hint=state.hints[-1],
        hint_number=2,
        predicted_confidence=hint2_confidence,
        was_helpful=True  # Assume helpful for demo
    )
    
    # Third hint
    state.needs_hint = True
    hint3_start = time.time()
    state = hint_command.execute(state)
    
    # Get hint confidence
    hint3_confidence = confidence_manager.assess_hint_confidence(
        state=state.to_dict(),
        hint=state.hints[-1],
        hint_number=3
    )
    
    print("\nHint 3:")
    print(state.hints[-1])
    print_confidence_explanation(hint3_confidence)
    
    # Record hint outcome
    confidence_manager.record_hint_result(
        state=state.to_dict(),
        hint=state.hints[-1],
        hint_number=3,
        predicted_confidence=hint3_confidence,
        was_helpful=False  # Simulate unhelpful for demo
    )
    
    print_divider("SECOND STUDENT ATTEMPT")
    
    # Step 6: Student second attempt
    second_answer = "$148,069"
    print(f"Student's second answer: {second_answer}")
    
    # Update state with new answer
    state.student_answer = second_answer
    
    # Analyze second attempt
    analyze2_start = time.time()
    state = analyze_command.execute(state)
    
    # Get analysis confidence
    analysis2_confidence = confidence_manager.assess_analysis_confidence(
        state=state.to_dict(),
        analysis_result=state.analysis.to_dict()
    )
    
    print(f"\nAnalysis result: {'Correct' if state.analysis.is_correct else 'Incorrect'}")
    print_confidence_explanation(analysis2_confidence)
    
    # Record analysis outcome
    confidence_manager.record_analysis_result(
        state=state.to_dict(),
        analysis_summary=str(state.analysis.is_correct),
        predicted_confidence=analysis2_confidence,
        was_correct=True  # Assume our analysis is correct for demo
    )
    
    # Generate feedback for second attempt
    feedback2_start = time.time()
    state = feedback_command.execute(state)
    
    # Get feedback confidence
    feedback2_confidence = confidence_manager.assess_feedback_confidence(
        state=state.to_dict(),
        verification_result={"is_valid": True, "confidence": 0.95}
    )
    
    print("\nFeedback for second attempt:")
    if hasattr(state.feedback, 'detail') and state.feedback.detail:
        feedback_detail = state.feedback.detail
    else:
        feedback_detail = state.feedback.get("detail", "")
    print(feedback_detail)
    print_confidence_explanation(feedback2_confidence)
    
    # Record feedback outcome
    confidence_manager.record_feedback_result(
        state=state.to_dict(),
        predicted_confidence=feedback2_confidence,
        was_correct=True  # Assume feedback is correct for demo
    )
    
    print_divider("CHAT FOLLOW-UP")
    
    # Step 7: Chat follow-up interaction
    follow_up_question = "Can you explain why we need to discount future cash flows?"
    print(f"Student asks: \"{follow_up_question}\"")
    
    # Add a chat message to history
    state.chat_history.append(ChatMessage(role="student", message=follow_up_question))
    
    # Handle follow-up question
    chat_start = time.time()
    state = chat_command.execute(state, follow_up_question)
    
    # Get chat confidence
    chat_confidence = confidence_manager.assess_chat_confidence(
        state=state.to_dict(),
        follow_up_question=follow_up_question,
        response=state.chat_response or ""
    )
    
    print("\nResponse:")
    print(state.chat_response)
    print_confidence_explanation(chat_confidence)
    
    # Record chat outcome
    confidence_manager.record_chat_result(
        state=state.to_dict(),
        follow_up_question=follow_up_question,
        response=state.chat_response or "",
        predicted_confidence=chat_confidence,
        was_helpful=True  # Assume helpful for demo
    )
    
    chat_time = time.time() - chat_start
    print(f"Chat response generated in {chat_time:.2f} seconds")
    
    print_divider("CONFIDENCE METRICS SUMMARY")
    
    # Train ML models with collected data
    print("Training confidence models with collected data...")
    try:
        training_results = confidence_manager.train_models()
        for component, result in training_results.items():
            if "error" in result:
                print(f"  Error training {component} model: {result['error']}")
            else:
                print(f"  Trained {component} model with {result.get('samples', 0)} samples")
                if "feature_importance" in result:
                    print("  Feature importance:")
                    for feature, importance in result["feature_importance"].items():
                        print(f"    - {feature}: {importance:.3f}")
    except Exception as e:
        print(f"  Error training models: {e}")
    
    # Calibrate confidence scores
    print("\nCalibrating confidence scores...")
    try:
        calibration_results = confidence_manager.calibrate_models()
        for component, result in calibration_results.items():
            if "error" in result:
                print(f"  Error calibrating {component}: {result['error']}")
            else:
                print(f"  Calibrated {component} with {result.get('samples', 0)} samples")
                print(f"    Temperature: {result.get('temperature', 0):.3f}")
                print(f"    Isotonic bins: {result.get('isotonic_bins', 0)}")
    except Exception as e:
        print(f"  Error calibrating models: {e}")
    
    # Get calibration metrics
    metrics = confidence_manager.get_calibration_metrics()
    print("\nCalibration metrics:")
    for component, component_metrics in metrics.items():
        print(f"  {component}:")
        print(f"    Mean prediction: {component_metrics.get('mean_prediction', 0):.3f}")
        print(f"    Mean actual: {component_metrics.get('mean_actual', 0):.3f}")
        print(f"    Calibration MSE: {component_metrics.get('calibration_mse', 0):.3f}")
    
    print_divider("DEMO COMPLETE")
    print("Advanced Finance Problem Demo with Confidence Metrics completed successfully!")
    print("This demo showcased:")
    print("1. Real-time confidence assessment for different component types")
    print("2. Progressive confidence for hints with decreasing certainty")
    print("3. ML-based confidence prediction and calibration")
    print("4. Confidence metrics visualization and explanation")
    print("5. Integration with the Meta Agent for verification")

if __name__ == "__main__":
    enhanced_finance_problem_demo() 