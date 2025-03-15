"""
Confidence Demo Script.

This script demonstrates the confidence indicators functionality with
sample math problems of varying difficulty and ambiguity.
"""

import os
import sys
import streamlit as st
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the necessary components
from app.math_services.models.state import MathState, MathAnalysis, MathFeedback, InteractionMode
from app.math_services.services.service_container import ServiceContainer
from app.math_services.services.llm.openai_service import OpenAILLMService
from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.ui.confidence_display import (
    display_confidence_bar,
    display_confidence_badge,
    display_confidence_tooltip,
    confidence_explanation
)

def create_demo_page():
    """Create the Streamlit demo page for confidence indicators."""
    
    st.set_page_config(
        page_title="Math Feedback Confidence Demo",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Math Feedback Confidence Demo")
    st.markdown("""
    This demo showcases how confidence indicators help students understand the reliability
    of feedback and hints for different types of math problems.
    """)
    
    # Create sample states with varying confidence levels
    states = create_sample_states()
    
    # Display the demo problems in tabs
    st.subheader("Sample Math Problems with Confidence Indicators")
    tabs = st.tabs([f"Problem {i+1}" for i in range(len(states))])
    
    for i, tab in enumerate(tabs):
        with tab:
            display_problem(states[i], i+1)

def create_sample_states():
    """Create sample math states with varying confidence levels."""
    
    # Problem 1: Simple arithmetic with high confidence
    state1 = MathState(
        question="What is 25% of 80?",
        student_answer="20",
        correct_answer="20"
    )
    state1.analysis = MathAnalysis(
        is_correct=True,
        error_type=None,
        misconception=None
    )
    state1.feedback = MathFeedback(
        assessment="Great job! You correctly calculated 25% of 80 by multiplying 80 by 0.25 to get 20.",
        is_correct=True,
        proximity_score=1.0,
        confidence=0.95
    )
    state1.context = {"analysis_confidence": 0.9}
    
    # Problem 2: Algebra problem with medium confidence
    state2 = MathState(
        question="Solve for x: 3x + 7 = 2x - 5",
        student_answer="x = -12",
        correct_answer="x = -12"
    )
    state2.analysis = MathAnalysis(
        is_correct=True,
        error_type=None,
        misconception=None
    )
    state2.feedback = MathFeedback(
        assessment="You've got the right answer! Solving 3x + 7 = 2x - 5 gives us x = -12. Good work on the algebraic manipulation.",
        is_correct=True,
        proximity_score=1.0,
        confidence=0.75
    )
    state2.context = {"analysis_confidence": 0.7}
    
    # Problem 3: Complex calculus with medium-low confidence
    state3 = MathState(
        question="Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
        student_answer="f'(x) = 3x^2 + 4x - 5",
        correct_answer="f'(x) = 3x^2 + 4x - 5"
    )
    state3.analysis = MathAnalysis(
        is_correct=True,
        error_type=None,
        misconception=None
    )
    state3.hints = [
        "Think about applying the power rule to each term separately.",
        "Remember that for a term x^n, the derivative is n*x^(n-1)."
    ]
    state3.feedback = MathFeedback(
        assessment="Your answer is correct! You've applied the power rule correctly to each term in the polynomial.",
        is_correct=True,
        proximity_score=1.0,
        confidence=0.65
    )
    state3.context = {"analysis_confidence": 0.6, "hint_confidence": 0.7}
    
    # Problem 4: Ambiguous word problem with low confidence
    state4 = MathState(
        question="A train travels from city A to city B at an average speed of 60 mph and returns at an average speed of 40 mph. What is the average speed for the entire journey?",
        student_answer="48 mph",
        correct_answer="48 mph"
    )
    state4.analysis = MathAnalysis(
        is_correct=False,
        error_type="conceptual",
        misconception="arithmetic_average"
    )
    state4.hints = [
        "Remember that average speed is total distance divided by total time.",
        "Calculate the time taken for each leg of the journey separately."
    ]
    state4.feedback = MathFeedback(
        assessment="The correct answer is actually 48 mph. Average speed isn't the arithmetic mean of the two speeds, but rather total distance divided by total time.",
        is_correct=False,
        proximity_score=0.0,
        confidence=0.45
    )
    state4.context = {"analysis_confidence": 0.4, "hint_confidence": 0.5}
    
    return [state1, state2, state3, state4]

def display_problem(state, number):
    """Display a sample problem with confidence indicators."""
    
    # Display problem information
    st.markdown(f"### Problem {number}: {state.question}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Student Answer:**")
        st.markdown(f"`{state.student_answer}`")
        
        st.markdown("**Correct Answer:**")
        st.markdown(f"`{state.correct_answer}`")
        
        # Display analysis information
        st.markdown("#### Analysis")
        if state.analysis.is_correct:
            st.success("✓ Analysis determined the answer is correct")
        else:
            st.error("✗ Analysis determined the answer is incorrect")
            
            if state.analysis.error_type:
                st.markdown(f"**Error Type:** {state.analysis.error_type}")
            if state.analysis.misconception:
                st.markdown(f"**Misconception:** {state.analysis.misconception}")
        
        # Display analysis confidence
        if "analysis_confidence" in state.context:
            display_confidence_bar(state.context["analysis_confidence"], "Analysis Confidence")
    
    with col2:
        # Display feedback
        st.markdown("#### Feedback")
        st.markdown(state.feedback.assessment)
        
        # Display feedback confidence
        display_confidence_bar(state.feedback.confidence, "Feedback Confidence")
        st.markdown(confidence_explanation(state.feedback.confidence))
        
        # Display hints if available
        if state.hints:
            st.markdown("#### Hints")
            for i, hint in enumerate(state.hints):
                with st.expander(f"Hint {i+1}"):
                    st.markdown(hint)
                    
                    # Display hint confidence for the last hint
                    if i == len(state.hints) - 1 and "hint_confidence" in state.context:
                        display_confidence_badge(state.context["hint_confidence"])

if __name__ == "__main__":
    create_demo_page() 