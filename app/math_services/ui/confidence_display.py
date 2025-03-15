"""
Confidence Display Components.

This module provides helper functions for displaying confidence ratings
in the user interface.
"""

import streamlit as st

def display_confidence_bar(confidence: float, label: str = "Confidence") -> None:
    """
    Display a confidence bar with appropriate color coding.
    
    Args:
        confidence: Confidence score from 0.0 to 1.0
        label: Label for the confidence indicator
    """
    # Format confidence as percentage
    confidence_pct = int(confidence * 100)
    
    # Define color based on confidence level
    if confidence >= 0.8:
        color = "#28a745"  # Green for high confidence
        level = "High"
    elif confidence >= 0.6:
        color = "#ff9800"  # Orange for medium confidence
        level = "Medium"
    else:
        color = "#dc3545"  # Red for low confidence
        level = "Low"
    
    # Display the confidence bar
    st.markdown(f"**{label}:** {level} ({confidence_pct}%)")
    st.progress(confidence)
    
    # Add color indicator
    st.markdown(
        f"""
        <div style="width:100%; height:5px; background-color:{color}; margin-bottom:15px;"></div>
        """, 
        unsafe_allow_html=True
    )

def display_confidence_badge(confidence: float) -> None:
    """
    Display a compact confidence badge for inline use.
    
    Args:
        confidence: Confidence score from 0.0 to 1.0
    """
    # Format confidence as percentage
    confidence_pct = int(confidence * 100)
    
    # Define badge style based on confidence level
    if confidence >= 0.8:
        color = "#28a745"  # Green
        bg_color = "#d4edda"
        level = "High"
    elif confidence >= 0.6:
        color = "#ff9800"  # Orange
        bg_color = "#fff3cd"
        level = "Medium"
    else:
        color = "#dc3545"  # Red
        bg_color = "#f8d7da"
        level = "Low"
    
    # Create badge with HTML/CSS
    st.markdown(
        f"""
        <span style="display:inline-block; padding:3px 8px; background-color:{bg_color}; 
        color:{color}; border-radius:12px; font-size:0.8rem; font-weight:bold; 
        border:1px solid {color};">
        {level} ({confidence_pct}%)
        </span>
        """,
        unsafe_allow_html=True
    )

def display_confidence_tooltip(confidence: float, text: str) -> None:
    """
    Display text with a confidence tooltip.
    
    Args:
        confidence: Confidence score from 0.0 to 1.0
        text: Text to display alongside confidence
    """
    # Format confidence as percentage
    confidence_pct = int(confidence * 100)
    
    # Define style based on confidence level
    if confidence >= 0.8:
        icon = "✓"  # Checkmark for high confidence
        color = "#28a745"
    elif confidence >= 0.6:
        icon = "i"  # Info for medium confidence
        color = "#ff9800"
    else:
        icon = "!"  # Warning for low confidence
        color = "#dc3545"
    
    # Create tooltip with HTML/CSS
    st.markdown(
        f"""
        <div style="position:relative; display:inline-block; margin-bottom:10px;">
            <span style="background-color:{color}; color:white; border-radius:50%; 
            width:20px; height:20px; display:inline-flex; align-items:center; 
            justify-content:center; font-weight:bold; font-size:0.8rem; 
            margin-right:5px;">{icon}</span>
            {text}
            <span style="color:{color}; font-size:0.8rem; margin-left:5px;">
            ({confidence_pct}% confidence)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def confidence_explanation(confidence: float) -> str:
    """
    Generate an explanation for why the confidence level might be what it is.
    
    Args:
        confidence: Confidence score from 0.0 to 1.0
        
    Returns:
        An explanation string
    """
    if confidence >= 0.9:
        return ("This assessment has very high confidence based on clear problem structure, " 
                "well-defined answer criteria, and strong verification results.")
    elif confidence >= 0.8:
        return ("This assessment has high confidence due to strong understanding of the problem " 
                "pattern and clear validation of the solution approach.")
    elif confidence >= 0.6:
        return ("This assessment has moderate confidence. Factors that might reduce confidence " 
                "include problem complexity or multiple valid solution approaches.")
    elif confidence >= 0.4:
        return ("This assessment has limited confidence due to potential ambiguity in the problem, " 
                "unclear student response, or limited verification.")
    else:
        return ("This assessment has low confidence and should be treated as a suggestion rather " 
                "than a definitive assessment. Consider getting additional help.") 