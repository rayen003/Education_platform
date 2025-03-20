"""
Animation Meta-Agent for verifying animation code before execution.

This module contains a specialized meta-agent that verifies and potentially corrects
animation code to ensure proper element placement, timing, and visual clarity.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Callable
import os
import time
from app.math_services.agent.meta_agent import MetaAgent

logger = logging.getLogger(__name__)

class AnimationMetaAgent(MetaAgent):
    """
    Meta-agent for verifying and correcting animation code.
    
    This agent implements a verification process that:
    1. Assesses the correctness of Manim animation code
    2. Validates element placement and timing
    3. Ensures animations are visually clear and educational
    4. Regenerates animation code when necessary
    """
    
    def __init__(self, model="gpt-4o", confidence_threshold=0.8, max_iterations=2, llm_service=None):
        """Initialize the animation meta-agent."""
        super().__init__(model, confidence_threshold, max_iterations, llm_service)
        logger.info(f"Initialized AnimationMetaAgent with model {model}")
        print(f"=== AnimationMetaAgent initialized with model: {model} ===")
    
    def verify_animation_code(self, question: str, explanation: str, animation_code: str) -> Dict[str, Any]:
        """
        Verify Manim animation code for correctness, visual clarity, and educational value.
        
        Args:
            question: The mathematical question being animated
            explanation: The explanation that the animation should illustrate
            animation_code: The Manim code to verify
            
        Returns:
            Dictionary with verification results and possibly corrected code
        """
        print(f"\n=== Verifying animation code for question: {question[:50]}... ===")
        
        system_prompt = """
        You are a visual education expert specializing in mathematical animations using Manim.
        
        Your task is to verify animation code for:
        1. Syntax correctness (valid Manim code)
        2. Proper element placement and timing
        3. Visual clarity and educational effectiveness
        4. Synchronization potential with audio narration
        
        Check specifically for:
        - Proper initialization of objects
        - Clear positioning of elements in the scene
        - Appropriate animation timing and pacing
        - Educational value of the visual elements
        - Potential for synchronization with audio narrative points
        
        Format your response as a JSON object with these fields:
        {
            "verified": true/false,
            "confidence": 0-100,
            "issues": ["issue1", "issue2", ...] or [] if none,
            "corrected_code": "corrected Manim code" or null if not needed,
            "sync_points": [
                {"time": seconds, "narration_point": "description", "animation_element": "element_name"},
                ...
            ] or [] if none identified
        }
        """
        
        user_prompt = f"""
        Question: {question}
        
        Explanation: {explanation}
        
        Animation Code to verify:
        ```python
        {animation_code}
        ```
        
        Verify this animation code for correctness, visual clarity, and educational value.
        """
        
        try:
            response = self.llm_service.generate_completion(system_prompt, user_prompt)
            content = response.get("content", "")
            
            try:
                verification_data = json.loads(content)
                
                # Normalize confidence to 0-1 scale
                if "confidence" in verification_data:
                    verification_data["confidence"] = verification_data["confidence"] / 100.0
                
                # Log verification results
                if verification_data.get("verified", False):
                    print(f"✅ Animation code VERIFIED with confidence: {verification_data.get('confidence', 0):.2f}")
                else:
                    print(f"❌ Animation code NOT VERIFIED with confidence: {verification_data.get('confidence', 0):.2f}")
                    print("Issues found:")
                    for issue in verification_data.get("issues", []):
                        print(f"  - {issue}")
                    if verification_data.get("corrected_code"):
                        print("🔧 Corrected code was provided")
                
                return verification_data
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse animation verification result as JSON: {content}")
                print("❌ ERROR: Failed to parse verification result as JSON")
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "issues": ["Failed to parse verification result"],
                    "corrected_code": None
                }
            
        except Exception as e:
            logger.error(f"Error verifying animation code: {e}")
            print(f"❌ ERROR: Failed to verify animation code: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "issues": [f"Verification error: {str(e)}"],
                "corrected_code": None
            }
    
    def generate_and_verify_animation(self, question: str, explanation: str, 
                                      generate_animation_func: Callable) -> Dict[str, Any]:
        """
        Generate animation code and verify it in an iterative process.
        
        Args:
            question: The mathematical question
            explanation: The explanation to visualize
            generate_animation_func: Function to generate animation code
            
        Returns:
            Dictionary with verified animation code and metadata
        """
        print(f"\n=== Starting animation generation and verification process ===")
        print(f"Question: {question[:50]}...")
        
        iterations = 0
        max_iterations = self.max_iterations
        
        # First generation
        print(f"\n--- Iteration {iterations+1}/{max_iterations}: Generating initial animation code ---")
        animation_code = generate_animation_func(question, explanation)
        
        while iterations < max_iterations:
            print(f"\n--- Iteration {iterations+1}/{max_iterations}: Verifying animation code ---")
            
            # Verify the animation code
            verification_result = self.verify_animation_code(question, explanation, animation_code)
            
            # Extract verification details
            verified = verification_result.get("verified", False)
            confidence = verification_result.get("confidence", 0.0)
            issues = verification_result.get("issues", [])
            corrected_code = verification_result.get("corrected_code", None)
            
            # If verified with sufficient confidence, or we've reached max iterations, break
            if verified and confidence >= self.confidence_threshold:
                logger.info(f"Animation code verified with confidence {confidence:.2f}")
                print(f"✅ Animation code verified successfully with confidence {confidence:.2f}")
                break
                
            # Use corrected code if provided
            if corrected_code:
                animation_code = corrected_code
                logger.info("Using corrected animation code from verification")
                print("🔄 Using corrected animation code provided by verification agent")
            # Otherwise regenerate
            elif iterations < max_iterations - 1:
                logger.info(f"Regenerating animation code (iteration {iterations+1})")
                print(f"\n--- Iteration {iterations+1}/{max_iterations}: Regenerating animation code ---")
                print(f"Issues to address:")
                for issue in issues:
                    print(f"  - {issue}")
                animation_code = generate_animation_func(question, explanation, issues)
            
            iterations += 1
        
        sync_points = verification_result.get("sync_points", [])
        print(f"\n=== Animation generation complete after {iterations} iterations ===")
        print(f"Final verification status: {'Verified' if verified else 'Not fully verified'}")
        print(f"Identified {len(sync_points)} synchronization points for narration")
        
        return {
            "animation_code": animation_code,
            "verified": verified,
            "confidence": confidence,
            "iterations": iterations,
            "issues": issues,
            "sync_points": sync_points
        } 