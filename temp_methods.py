def _perform_analysis(self, state):
    """
    Perform analysis on the student's answer.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with analysis results
    """
    try:
        # Extract question and student answer
        question = state.get("question", "")
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        
        # Get parsed question information if available
        analysis = state.get("analysis", {})
        parsed_info = analysis.get("parsed_question", {})
        problem_type = parsed_info.get("type", "symbolic")
        
        # Use utility functions for normalization and comparison
        from app.agents.agents.math_commands.utilities.math_utils import normalize_answer, calculate_string_similarity, is_answer_correct
        
        # Normalize answers
        normalized_student = normalize_answer(student_answer)
        normalized_correct = normalize_answer(correct_answer)
        
        # Check if the answers are equivalent
        is_correct = is_answer_correct(student_answer, correct_answer)
        
        # Calculate similarity score
        similarity = calculate_string_similarity(normalized_student, normalized_correct)
        
        # Store results in state
        state["is_correct"] = is_correct
        state["proximity_score"] = similarity
        
        # Add detailed analysis
        if "analysis" not in state:
            state["analysis"] = {}
            
        state["analysis"]["comparison"] = {
            "student_normalized": normalized_student,
            "correct_normalized": normalized_correct,
            "is_equivalent": is_correct,
            "similarity_score": similarity
        }
        
        # Record event
        return self._record_event(state, {
            "type": "analysis",
            "is_correct": is_correct,
            "similarity": similarity
        })
        
    except Exception as e:
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Error performing analysis: {e}")
        logger.error(traceback.format_exc())
        
        # Default to incorrect when there's an error
        state["is_correct"] = False
        
        # Record error event
        return self._record_event(state, {
            "type": "error",
            "action": "perform_analysis",
            "error": str(e)
        })

def _generate_feedback(self, state):
    """
    Generate feedback for the student's answer.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with feedback
    """
    try:
        # Extract relevant information
        is_correct = state.get("is_correct", False)
        question = state.get("question", "")
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        proximity_score = state.get("proximity_score", 0.0)
        
        # Use the utility function to generate feedback
        from app.agents.agents.math_commands.utilities.feedback_utils import generate_feedback_based_on_correctness
        
        # Generate feedback
        feedback = generate_feedback_based_on_correctness(
            is_correct=is_correct,
            proximity_score=proximity_score,
            question=question,
            student_answer=student_answer,
            correct_answer=correct_answer
        )
        
        # Store feedback in state
        state["feedback"] = feedback
        
        # Record event
        return self._record_event(state, {
            "type": "feedback_generation",
            "feedback_generated": True
        })
        
    except Exception as e:
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Error generating feedback: {e}")
        logger.error(traceback.format_exc())
        
        # Use utility function for default feedback
        from app.agents.agents.math_commands.utilities.feedback_utils import create_default_feedback
        
        # Create default feedback based on correctness
        default_feedback = create_default_feedback(state.get("is_correct", False))
        
        # Store feedback in state
        state["feedback"] = default_feedback
        
        # Record error event
        return self._record_event(state, {
            "type": "error",
            "action": "generate_feedback",
            "error": str(e)
        })

def _determine_hint_need(self, state):
    """
    Determine if a hint is needed based on the student's answer.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with hint need determination
    """
    try:
        # Extract relevant information
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        is_correct = state.get("is_correct", False)
        proximity_score = state.get("proximity_score", 0.0)
        
        # If the answer is already correct, no hint is needed
        if is_correct:
            state["needs_hint"] = False
            hint_reason = "Answer is correct"
        # If proximity score is available, use it to determine hint need
        elif proximity_score is not None:
            # Use the utility function to determine if hint is needed
            from app.agents.agents.math_commands.utilities.hint_utils import determine_hint_need
            state["needs_hint"], hint_reason = determine_hint_need(proximity_score)
        else:
            # If no proximity score is available, default to providing a hint
            state["needs_hint"] = True
            hint_reason = "Unable to determine proximity to correct answer"
        
        # Record event
        return self._record_event(state, {
            "type": "hint_need_determination",
            "needs_hint": state["needs_hint"],
            "reason": hint_reason
        })
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error determining hint need: {e}")
        
        # Default to providing a hint when there's an error
        state["needs_hint"] = True
        
        # Record error event
        return self._record_event(state, {
            "type": "error",
            "action": "determine_hint_need",
            "error": str(e),
            "needs_hint": True
        })

def _generate_hints(self, state):
    """
    Generate hints for the student.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with hints
    """
    try:
        # Check if hints are needed
        if not state.get("needs_hint", False):
            return self._record_event(state, {
                "type": "hint_generation",
                "hint_generated": False,
                "reason": "No hint needed"
            })
        
        # Extract relevant information
        question = state.get("question", "")
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        parsed_info = state.get("analysis", {}).get("parsed_question", {})
        problem_type = parsed_info.get("type", "symbolic")
        
        # Use utility functions to generate a hint
        from app.agents.agents.math_commands.utilities.hint_utils import generate_hint_for_problem_type, create_hint_object
        
        # Generate a hint based on the problem type
        hint_count = state.get("hint_count", 0)
        hint_text = generate_hint_for_problem_type(
            problem_type=problem_type,
            parsed_info=parsed_info,
            hint_count=hint_count,
            correct_answer=correct_answer
        )
        
        # Create hint object
        hint = create_hint_object(hint_text)
        
        # Store the hint in the state
        if "hints" not in state:
            state["hints"] = []
        
        state["hints"].append(hint)
        state["hint_count"] = state.get("hint_count", 0) + 1
        
        # Record event
        return self._record_event(state, {
            "type": "hint_generation",
            "hint_generated": True,
            "hint": hint_text
        })
        
    except Exception as e:
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Error generating hints: {e}")
        logger.error(traceback.format_exc())
        
        # Use utility function for fallback hint
        from app.agents.agents.math_commands.utilities.hint_utils import generate_fallback_hint, create_hint_object
        
        # Fallback hint
        hint_text = generate_fallback_hint()
        
        # Create hint object
        hint = create_hint_object(hint_text)
        
        # Store the hint in the state
        if "hints" not in state:
            state["hints"] = []
        
        state["hints"].append(hint)
        
        # Record error event
        return self._record_event(state, {
            "type": "error",
            "action": "generate_hints",
            "error": str(e),
            "hint": hint_text
        })
