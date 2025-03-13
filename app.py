from flask import Flask, render_template, jsonify, request, session
import os
import json
from app.agents.agents.math_agent import MathAgent
from app.workflows.math_workflow import math_assessment_flow

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize agents
math_agent = MathAgent()

@app.route('/')
def index():
    return render_template('knowledge_graph.html')

@app.route('/api/knowledge_graph')
def get_knowledge_graph():
    data = {
        "nodes": [
            {
                "id": "1",
                "name": "Supply",
                "description": "The quantity of a good that producers are willing to sell at various prices",
                "difficulty": "Beginner",
                "resources": [
                    {"type": "video", "url": "https://example.com/supply-video", "title": "Understanding Supply"},
                    {"type": "article", "url": "https://example.com/supply-article", "title": "Supply in Economics"}
                ]
            },
            {
                "id": "2",
                "name": "Demand",
                "description": "The quantity of a good that consumers are willing to purchase at various prices",
                "difficulty": "Beginner",
                "resources": [
                    {"type": "video", "url": "https://example.com/demand-video", "title": "Demand Explained"},
                    {"type": "article", "url": "https://example.com/demand-article", "title": "Understanding Demand"}
                ]
            },
            {
                "id": "3",
                "name": "Price",
                "description": "The amount of money expected in exchange for a good or service",
                "difficulty": "Beginner",
                "resources": [
                    {"type": "video", "url": "https://example.com/price-video", "title": "Price Theory"},
                    {"type": "article", "url": "https://example.com/price-article", "title": "Price Mechanisms"}
                ]
            },
            {
                "id": "4",
                "name": "Elasticity",
                "description": "Measure of how demand or supply responds to changes in price",
                "difficulty": "Intermediate",
                "resources": [
                    {"type": "video", "url": "https://example.com/elasticity-video", "title": "Understanding Elasticity"},
                    {"type": "article", "url": "https://example.com/elasticity-article", "title": "Types of Elasticity"}
                ]
            },
            {
                "id": "5",
                "name": "Market Equilibrium",
                "description": "The point where supply meets demand",
                "difficulty": "Intermediate",
                "resources": [
                    {"type": "video", "url": "https://example.com/equilibrium-video", "title": "Market Equilibrium"},
                    {"type": "article", "url": "https://example.com/equilibrium-article", "title": "Understanding Market Balance"}
                ]
            }
        ],
        "links": [
            {"source": "1", "target": "3", "relationship": "determines"},
            {"source": "2", "target": "3", "relationship": "affects"},
            {"source": "1", "target": "5", "relationship": "contributes to"},
            {"source": "2", "target": "5", "relationship": "contributes to"},
            {"source": "3", "target": "5", "relationship": "establishes"},
            {"source": "1", "target": "4", "relationship": "measured by"},
            {"source": "2", "target": "4", "relationship": "measured by"}
        ]
    }
    return jsonify(data)

@app.route('/math/assessment', methods=['GET'])
def math_assessment_form():
    """Render the math assessment form."""
    return render_template('math_assessment.html')

@app.route('/math/submit', methods=['POST'])
def submit_math_answer():
    """
    Handle submission of a math problem answer.
    
    This endpoint demonstrates the conditional flow:
    1. First attempt -> check if correct
    2. If incorrect -> provide hint
    3. Second attempt -> check if correct
    4. If still incorrect -> provide another hint
    5. Third attempt -> final assessment and feedback
    """
    # Get form data
    question = request.form.get('question')
    student_answer = request.form.get('answer')
    attempt = int(request.form.get('attempt', 1))
    
    # Initialize session state if first attempt
    if attempt == 1 or 'math_state' not in session:
        session['math_state'] = {
            'question': question,
            'student_answer': student_answer,
            'hint_count': 0,
            'hints': [],
            'needs_hint': False,
            'feedback': [],
            'attempts': 1
        }
    else:
        # Update state with new answer for subsequent attempts
        state = session['math_state']
        state['student_answer'] = student_answer
        state['attempts'] = attempt
        session['math_state'] = state
    
    # Process the assessment
    state = session['math_state']
    
    # Run the math assessment flow for this attempt
    result = math_assessment_flow(state, agent=math_agent, max_attempts=3, attempt=attempt)
    
    # Save updated state to session
    session['math_state'] = result
    
    # Check if answer is correct
    is_correct = result.get('is_correct', False)
    
    # Prepare response
    response = {
        'is_correct': is_correct,
        'attempt': attempt,
        'max_attempts': 3
    }
    
    # Add hint or feedback based on the result
    if is_correct:
        response['message'] = "Correct! Well done."
        if 'feedback' in result and result['feedback']:
            # Format the feedback for display
            if isinstance(result['feedback'], list) and len(result['feedback']) > 0:
                latest_feedback = result['feedback'][-1]  # Get the most recent feedback
                
                # Format the feedback as HTML
                formatted_feedback = ""
                if 'summary' in latest_feedback:
                    formatted_feedback += f"<p><strong>Summary:</strong> {latest_feedback['summary']}</p>"
                
                if 'strengths' in latest_feedback and latest_feedback['strengths']:
                    formatted_feedback += "<p><strong>Strengths:</strong></p><ul>"
                    for strength in latest_feedback['strengths']:
                        formatted_feedback += f"<li>{strength}</li>"
                    formatted_feedback += "</ul>"
                
                if 'areas_for_improvement' in latest_feedback and latest_feedback['areas_for_improvement']:
                    formatted_feedback += "<p><strong>Areas for Improvement:</strong></p><ul>"
                    for area in latest_feedback['areas_for_improvement']:
                        formatted_feedback += f"<li>{area}</li>"
                    formatted_feedback += "</ul>"
                
                if 'next_steps' in latest_feedback and latest_feedback['next_steps']:
                    formatted_feedback += "<p><strong>Next Steps:</strong></p><ul>"
                    for step in latest_feedback['next_steps']:
                        formatted_feedback += f"<li>{step}</li>"
                    formatted_feedback += "</ul>"
                
                response['feedback'] = formatted_feedback
            else:
                # If feedback is not in expected format, convert to string
                response['feedback'] = str(result['feedback'])
    elif attempt < 3:
        # Not correct and more attempts available - provide hint
        if 'hints' in result and isinstance(result['hints'], list) and len(result['hints']) > 0:
            latest_hint = result['hints'][-1]  # Get the latest hint
            
            # Format the hint as HTML
            formatted_hint = ""
            if isinstance(latest_hint, dict):
                if 'summary' in latest_hint:
                    formatted_hint += f"<p><strong>{latest_hint['summary']}</strong></p>"
                
                if 'areas_for_improvement' in latest_hint and latest_hint['areas_for_improvement']:
                    formatted_hint += "<ul>"
                    for area in latest_hint['areas_for_improvement']:
                        formatted_hint += f"<li>{area}</li>"
                    formatted_hint += "</ul>"
                
                if 'next_steps' in latest_hint and latest_hint['next_steps']:
                    formatted_hint += "<p><strong>Try this:</strong></p><ul>"
                    for step in latest_hint['next_steps']:
                        formatted_hint += f"<li>{step}</li>"
                    formatted_hint += "</ul>"
            else:
                formatted_hint = str(latest_hint)
            
            response['hint'] = formatted_hint
        response['message'] = f"Not quite right. Try again! (Attempt {attempt}/3)"
    else:
        # Final attempt and still not correct - provide full feedback
        response['message'] = "Let's see the correct approach."
        if 'feedback' in result and result['feedback']:
            # Format the feedback for display
            if isinstance(result['feedback'], list) and len(result['feedback']) > 0:
                latest_feedback = result['feedback'][-1]  # Get the most recent feedback
                
                # Format the feedback as HTML
                formatted_feedback = ""
                if 'summary' in latest_feedback:
                    formatted_feedback += f"<p><strong>Summary:</strong> {latest_feedback['summary']}</p>"
                
                # Add solution explanation if available
                if 'solution_explanation' in latest_feedback:
                    formatted_feedback += f"<div class='solution-box'>{latest_feedback['solution_explanation']}</div>"
                
                if 'strengths' in latest_feedback and latest_feedback['strengths']:
                    formatted_feedback += "<p><strong>Strengths:</strong></p><ul>"
                    for strength in latest_feedback['strengths']:
                        formatted_feedback += f"<li>{strength}</li>"
                    formatted_feedback += "</ul>"
                
                if 'areas_for_improvement' in latest_feedback and latest_feedback['areas_for_improvement']:
                    formatted_feedback += "<p><strong>Areas for Improvement:</strong></p><ul>"
                    for area in latest_feedback['areas_for_improvement']:
                        formatted_feedback += f"<li>{area}</li>"
                    formatted_feedback += "</ul>"
                
                if 'next_steps' in latest_feedback and latest_feedback['next_steps']:
                    formatted_feedback += "<p><strong>Next Steps:</strong></p><ul>"
                    for step in latest_feedback['next_steps']:
                        formatted_feedback += f"<li>{step}</li>"
                    formatted_feedback += "</ul>"
                
                # Add the correct answer to the feedback
                if 'correct_answer' in result:
                    formatted_feedback += f"<p><strong>Correct Answer:</strong> {result['correct_answer']}</p>"
                
                response['feedback'] = formatted_feedback
            else:
                # If feedback is not in expected format, convert to string
                response['feedback'] = str(result['feedback'])
    
    return jsonify(response)

@app.route('/math/reset', methods=['POST'])
def reset_math_session():
    """Reset the math assessment session."""
    if 'math_state' in session:
        session.pop('math_state')
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
