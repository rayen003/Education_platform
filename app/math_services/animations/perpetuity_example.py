"""
Example of generating an educational animation for a perpetuity problem.
"""

import logging
from app.math_services.animations.animation_generator import AnimationGenerator
from app.llm_services.llm_service import LLMService
from app.speech_services.tts_service import TTSService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Generate an educational animation for a perpetuity problem."""
    
    # Initialize services
    llm_service = LLMService()
    tts_service = TTSService()
    
    # Initialize animation generator
    animation_generator = AnimationGenerator(
        llm_service=llm_service,
        tts_service=tts_service
    )
    
    # Define problem and explanation
    question = "If you deposit $1000 at the beginning of each year into an account that earns 5% interest annually, what is the value of this perpetuity?"
    
    explanation = """
    A perpetuity is a stream of equal payments that continue indefinitely.
    
    For a perpetuity with payment P and interest rate r, the present value is given by:
    PV = P/r
    
    In this case, P = $1000 and r = 5% = 0.05
    
    So the present value is:
    PV = $1000/0.05 = $20,000
    
    This means that $20,000 invested today at 5% would generate $1000 per year indefinitely.
    """
    
    # Generate the educational animation
    result = animation_generator.generate_educational_animation(question, explanation)
    
    if result["success"]:
        logger.info(f"Successfully generated animation: {result['video_path']}")
    else:
        logger.error(f"Failed to generate animation: {result['error']}")
    
    # Return the full result for inspection
    return result

if __name__ == "__main__":
    main() 