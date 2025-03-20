"""
Animation Generator for math education content.

This module coordinates the process of creating educational animations with synchronized
narration for mathematical concepts and problems.
"""

import logging
import os
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
from app.math_services.animations.animation_meta_agent import AnimationMetaAgent
from app.llm_services.llm_service import LLMService

logger = logging.getLogger(__name__)

class AnimationGenerator:
    """
    Generates educational animations with synchronized narration.
    
    This class orchestrates:
    1. Animation code generation using Manim
    2. Verification of animation code
    3. Generation of narration scripts
    4. Text-to-speech conversion
    5. Synchronization of animation and narration
    """
    
    def __init__(self, llm_service=None, tts_service=None):
        """
        Initialize the animation generator.
        
        Args:
            llm_service: Service for LLM interactions
            tts_service: Service for text-to-speech conversion
        """
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.animation_meta_agent = AnimationMetaAgent(llm_service=llm_service)
        logger.info("Initialized AnimationGenerator")
    
    def generate_animation_code(self, question: str, explanation: str, issues: List[str] = None) -> str:
        """
        Generate Manim code for animating a mathematical explanation.
        
        Args:
            question: The math question
            explanation: The explanation to visualize
            issues: Optional list of issues from previous generation attempts
            
        Returns:
            Manim code as a string
        """
        system_prompt = """
        You are a Manim expert creating educational animations for mathematics.
        
        Create Python code using the Manim library to animate the explanation for the given math problem.
        The animation should:
        
        1. Be clear and visually engaging
        2. Illustrate key concepts in the explanation
        3. Have appropriate timing (around 20 seconds total)
        4. Include mathematical notation where appropriate
        5. Be structured for synchronization with narration
        
        Return only the Python code that would be in a .py file for Manim.
        Use Manim's best practices for educational content.
        """
        
        user_prompt = f"""
        Question: {question}
        
        Explanation to visualize: {explanation}
        
        {"Previous issues to address:" + "\\n- " + "\\n- ".join(issues) if issues else ""}
        
        Create Manim animation code (approximately 20 seconds) that clearly visualizes this explanation.
        """
        
        response = self.llm_service.generate_completion(system_prompt, user_prompt)
        content = response.get("content", "")
        
        # Extract code from response if needed
        code_pattern = r'```python\s+(.*?)\s+```'
        code_match = re.search(code_pattern, content, re.DOTALL)
        if code_match:
            return code_match.group(1)
        return content
    
    def generate_narration_script(self, question: str, explanation: str, 
                                 sync_points: List[Dict[str, Any]]) -> str:
        """
        Generate a narration script with timing cues.
        
        Args:
            question: The math question
            explanation: The explanation
            sync_points: Synchronization points from animation verification
            
        Returns:
            Narration script with timing annotations
        """
        system_prompt = """
        You are an educational narrator specializing in clear explanations of mathematical concepts.
        
        Create a narration script for an animation about the given math problem.
        The script should:
        
        1. Be clear, concise, and engaging
        2. Match the provided sync points for animation timing
        3. Be approximately 20 seconds when read at a natural pace
        4. Explain the key concepts from the explanation
        
        Format your response as a simple script with timing annotations in [brackets].
        """
        
        # Format sync points for the prompt
        sync_points_text = "\n".join([
            f"- At {sp['time']}s: {sp['narration_point']} (animation: {sp['animation_element']})"
            for sp in sync_points
        ]) if sync_points else "No specific sync points provided."
        
        user_prompt = f"""
        Question: {question}
        
        Explanation: {explanation}
        
        Sync points for animation:
        {sync_points_text}
        
        Create a narration script (approximately 20 seconds) that aligns with these animation points.
        """
        
        response = self.llm_service.generate_completion(system_prompt, user_prompt)
        return response.get("content", "")
    
    def generate_tts_audio(self, narration_script: str) -> str:
        """
        Generate TTS audio from narration script.
        
        Args:
            narration_script: The narration script
            
        Returns:
            Path to generated audio file
        """
        # Remove timing annotations for TTS
        clean_script = re.sub(r'\[\d+\.?\d*s\]', '', narration_script)
        
        # Generate audio using TTS service
        if self.tts_service:
            return self.tts_service.generate_speech(clean_script)
        else:
            logger.warning("TTS service not provided, returning placeholder")
            return "placeholder_audio.wav"
    
    def execute_animation_code(self, animation_code: str) -> str:
        """
        Execute Manim animation code and return path to generated video.
        
        Args:
            animation_code: Manim code to execute
            
        Returns:
            Path to generated video file
        """
        # Create temporary file for the animation code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_file.write(animation_code.encode('utf-8'))
            temp_file_path = temp_file.name
        
        try:
            # Execute Manim command
            output_path = tempfile.mkdtemp()
            cmd = [
                'manim', 
                '-qm',  # Medium quality
                '-o', output_path,
                temp_file_path
            ]
            
            subprocess.run(cmd, check=True)
            
            # Find the generated video file
            for file in os.listdir(output_path):
                if file.endswith('.mp4'):
                    return os.path.join(output_path, file)
            
            logger.error("No video file generated")
            return None
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing Manim: {e}")
            return None
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def combine_video_and_audio(self, video_path: str, audio_path: str) -> str:
        """
        Combine video and audio files.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            
        Returns:
            Path to combined video file
        """
        output_path = video_path.replace('.mp4', '_with_audio.mp4')
        
        try:
            # Use ffmpeg to combine video and audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v',
                '-map', '1:a',
                '-shortest',
                output_path
            ]
            
            subprocess.run(cmd, check=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error combining video and audio: {e}")
            return None
    
    def generate_educational_animation(self, question: str, explanation: str) -> Dict[str, Any]:
        """
        Generate a complete educational animation with synchronized narration.
        
        Args:
            question: The math question
            explanation: The explanation to animate
            
        Returns:
            Dictionary with paths to generated files and metadata
        """
        logger.info(f"Generating animation for question: {question[:50]}...")
        
        # Generate and verify animation code
        animation_result = self.animation_meta_agent.generate_and_verify_animation(
            question, 
            explanation,
            self.generate_animation_code
        )
        
        animation_code = animation_result["animation_code"]
        sync_points = animation_result["sync_points"]
        
        # Generate narration and audio
        narration_script = self.generate_narration_script(question, explanation, sync_points)
        audio_path = self.generate_tts_audio(narration_script)
        
        # Execute animation code
        video_path = self.execute_animation_code(animation_code)
        
        if not video_path or not audio_path:
            logger.error("Failed to generate video or audio")
            return {
                "success": False,
                "error": "Failed to generate video or audio",
                "animation_code": animation_code,
                "narration_script": narration_script
            }
        
        # Combine video and audio
        combined_path = self.combine_video_and_audio(video_path, audio_path)
        
        return {
            "success": True,
            "video_path": combined_path,
            "animation_code": animation_code,
            "narration_script": narration_script,
            "audio_path": audio_path,
            "raw_video_path": video_path,
            "verification_data": animation_result
        } 