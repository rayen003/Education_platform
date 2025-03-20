"""
Text-to-Speech service for generating narration audio.
"""

import logging
import os
import tempfile
from typing import Optional
import requests
from dotenv import load_dotenv
import boto3
import google.cloud.texttospeech as tts

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TTSService:
    """
    Service for converting text to speech using various providers.
    
    Supports:
    - Amazon Polly
    - Google Cloud TTS
    - OpenAI TTS
    - Local fallback (if available)
    """
    
    def __init__(self, provider="openai"):
        """
        Initialize the TTS service.
        
        Args:
            provider: The TTS provider to use ("openai", "google", "aws", or "local")
        """
        self.provider = provider
        logger.info(f"Initialized TTSService with provider: {provider}")
        
        # Initialize the appropriate client based on provider
        if provider == "aws":
            self.client = boto3.client('polly')
        elif provider == "google":
            self.client = tts.TextToSpeechClient()
        elif provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.client = None  # Uses requests directly
        else:
            self.client = None
    
    def generate_speech(self, text: str, voice_id: Optional[str] = None) -> str:
        """
        Generate speech from text.
        
        Args:
            text: The text to convert to speech
            voice_id: Optional voice identifier
            
        Returns:
            Path to the generated audio file
        """
        if self.provider == "aws":
            return self._generate_speech_aws(text, voice_id or "Joanna")
        elif self.provider == "google":
            return self._generate_speech_google(text, voice_id or "en-US-Wavenet-D")
        elif self.provider == "openai":
            return self._generate_speech_openai(text, voice_id or "alloy")
        else:
            return self._generate_speech_local(text)
    
    def _generate_speech_aws(self, text: str, voice_id: str) -> str:
        """Generate speech using AWS Polly."""
        try:
            response = self.client.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId=voice_id
            )
            
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(response['AudioStream'].read())
                temp_file_path = temp_file.name
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error generating speech with AWS Polly: {e}")
            return self._generate_speech_local(text)
    
    def _generate_speech_google(self, text: str, voice_id: str) -> str:
        """Generate speech using Google Cloud TTS."""
        try:
            # Set the text input to be synthesized
            synthesis_input = tts.SynthesisInput(text=text)
            
            # Build the voice request
            voice = tts.VoiceSelectionParams(
                language_code="en-US",
                name=voice_id
            )
            
            # Select the type of audio file
            audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MP3
            )
            
            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            # Save the audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(response.audio_content)
                temp_file_path = temp_file.name
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error generating speech with Google Cloud TTS: {e}")
            return self._generate_speech_local(text)
    
    def _generate_speech_openai(self, text: str, voice_id: str) -> str:
        """Generate speech using OpenAI's TTS API."""
        try:
            url = "https://api.openai.com/v1/audio/speech"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "tts-1",
                "input": text,
                "voice": voice_id
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                logger.error(f"OpenAI TTS API error: {response.text}")
                return self._generate_speech_local(text)
            
            # Save the audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error generating speech with OpenAI TTS: {e}")
            return self._generate_speech_local(text)
    
    def _generate_speech_local(self, text: str) -> str:
        """Generate speech using local TTS if available, or return a placeholder."""
        try:
            # Try to use pyttsx3 if available
            import pyttsx3
            
            engine = pyttsx3.init()
            
            # Save to a temporary file
            temp_file_path = tempfile.mktemp(suffix='.mp3')
            engine.save_to_file(text, temp_file_path)
            engine.runAndWait()
            
            return temp_file_path
            
        except ImportError:
            logger.warning("pyttsx3 not available for local TTS")
            # Return a placeholder path
            return os.path.join(os.path.dirname(__file__), "placeholder.mp3") 