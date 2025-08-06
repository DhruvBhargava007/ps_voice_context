import base64
import os
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()

class BrainClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAIN_API_KEY")
        if not self.api_key:
            raise ValueError("BRAIN_API_KEY not found in environment or provided.")
        self.base_url = os.getenv("BRAIN_URL", "https://brain-platform.pattern.com")

    def _get_headers(self, content_type: str = "application/json") -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": content_type,
        }

    def transcribe(self, audio_bytes: bytes, timestamps: bool) -> Optional[dict]:
        """Handles audio transcription by calling the correct endpoint."""
        transcribe_url = f"{self.base_url}/api/v1/audio/transcriptions"
        headers = self._get_headers(content_type="multipart/form-data")
        del headers['Content-Type']

        files = {'audio': ('recording.wav', audio_bytes, 'audio/wav')}
        data = {
            "timestamp_granularity": "segment" if timestamps else None
        }

        try:
            response = requests.post(transcribe_url, headers=headers, files=files, data=data, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error invoking Brain Transcription API: {e}")
            if e.response: print(f"Response body: {e.response.text}")
            return None

    def invoke_llm(self, messages: list, model: str) -> Optional[dict]:
        """Handles language model invocation."""
        invoke_url = f"{self.base_url}/api/v1/llm/invoke"

        payload = {
            "model": model,
            "list_of_messages": messages,
            "prompt": messages[-1]['content'], # Use last message as main prompt
            "response_format": "json_object",
            "temperature": 0.1, # A default from the official client
            "system_message": "" # A default from the official client
        }

        try:
            response = requests.post(invoke_url, headers=self._get_headers(), json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error invoking Brain LLM API: {e}")
            if e.response: print(f"Response body: {e.response.text}")
            return None

    def list_conversations(self) -> Optional[dict]:
        """Lists all conversations for the user."""
        list_url = f"{self.base_url}/api/v1/conversations"
        print(f"Attempting to GET: {list_url}")
        try:
            response = requests.get(list_url, headers=self._get_headers(), timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error listing conversations: {e}")
            if e.response: print(f"Response body: {e.response.text}")
            return None 