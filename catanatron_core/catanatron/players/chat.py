import os
import logging
import httpx
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from google import genai

logger = logging.getLogger(__name__)

class ChatProvider(ABC):
    """Abstract base class for chat providers"""
    @abstractmethod
    async def chat(self, prompt: str) -> str:
        """Send chat message and get response"""
        pass

class AnthropicChat(ChatProvider):
    """Anthropic Claude chat provider"""
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

    async def chat(self, prompt: str) -> str:
        """Send chat message to Anthropic Claude"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2024-01-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]

class OpenAIChat(ChatProvider):
    """OpenAI chat provider"""
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")

    async def chat(self, prompt: str) -> str:
        """Send chat message to OpenAI"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

class GoogleChat(ChatProvider):
    """Google Gemini chat provider"""
    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        
        self.client = genai.Client(api_key=self.api_key)

    async def chat(self, prompt: str) -> str:
        """Send chat messages to Google Gemini"""
        # Combine all messages into a single context
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        return response.text

class XAIChat(ChatProvider):
    """XAI chat provider"""
    def __init__(self, model: str = "grok-beta", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY not found")

    async def chat(self, prompt: str) -> str:
        """Send chat message to XAI"""
        # Note: This is a placeholder as XAI's API details aren't public yet
        # Update this implementation when the API becomes available
        url = "https://api.xai.com/v1/chat"  # Placeholder URL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["response"]

def create_chat_provider(provider: str, **kwargs) -> ChatProvider:
    """Factory function to create chat provider instances"""
    providers = {
        "anthropic": AnthropicChat,
        "openai": OpenAIChat,
        "google": GoogleChat,
        "xai": XAIChat
    }
    
    if provider not in providers:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return providers[provider](**kwargs) 