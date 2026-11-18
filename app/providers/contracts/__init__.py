"""Compatibility exports; new callers import from capability modules."""
from app.exceptions import ProviderError
from app.providers.contracts.embedding import EmbeddingProvider
from app.providers.contracts.inference import InferenceProvider
from app.providers.contracts.discovery import ModelDiscovery
from app.providers.types.chat import ChatMessage, ChatResult
from app.providers.types.model import ModelInfo

__all__ = ['ProviderError', 'EmbeddingProvider', 'InferenceProvider', 'ModelDiscovery',
           'ChatMessage', 'ChatResult', 'ModelInfo']
