"""Embedding adapter for OpenAI-compatible endpoints."""
from app.exceptions import ProviderRequestError, ValidationError
from app.providers.config import ProviderSettings
from app.providers.openai_compatible._client import ClientOwner
from app.providers.contracts.embedding import EmbeddingProvider
from app.providers.normalization import embedding_response
from app.providers.validation import validate_texts


class OpenAICompatibleEmbeddingProvider(ClientOwner[ProviderSettings], EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return self.settings.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        validate_texts(texts)
        if self.settings.capability != 'embedding':
            raise ValidationError('Provider was not configured for embeddings')
        if not texts:
            return []
        options = {}
        if self.settings.dimensions is not None:
            options['dimensions'] = self.settings.dimensions
        try:
            response = self.client.embeddings.create(model=self.model_name, input=texts, **options)
        except Exception:
            raise ProviderRequestError('Embedding request failed; check provider availability and configuration') from None
        return embedding_response(response, len(texts), self.settings.embedding_size)
