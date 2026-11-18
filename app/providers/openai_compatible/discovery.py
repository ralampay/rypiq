"""Model discovery without a selected embedding or inference model."""
from app.exceptions import ProviderRequestError, ProviderResponseError
from app.providers.config import DiscoverySettings
from app.providers.openai_compatible._client import ClientOwner
from app.providers.types.model import ModelInfo


class OpenAICompatibleModelDiscovery(ClientOwner[DiscoverySettings]):
    def list_models(self) -> list[ModelInfo]:
        try:
            response = self.client.models.list()
        except Exception:
            raise ProviderRequestError('Model discovery failed; check provider availability and configuration') from None
        try:
            data = response['data'] if isinstance(response, dict) else [item.model_dump() for item in response]
            ids = {item['id'] for item in data}
            if any(not isinstance(model, str) or not model.strip() for model in ids):
                raise ValueError
            return [ModelInfo(model, self.settings.provider, (), 'discovered', 'available', model)
                    for model in sorted(ids)]
        except (TypeError, ValueError, KeyError, AttributeError):
            raise ProviderResponseError('Invalid model discovery response from provider') from None
