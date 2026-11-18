"""Internal SDK client ownership shared by remote adapters."""
from typing import Generic, TypeVar

from app.exceptions import ProviderCleanupError
from app.providers.config import ProviderSettings, DiscoverySettings


Settings = TypeVar('Settings', ProviderSettings, DiscoverySettings)


class ClientOwner(Generic[Settings]):
    def __init__(self, settings: Settings, *, client, owns_client: bool = False) -> None:
        self.settings = settings
        self.client = client
        self.owns_client = owns_client

    def close(self) -> None:
        if self.owns_client:
            try:
                self.client.close()
            except Exception:
                raise ProviderCleanupError('Provider client cleanup failed') from None
