"""Explicit, thread-safe ownership of reusable local model providers."""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Condition, get_ident

from app.providers.config import ProviderSettings, validate_options
from app.providers.contracts import EmbeddingProvider, InferenceProvider, ProviderError
from app.providers.factory import create_provider


class LocalModelSession:
    """Borrow models inside leases; close outside leases, after submitting all work.

    One successful construction per canonical model/runtime configuration is retained
    until shutdown. Failed loads can be retried. No model is loaded at construction.
    """

    def __init__(self, *, embedding_factory: Callable | None = None,
                 inference_factory: Callable | None = None) -> None:
        self._dependencies = dict(embedding_factory=embedding_factory, inference_factory=inference_factory)
        self._providers: dict[tuple, EmbeddingProvider | InferenceProvider] = {}
        self._condition = Condition()
        self._leases: dict[int, int] = {}
        self._closing = False
        self._closed = False

    @contextmanager
    def open_provider(self, settings: ProviderSettings) -> Iterator[EmbeddingProvider | InferenceProvider]:
        """Borrow a local provider; callers must not close or retain it past the lease."""
        if settings.provider != 'local' or settings.capability not in ('embedding', 'inference'):
            raise ValueError('Local model sessions require a supported local capability')
        if not isinstance(settings.model, str) or not settings.model.strip():
            raise ValueError('Local model path must be nonempty')
        options = dict(context_size=settings.context_size, gpu_layers=settings.gpu_layers)
        if settings.capability == 'inference' and settings.chat_format is not None:
            options['chat_format'] = settings.chat_format
        validate_options(options, settings.capability)
        settings = replace(settings, model=str(Path(settings.model).expanduser().resolve()))
        key = (settings.capability, settings.model, settings.context_size, settings.gpu_layers,
               settings.chat_format if settings.capability == 'inference' else None)
        owner = get_ident()
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError('Local model session is closed or closing')
            # Construction under the lock makes concurrent first use atomic. Models
            # execute outside this lock, using their own existing execution locks.
            if key not in self._providers:
                self._providers[key] = create_provider(settings, **self._dependencies)
            provider = self._providers[key]
            self._leases[owner] = self._leases.get(owner, 0) + 1
        try:
            yield provider
        finally:
            with self._condition:
                self._leases[owner] -= 1
                if not self._leases[owner]:
                    del self._leases[owner]
                self._condition.notify_all()

    def warmup(self, *settings: ProviderSettings) -> None:
        """Load selected capabilities; successful loads remain cached if another fails."""
        for value in settings:
            with self.open_provider(value):
                pass

    def close(self) -> None:
        """Reject new leases, wait for active work, and attempt each cleanup once."""
        with self._condition:
            if get_ident() in self._leases:
                raise RuntimeError('Cannot close a local model session while holding a provider lease')
            if self._closing:
                self._condition.wait_for(lambda: self._closed)
            if self._closed:
                return
            self._closing = True
            self._condition.wait_for(lambda: not self._leases)
            providers = list(self._providers.values())
            self._providers.clear()
        failed = False
        try:
            for provider in providers:
                try:
                    provider.close()
                except Exception:
                    failed = True
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
        if failed:
            raise ProviderError('Local model session cleanup failed; resources may remain open') from None

    def __enter__(self) -> 'LocalModelSession':
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError('Local model session is closed or closing')
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        except Exception:
            if exc is None:
                raise
            exc.add_note('Local model session cleanup failed; resources may remain open.')
