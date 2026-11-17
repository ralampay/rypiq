from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import Mock

import pytest

from app.providers.config import ProviderSettings
from app.providers.contracts import ProviderError
from app.providers.session import LocalModelSession


def settings(**changes):
    return ProviderSettings(**dict(provider='local', capability='embedding', model='embed.gguf',
                                   embedding_size=768, **changes))


def test_lazy_reuse_warmup_and_cleanup():
    embedding, inference = Mock(), Mock()
    value = settings()
    chat = replace(value, capability='inference', model='chat.gguf')
    with LocalModelSession(embedding_factory=embedding, inference_factory=inference) as session:
        embedding.assert_not_called()
        inference.assert_not_called()
        for _ in range(2):
            with session.open_provider(value) as provider:
                assert provider is embedding.return_value
        inference.assert_not_called()
        session.warmup(value, chat)
        with session.open_provider(chat) as provider:
            assert provider is inference.return_value
        embedding.assert_called_once()
        inference.assert_called_once()
        embedding.return_value.close.assert_not_called()
        inference.return_value.close.assert_not_called()
    session.close()
    embedding.return_value.close.assert_called_once()
    inference.return_value.close.assert_called_once()
    with pytest.raises(RuntimeError, match='closed'):
        with session.open_provider(value):
            pass
    with pytest.raises(RuntimeError, match='closed'):
        with session:
            pass


def test_cache_identity_and_request_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / 'embed.gguf'
    target.touch()
    (tmp_path / 'link.gguf').symlink_to(target)
    factory = Mock(side_effect=lambda **kwargs: Mock())
    inference = Mock(side_effect=lambda value: Mock())
    with LocalModelSession(embedding_factory=factory, inference_factory=inference) as session:
        value = settings()
        for model in ['embed.gguf', str(target), 'link.gguf']:
            session.warmup(replace(value, model=model))
        session.warmup(replace(value, document_prefix='different', query_prefix='', embedding_size=3))
        factory.assert_called_once_with(model_path=str(target), context_size=2048, gpu_layers=0)
        for changed in [replace(value, context_size=4096), replace(value, gpu_layers=-1),
                        replace(value, model='other/embed.gguf')]:
            session.warmup(changed)
        assert factory.call_count == 4
        chat = replace(value, capability='inference')
        session.warmup(chat, replace(chat, chat_format='chatml'))
        assert inference.call_count == 2


def test_failures_release_leases_and_allow_retry():
    provider = Mock()
    factory = Mock(side_effect=[ValueError('load failed'), provider])
    with LocalModelSession(embedding_factory=factory) as session:
        with pytest.raises(ValueError, match='load failed'):
            session.warmup(settings())
        failure = ValueError('operation failed')
        with pytest.raises(ValueError) as caught:
            with session.open_provider(settings()):
                raise failure
        assert caught.value is failure
        session.warmup(settings())
        assert factory.call_count == 2
        provider.close.assert_not_called()
    provider.close.assert_called_once()


def test_partial_warmup_and_cleanup_failures_preserve_primary_error():
    embedding = Mock()
    embedding.return_value.close.side_effect = RuntimeError('private detail')
    inference = Mock(side_effect=ValueError('load failed'))
    with pytest.raises(ValueError, match='load failed') as caught:
        with LocalModelSession(embedding_factory=embedding, inference_factory=inference) as session:
            session.warmup(settings(), replace(settings(), capability='inference'))
    assert any('cleanup' in note for note in caught.value.__notes__)
    embedding.return_value.close.assert_called_once()
    session.close()


def test_cleanup_attempts_all_providers_and_reports_safe_error():
    embedding, inference = Mock(), Mock()
    embedding.return_value.close.side_effect = RuntimeError('private detail')
    with pytest.raises(ProviderError, match='cleanup') as caught:
        with LocalModelSession(embedding_factory=embedding, inference_factory=inference) as session:
            session.warmup(settings(), replace(settings(), capability='inference'))
    assert 'private detail' not in str(caught.value)
    inference.return_value.close.assert_called_once()
    session.close()


def test_concurrent_acquisition_loads_once_and_close_waits():
    factory = Mock()
    acquired = Barrier(4)
    release = Event()
    closing = Event()
    session = LocalModelSession(embedding_factory=factory)

    def borrow():
        with session.open_provider(settings()):
            acquired.wait(timeout=5)
            assert release.wait(5)

    def close():
        closing.set()
        session.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        workers = [executor.submit(borrow) for _ in range(3)]
        try:
            acquired.wait(timeout=5)
            shutdown = executor.submit(close)
            assert closing.wait(5)
            assert not shutdown.done()
            factory.return_value.close.assert_not_called()
        finally:
            release.set()
        for worker in workers:
            worker.result(timeout=5)
        shutdown.result(timeout=5)
    factory.assert_called_once()
    factory.return_value.close.assert_called_once()


def test_concurrent_first_use_returns_same_instance():
    barrier = Barrier(4)
    factory = Mock(side_effect=lambda **kwargs: Mock())
    with LocalModelSession(embedding_factory=factory) as session:
        def acquire():
            barrier.wait(timeout=5)
            with session.open_provider(settings()) as provider:
                return provider
        with ThreadPoolExecutor(max_workers=4) as executor:
            providers = list(executor.map(lambda _: acquire(), range(4)))
        assert all(provider is providers[0] for provider in providers)
        factory.assert_called_once()


@pytest.mark.parametrize('change', [dict(provider='openai'), dict(capability='unknown'),
                                    dict(context_size=0), dict(gpu_layers=-2), dict(model='')])
def test_invalid_settings_fail_before_construction(change):
    factory = Mock()
    with LocalModelSession(embedding_factory=factory, inference_factory=factory) as session:
        with pytest.raises(ValueError):
            session.warmup(replace(settings(), **change))
    factory.assert_not_called()


def test_close_with_own_lease_fails_instead_of_deadlocking():
    with LocalModelSession(embedding_factory=Mock()) as session:
        with session.open_provider(settings()):
            with pytest.raises(RuntimeError, match='lease'):
                session.close()
        session.warmup(settings())


def test_concurrent_shutdown_closes_once_and_rejects_new_work():
    started, release = Event(), Event()
    factory = Mock()
    def cleanup():
        started.set()
        assert release.wait(5)
    factory.return_value.close.side_effect = cleanup
    session = LocalModelSession(embedding_factory=factory)
    session.warmup(settings())
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(session.close)
        try:
            assert started.wait(5)
            second = executor.submit(session.close)
            with pytest.raises(RuntimeError, match='closed'):
                session.warmup(settings())
            with pytest.raises(RuntimeError, match='closed'):
                with session:
                    pass
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    factory.return_value.close.assert_called_once()


def test_nested_leases_keep_outer_borrow_active():
    factory = Mock()
    with LocalModelSession(embedding_factory=factory) as session:
        with session.open_provider(settings()) as outer:
            with session.open_provider(settings()) as inner:
                assert inner is outer
            outer.close.assert_not_called()
            with pytest.raises(RuntimeError, match='lease'):
                session.close()
        factory.assert_called_once()
    outer.close.assert_called_once()
