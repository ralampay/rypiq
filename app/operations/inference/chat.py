"""Application-scoped RAG inference using borrowed resources."""
from collections.abc import Callable

from app.database.repositories.embedding import EmbeddingRepository
from app.exceptions import ValidationError
from app.logging import log_operation
from app.models.application import Application
from app.models.chat import RagChatResult
from app.models.search import SearchResult
from app.operations.documents.search import SearchDocuments
from app.operations.inference.prompts import build_rag_messages
from app.operations.inference.sources import build_chat_sources
from app.providers.types.chat import ChatMessage
from app.providers.contracts.embedding import EmbeddingProvider
from app.providers.contracts.inference import InferenceProvider
from app.providers.validation import validate_chat


PromptBuilder = Callable[[list[ChatMessage], list[SearchResult]], list[ChatMessage]]


def validate_rag_chat(messages: list[ChatMessage], max_tokens: int, temperature: float | None) -> None:
    validate_chat(messages, max_tokens, temperature)
    if messages[-1].role != 'user':
        raise ValidationError('RAG chat requires a final user message')


class Chat:
    def __init__(self, *, messages: list[ChatMessage], application: Application,
                 repository: EmbeddingRepository, embedding_provider: EmbeddingProvider,
                 provider: InferenceProvider, max_tokens: int = 512,
                 temperature: float | None = None,
                 prompt_builder: PromptBuilder = build_rag_messages) -> None:
        self.messages = messages
        self.application = application
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_builder = prompt_builder

    def execute(self) -> RagChatResult:
        with log_operation('chat'):
            validate_rag_chat(self.messages, self.max_tokens, self.temperature)
            results = SearchDocuments(query=self.messages[-1].content,
                application=self.application, repository=self.repository,
                embedding_provider=self.embedding_provider, limit=5).execute()
            sources = build_chat_sources(results)
            messages = self.prompt_builder(list(self.messages), results)
            validate_chat(messages, self.max_tokens, self.temperature)
            answer = self.provider.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature)
            return RagChatResult(text=answer.text, model=answer.model, finish_reason=answer.finish_reason,
                                 usage=answer.usage, sources=sources)
