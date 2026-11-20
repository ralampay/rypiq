"""Stable document citation indicators for ranked retrieval results."""
from app.models.chat import ChatSource
from app.models.search import SearchResult


def build_chat_sources(results: list[SearchResult]) -> list[ChatSource]:
    sources = {}
    for result in results:
        document = result.document
        if document.id not in sources:
            sources[document.id] = ChatSource(
                document_id=document.id, name=document.name, filename=document.filename,
                source=document.source, indicator=f'[{len(sources) + 1}]')
    return list(sources.values())
