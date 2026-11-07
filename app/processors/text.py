from app.processors.base import BaseProcessor

class TextProcessor(BaseProcessor):
    def process(self, content: str) -> dict:
        return {
            "medium": "text",
            "content": content,
            "characters": len(content),
            "words": len(content.split()),
        }
