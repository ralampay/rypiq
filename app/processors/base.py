from abc import ABC, abstractmethod

class BaseProcessor(ABC):

    @abstractmethod
    def process(self, content: str) -> dict:
        pass
