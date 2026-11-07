from abc import ABC, abstractmethhod,n

class BaseProcessor(ABC):

    @abstractmethod
    def process(self, content: str) -> dict:
        pass
