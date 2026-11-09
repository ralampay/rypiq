from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the identifier of the embedding model
        """
        raise NotImplementedError

    @abstractmethod
    def embed(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate one embedding vector per input string.

        Args:
            texts: List of input strings.

        Returns:
            A list of embedding vectors, each represented
            as a list of floats.

        Contract:
            - Output order matches input order.
            - One vector is returned per input string.
            - All vectors have consistent dimensions.
        """
        raise NotImplementedError
