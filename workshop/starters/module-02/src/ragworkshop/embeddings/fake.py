import re


class FakeEmbeddingProvider:
    """Deterministic lexical demo, NOT a semantic embedding model."""
    model_name = "demo-keyword-v1"
    dimensions = 8
    vocabulary = ("python", "cat", "dog", "database", "storage", "text", "vector")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = re.findall(r"\w+", text.casefold())
            vectors.append([float(words.count(word)) for word in self.vocabulary] + [1.0])
        return vectors
