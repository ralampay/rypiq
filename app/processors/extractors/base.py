"""Common contract for document text extraction."""
from abc import ABC, abstractmethod
from typing import BinaryIO


class BaseExtractor(ABC):
    """Stateless reader of a caller-owned, seekable binary stream.

    Read from the supplied stream and return extracted text. Do not close the
    stream; release any parser resources created during extraction. Let parser
    errors propagate to the file dispatcher for safe public diagnostics.
    """

    @abstractmethod
    def extract(self, stream: BinaryIO) -> str:
        """Extract text without taking ownership of the input stream."""
        pass
