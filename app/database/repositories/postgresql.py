"""SQLAlchemy Core CRUD and exact pgvector cosine search."""
import math
import struct
from dataclasses import asdict
from datetime import datetime, timezone
from numbers import Real

import sqlalchemy as sa

from app.database.repositories.base import DocumentRepository
from app.models.document import Document
from app.models.search import SearchResult


def validate_embedding(embedding, dimensions):
    if not isinstance(embedding, (list, tuple)) or len(embedding) != dimensions:
        raise ValueError(f'Embedding must contain exactly {dimensions} numbers')
    try:
        if any(isinstance(x, bool) or not isinstance(x, Real) for x in embedding):
            raise ValueError
        # pgvector stores float32: validate the values PostgreSQL will receive.
        vector = [struct.unpack('f', struct.pack('f', x))[0] for x in embedding]
        if any(not math.isfinite(x) for x in vector) or not any(vector):
            raise ValueError
    except (TypeError, ValueError, OverflowError, struct.error):
        raise ValueError('Embedding must contain finite float32 values and have nonzero magnitude') from None
    return vector


def positive_limit(limit):
    if type(limit) is not int or limit < 1:
        raise ValueError('Limit must be a positive integer')


class PostgreSQLDocumentRepository(DocumentRepository):
    def __init__(self, engine, table):
        self.engine = engine
        self.table = table
        self.dimensions = table.c.embedding.type.dim

    def _values(self, document):
        values = asdict(document)
        values['embedding'] = validate_embedding(document.embedding, self.dimensions)
        return values

    @staticmethod
    def _document(row):
        values = dict(row)
        values['embedding'] = [float(value) for value in values['embedding']]
        return Document(**values)

    def create(self, document):
        values = self._values(document)
        with self.engine.begin() as connection:
            connection.execute(self.table.insert().values(**values))
        return document

    def get(self, id):
        with self.engine.connect() as connection:
            row = connection.execute(sa.select(self.table).where(self.table.c.id == id)).mappings().first()
        return self._document(row) if row else None

    def update(self, document):
        values = self._values(document)
        values['updated_at'] = datetime.now(timezone.utc)
        with self.engine.begin() as connection:
            result = connection.execute(self.table.update().where(
                self.table.c.id == document.id).values(**values))
            if not result.rowcount:
                raise KeyError(document.id)
        document.updated_at = values['updated_at']
        return document

    def delete(self, id):
        with self.engine.begin() as connection:
            return bool(connection.execute(self.table.delete().where(self.table.c.id == id)).rowcount)

    def list(self, limit=100, offset=0):
        positive_limit(limit)
        if type(offset) is not int or offset < 0:
            raise ValueError('Offset must be a nonnegative integer')
        with self.engine.connect() as connection:
            rows = connection.execute(sa.select(self.table).order_by(
                self.table.c.id).limit(limit).offset(offset)).mappings().all()
        return [self._document(row) for row in rows]

    def search_by_vector(self, embedding, limit=10, filters=None):
        vector = validate_embedding(embedding, self.dimensions)
        positive_limit(limit)
        distance = self.table.c.embedding.cosine_distance(vector).label('distance')
        query = sa.select(self.table, distance)
        if filters is not None:
            if not isinstance(filters, dict):
                raise ValueError('Filters must be scalar equality mappings')
            for key, value in filters.items():
                if (key not in self.table.c or key in ('embedding', 'metadata')
                        or isinstance(value, (dict, list, tuple, set))):
                    raise ValueError('Only scalar field equality filters are supported')
                query = query.where(self.table.c[key] == value)
        query = query.order_by(distance, self.table.c.id).limit(limit)
        with self.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        matches = []
        for row in rows:
            values = dict(row)
            score = values.pop('distance')
            if score is None or not math.isfinite(score):
                raise ValueError('Stored embedding has undefined cosine distance')
            matches.append(SearchResult(self._document(values), float(score)))
        return matches

    def close(self) -> None:
        """Release the pool when this repository was obtained as an owned resource."""
        self.engine.dispose()
