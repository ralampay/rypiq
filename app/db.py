from app.database.factory import get_database


def get_document_repository():
    return get_database().documents()
