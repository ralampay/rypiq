"""Non-secret snapshots used to compare processing and search configuration."""
from app.models.application import Application, chunking_settings
from app.providers.config import embedding_preprocessing

SNAPSHOT_KEY = '_rypiq_processing'


def embedding_settings(application: Application) -> dict:
    return dict(embedding_model=application.embedding_model,
                embedding_size=application.embedding_size,
                provision_type=application.provision_type,
                preprocessing=list(embedding_preprocessing(application.data,
                    application.provision_type, application.embedding_size)))


def processing_settings(application: Application) -> dict:
    return {**embedding_settings(application), 'chunking': list(chunking_settings(application.data))}
