"""Custom exception types for the modular app."""


class AppError(Exception):
    """Base application exception."""


class IngestionError(AppError):
    """Raised for document ingestion failures."""


class IndexingError(AppError):
    """Raised for index lifecycle failures."""

