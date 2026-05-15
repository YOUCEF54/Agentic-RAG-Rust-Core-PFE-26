"""Shared API dependencies for route modules."""

from app.core import config


def get_runtime_config() -> dict:
    """Expose key runtime config values to route handlers."""
    return {
        "app_name": config.APP_NAME,
        "api_type": config.API_TYPE,
        "db_dir": config.DB_DIR,
        "table_name": config.TABLE_NAME,
    }

