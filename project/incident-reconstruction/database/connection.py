"""Reusable MongoDB connection utilities.

All database connections are handled through this module. Credentials are never
hardcoded; they are read from environment variables loaded via python-dotenv.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_database = None


def load_env(env_path: Optional[str] = None) -> None:
    """Load environment variables from a .env file (does not override existing vars)."""
    load_dotenv(dotenv_path=env_path, override=False)


def get_client(env_path: Optional[str] = None) -> MongoClient:
    """Return a cached MongoClient, creating and pinging it on first use.

    Raises ConnectionFailure if MongoDB cannot be reached so callers can handle
    the failure gracefully.
    """
    global _client
    if _client is None:
        load_env(env_path)
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        try:
            _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Force a connection check immediately so failures surface early.
            _client.admin.command("ping")
            logger.info("Connected to MongoDB (%s)", uri)
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            logger.error("Failed to connect to MongoDB at %s: %s", uri, exc)
            raise
    return _client


def get_database(env_path: Optional[str] = None):
    """Return the configured database handle, creating it on first use."""
    global _database
    if _database is None:
        client = get_client(env_path)
        name = os.getenv("DATABASE_NAME", "incident_reconstruction")
        _database = client[name]
        logger.info("Using database: %s", name)
    return _database


def close_connection() -> None:
    """Close the cached MongoClient (useful for tests / clean shutdown)."""
    global _client, _database
    if _client is not None:
        try:
            _client.close()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("Error while closing MongoClient", exc_info=True)
        _client = None
        _database = None


def ping() -> bool:
    """Return True if the MongoDB server is reachable, False otherwise."""
    try:
        get_client().admin.command("ping")
        return True
    except Exception:  # connection failures are expected here
        logger.debug("MongoDB ping failed", exc_info=True)
        return False