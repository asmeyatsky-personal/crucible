"""
Application Settings

Architectural Intent:
- Centralised configuration loaded from environment variables
- No business logic — pure infrastructure concern
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite+aiosqlite:///crucible.db"
    storage_path: str = "./storage"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8100
    log_level: str = "INFO"

    @staticmethod
    def from_env() -> Settings:
        return Settings(
            database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///crucible.db"),
            storage_path=os.getenv("STORAGE_PATH", "./storage"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8100")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
