import logging

from storage.database import LogEntry

logger = logging.getLogger(__name__)


class LogRepository:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def save(self, entry: LogEntry) -> int | None:
        try:
            with self._session_factory() as session:
                session.add(entry)
                session.commit()
                return entry.id
        except Exception as e:
            logger.error("DB save failed: %s", e)
            return None
