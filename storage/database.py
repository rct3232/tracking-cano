import json
import logging
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, DateTime, String, Boolean, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class LogEntry(Base):
    __tablename__ = "log_entries"

    id = Column(Integer, primary_key=True)
    batch_id = Column(String(36), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    log_type = Column(String(20), nullable=False, index=True)
    subject_id = Column(String(100), nullable=True, index=True)
    target_present = Column(Boolean, nullable=True)
    description = Column(Text, nullable=True)
    target_coordinate = Column(Text, nullable=True)
    raw_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)


def init_db(db_url: str):
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    try:
        engine = create_engine(db_url, connect_args=connect_args)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        logger.info("DB initialized: %s", db_url.split("://")[0])
        return engine, Session
    except Exception as e:
        logger.error("DB init failed (%s): %s", db_url, e)
        return None, None
