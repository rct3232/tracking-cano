import json
import logging
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
)
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


class AppSetting(Base):
    __tablename__ = "app_settings"

    key = Column(Text, primary_key=True)
    key_prefix = Column(Text, nullable=False)
    value_text = Column(Text, nullable=True)
    value_number = Column(Numeric(asdecimal=False), nullable=True)
    value_bool = Column(Boolean, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_app_settings_prefix", "key_prefix"),
    )


class ConfigVersion(Base):
    __tablename__ = "config_version"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        CheckConstraint("id = 1", name="config_version_id_check"),
    )


def init_db(db_url: str):
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    try:
        engine = create_engine(db_url, connect_args=connect_args)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        # Seed config_version row (idempotent)
        with Session() as session:
            existing = session.get(ConfigVersion, 1)
            if not existing:
                session.add(ConfigVersion(id=1, version=0))
                session.commit()

        logger.info("DB initialized: %s", db_url.split("://")[0])
        return engine, Session
    except Exception as e:
        logger.error("DB init failed (%s): %s", db_url, e)
        return None, None
