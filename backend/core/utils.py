from datetime import datetime, timezone, timedelta
import logging

UTC7 = timezone(timedelta(hours=7))

def utcnow() -> datetime:
    # Return current time as a timezone-aware UTC datetime
    return datetime.now(timezone.utc)


def to_utc7(dt: datetime) -> datetime:
    # Convert a UTC datetime to UTC+7
    if dt is None:
        return None
    if dt.tzinfo is None:
        # treat naive datetimes as UTC before converting
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(UTC7)


def utc7_now() -> datetime:
    # Return current time as a timezone-aware UTC+7 datetime
    return datetime.now(UTC7)


class UTC7Formatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC7)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()