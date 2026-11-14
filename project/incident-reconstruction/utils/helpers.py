"""Small shared helpers used across modules."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from dateutil import parser


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value) -> Optional[datetime]:
    """Parse many timestamp shapes into a UTC-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    dt = parser.isoparse(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def isoformat(dt) -> Optional[str]:
    if dt is None:
        return None
    dt = parse_timestamp(dt)
    return dt.isoformat() if dt else None


def to_minutes(td) -> float:
    """Convert a timedelta-like (or seconds number) to minutes."""
    if td is None:
        return 0.0
    seconds = td.total_seconds() if hasattr(td, "total_seconds") else float(td)
    return seconds / 60.0


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def next_person_id(existing: list[str]) -> str:
    """Generate the next anonymous Person_### id not in ``existing``."""
    nums = []
    for pid in existing or []:
        try:
            nums.append(int(pid.split("_")[-1]))
        except (ValueError, IndexError):
            continue
    n = max(nums) + 1 if nums else 1
    return f"Person_{n:03d}"