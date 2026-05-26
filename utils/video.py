"""Video source resolution utilities."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

GO2RTC_PREFIX = "go2rtc:"

def resolve_source(source: str, go2rtc_url: Optional[str] = None) -> str:
    """Resolve a source string to an OpenCV-readable URL.

    Supports:
    - ``go2rtc:stream_name`` → ``{go2rtc_url}/stream?src=stream_name``
    - ``rtsp://...``, ``http://...``, file paths → returned as-is

    Args:
        source: The source string from the config.
        go2rtc_url: The go2rtc base URL (from GO2RTC_URL env var).

    Returns:
        A URL or path that can be passed to cv2.VideoCapture.

    Raises:
        ValueError: If source uses the go2rtc prefix but GO2RTC_URL is not set.
    """
    if not source.startswith(GO2RTC_PREFIX):
        return source

    if not go2rtc_url:
        raise ValueError(
            f"Source '{source}' requires GO2RTC_URL in .env"
        )

    stream_name = source[len(GO2RTC_PREFIX):]
    # Strip trailing slash from go2rtc_url to avoid double slashes
    base = go2rtc_url.rstrip("/")
    resolved = f"{base}/stream?src={stream_name}"
    logger.debug("Resolved %s → %s", source, resolved)
    return resolved
