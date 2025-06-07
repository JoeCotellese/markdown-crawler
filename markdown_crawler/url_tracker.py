import threading
import logging
from typing import Set
import urllib.parse

logger = logging.getLogger(__name__)

def normalize_url(url: str) -> str:
    """Normalize URL by removing trailing slashes and fragments."""
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), None, None, None))

class ThreadSafeURLTracker:
    """Thread-safe URL tracker to prevent duplicate crawling."""
    
    def __init__(self):
        self._crawled: Set[str] = set()
        self._lock = threading.Lock()
    
    def add_if_new(self, url: str) -> bool:
        """
        Add URL if new. Returns True if added, False if already existed.
        Thread-safe operation that atomically checks and adds.
        """
        normalized = normalize_url(url)
        with self._lock:
            if normalized not in self._crawled:
                self._crawled.add(normalized)
                logger.debug(f'✅ Added new URL to tracker: {normalized}')
                return True
            logger.debug(f'⏭️ URL already crawled: {normalized}')
            return False
    
    def is_crawled(self, url: str) -> bool:
        """Check if URL has been crawled (thread-safe)."""
        normalized = normalize_url(url)
        with self._lock:
            return normalized in self._crawled
    
    def get_crawled_count(self) -> int:
        """Get total number of crawled URLs."""
        with self._lock:
            return len(self._crawled)
    
    def get_all_crawled(self) -> Set[str]:
        """Get a copy of all crawled URLs (thread-safe)."""
        with self._lock:
            return self._crawled.copy()
    
    def clear(self) -> None:
        """Clear all tracked URLs (thread-safe)."""
        with self._lock:
            self._crawled.clear()
            logger.debug('🗑️ Cleared all tracked URLs')