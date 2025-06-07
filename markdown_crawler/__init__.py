from bs4 import BeautifulSoup
import urllib.parse
import threading
from contextlib import contextmanager
from markdownify import markdownify as md
import requests
import logging
import queue
import time
import os
import re
from typing import (
    List,
    Optional,
    Union,
    Set
)
from .url_tracker import ThreadSafeURLTracker


# Import Playwright with optional handling
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    sync_playwright = None
__version__ = '0.1'
__author__ = 'Paul Pierre (github.com/paulpierre)'
__copyright__ = "(C) 2023 Paul Pierre. MIT License."
__contributors__ = ['Paul Pierre']

BANNER = """
                |                                     |             
 __ `__ \    _` |        __|   __|   _` | \ \  \   /  |   _ \   __| 
 |   |   |  (   |       (     |     (   |  \ \  \ /   |   __/  |    
_|  _|  _| \__._|      \___| _|    \__._|   \_/\_/   _| \___| _|    

-------------------------------------------------------------------------
A multithreaded 🕸️ web crawler that recursively crawls a website and
creates a 🔽 markdown file for each page by https://github.com/paulpierre
-------------------------------------------------------------------------
"""

logger = logging.getLogger(__name__)
DEFAULT_BASE_DIR = 'markdown'
DEFAULT_MAX_DEPTH = 3
DEFAULT_NUM_THREADS = 5
DEFAULT_TARGET_CONTENT = ['article', 'div', 'main', 'p']
DEFAULT_TARGET_LINKS = ['body']
DEFAULT_DOMAIN_MATCH = True
DEFAULT_BASE_PATH_MATCH = True
DEFAULT_RENDER_HTML = False


# --------------
# URL validation
# --------------
def is_valid_url(url: str) -> bool:
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        logger.debug(f'❌ Invalid URL {url}')
        return False


# ----------------
# Clean up the URL
# ----------------
def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), None, None, None))


# ---------------------
# HTML rendering with JS
# ---------------------
_thread_local = threading.local()

@contextmanager
def get_browser_for_thread():
    """Get or create a browser instance for the current thread."""
    if not hasattr(_thread_local, 'playwright'):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError('Playwright is required for HTML rendering but not installed')
        
        _thread_local.playwright = sync_playwright()
        _thread_local.playwright_context = _thread_local.playwright.__enter__()
        _thread_local.browser = _thread_local.playwright_context.firefox.launch(headless=True)
        logger.debug(f'🚀 Launched browser for thread {threading.current_thread().ident}')
    
    try:
        yield _thread_local.browser
    finally:
        pass  # Keep browser alive for thread reuse

def cleanup_thread_browser():
    """Clean up browser for current thread."""
    if hasattr(_thread_local, 'browser'):
        _thread_local.browser.close()
        _thread_local.playwright_context.__exit__(None, None, None)
        logger.debug(f'🗑️ Cleaned up browser for thread {threading.current_thread().ident}')
        delattr(_thread_local, 'browser')
        delattr(_thread_local, 'playwright_context')
        delattr(_thread_local, 'playwright')

def render_html_with_js(url: str) -> str:
    """
    Render HTML content using Playwright with thread-local browser reuse.
    """
    try:
        with get_browser_for_thread() as browser:
            page = browser.new_page()
            page.set_default_timeout(30000)
            page.goto(url, wait_until='networkidle')
            html_content = page.content()
            page.close()  # Close page but keep browser alive
            logger.debug(f'✅ Successfully rendered HTML for {url}')
            return html_content
            
    except Exception as e:
        logger.error(f'❌ Error rendering HTML for {url}: {e}')
        raise


# ------------------
# HTML parsing logic
# ------------------
def crawl(
    url: str,
    base_url: str,
    url_tracker: ThreadSafeURLTracker,  # Changed from already_crawled: set
    file_path: str,
    target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
    target_content: Union[str, List[str]] = None,
    valid_paths: Union[str, List[str]] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH,
    is_links: Optional[bool] = False,
    render_html: Optional[bool] = DEFAULT_RENDER_HTML
) -> List[str]:

    # Atomically check and mark URL as crawled
    if not url_tracker.add_if_new(url):
        return []  # Already crawled by another thread
    
    # Get HTML content either with requests or Playwright
    try:
        logger.debug(f'🕷️ Crawling: {url}')
        if render_html:
            html_content = render_html_with_js(url)
            response_content_type = 'text/html'
        else:
            response = requests.get(url)
            html_content = response.text
            response_content_type = response.headers.get('Content-Type', '')
            
    except Exception as e:
        logger.error(f'❌ Request error for {url}: {e}')
        return []
        
    if 'text/html' not in response_content_type:
        logger.error(f'❌ Content not text/html for {url}')
        return []

    # Remove the old already_crawled.add(url) line since it's handled by url_tracker
    
    # ---------------------------------
    # List of elements we want to strip
    # ---------------------------------
    strip_elements = []

    if is_links:
        strip_elements = ['a']

    # -------------------------------
    # Create BS4 instance for parsing
    # -------------------------------
    soup = BeautifulSoup(html_content, 'html.parser')

    # Strip unwanted tags
    for script in soup(['script', 'style']):
        script.decompose()

    # --------------------------------------------
    # Write the markdown file if it does not exist
    # --------------------------------------------
    if not os.path.exists(file_path):

        file_name = file_path.split("/")[-1]

        # ------------------
        # Get target content
        # ------------------

        content = get_target_content(soup, target_content=target_content)

        if content:
            # --------------
            # Parse markdown
            # --------------
            output = md(
                content,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )

            logger.info(f'Created 📝 {file_name}')

            # ------------------------------
            # Write markdown content to file
            # ------------------------------
            with open(file_path, 'w') as f:
                f.write(output)
        else:
            logger.error(f'❌ Empty content for {file_path}. Please check your targets skipping.')

    child_urls = get_target_links(
        soup,
        base_url,
        target_links,
        valid_paths=valid_paths,
        is_domain_match=is_domain_match,
        is_base_path_match=is_base_path_match    
    )

    logger.debug(f'Found {len(child_urls) if child_urls else 0} child URLs')
    return child_urls


def get_target_content(
    soup: BeautifulSoup,
    target_content: Union[List[str], None] = None
) -> str:

    content = ''

    # -------------------------------------
    # Get target content by target selector
    # -------------------------------------
    if target_content:
        for target in target_content:
            for tag in soup.select(target):
                content += f'{str(tag)}'.replace('\n', '')

    # ---------------------------
    # Naive estimation of content
    # ---------------------------
    else:
        max_text_length = 0
        for tag in soup.find_all(DEFAULT_TARGET_CONTENT):
            text_length = len(tag.get_text())
            if text_length > max_text_length:
                max_text_length = text_length
                main_content = tag

        content = str(main_content)

    return content if len(content) > 0 else False


def get_target_links(
    soup: BeautifulSoup,
    base_url: str,
    target_links: List[str] = DEFAULT_TARGET_LINKS,
    valid_paths: Union[List[str], None] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH
) -> List[str]:

    child_urls = []
    base_parsed = urllib.parse.urlparse(base_url)

    # Get all urls from target_links
    for target in soup.find_all(target_links):
        # Get all the links in target
        for link in target.find_all('a'):
            href = link.get('href')
            if not href:
                continue
            
            # Convert relative URLs to absolute
            full_url = urllib.parse.urljoin(base_url, href)
            
            # Skip invalid URLs
            if not is_valid_url(full_url):
                continue
                
            child_urls.append(full_url)

    result = []
    for url in child_urls:
        child_parsed = urllib.parse.urlparse(url)

        # ---------------------------------
        # Check if domain match is required
        # ---------------------------------
        if is_domain_match and child_parsed.netloc != base_parsed.netloc:
            logger.debug(f'Skipping {url} - domain mismatch')
            continue

        # ---------------------------------
        # Check if base path match is required
        # ---------------------------------
        if is_base_path_match and not child_parsed.path.startswith(base_parsed.path):
            logger.debug(f'Skipping {url} - base path mismatch')
            continue

        # ---------------------------------
        # Check valid paths if specified
        # ---------------------------------
        if valid_paths:
            path_match = False
            for valid_path in valid_paths:
                if child_parsed.path.startswith(valid_path):
                    path_match = True
                    break
            if not path_match:
                logger.debug(f'Skipping {url} - not in valid paths')
                continue

        # If we get here, the URL passed all filters
        normalized_url = normalize_url(url)
        if normalized_url not in result:
            result.append(normalized_url)

    logger.debug(f'Found {len(result)} valid child URLs from {len(child_urls)} total links')
    return result


# ------------------
# Worker thread logic
# ------------------
def worker(
    q: object,
    base_url: str,
    max_depth: int,
    already_crawled: set,
    base_dir: str,
    target_links: Union[List[str], None] = DEFAULT_TARGET_LINKS,
    target_content: Union[List[str], None] = None,
    valid_paths: Union[List[str], None] = None,
    is_domain_match: bool = None,
    is_base_path_match: bool = None,
    is_links: Optional[bool] = False,
    render_html: Optional[bool] = DEFAULT_RENDER_HTML
) -> None:

    try:
        while not q.empty():
            depth, url = q.get()
            if depth > max_depth:
                continue
            file_name = '-'.join(re.findall(r'\w+', urllib.parse.urlparse(url).path))
            file_name = 'index' if not file_name else file_name
            file_path = f'{base_dir.rstrip("/") + "/"}{file_name}.md'

            child_urls = crawl(
                url,
                base_url,
                already_crawled,
                file_path,
                target_links,
                target_content,
                valid_paths,
                is_domain_match,
                is_base_path_match,
                is_links,
                render_html
            )
            child_urls = [normalize_url(u) for u in child_urls]
            for child_url in child_urls:
                q.put((depth + 1, child_url))
            time.sleep(1)
    finally:
        # Clean up browser when thread finishes
        if render_html:
            cleanup_thread_browser()


# -----------------
# Thread management
# -----------------
def md_crawl(
        base_url: str,
        max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
        num_threads: Optional[int] = DEFAULT_NUM_THREADS,
        base_dir: Optional[str] = DEFAULT_BASE_DIR,
        target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
        target_content: Union[str, List[str]] = None,
        valid_paths: Union[str, List[str]] = None,
        is_domain_match: Optional[bool] = None,
        is_base_path_match: Optional[bool] = None,
        is_debug: Optional[bool] = False,
        is_links: Optional[bool] = False,
        render_html: Optional[bool] = DEFAULT_RENDER_HTML
) -> None:
    if is_domain_match is False and is_base_path_match is True:
        raise ValueError('❌ Domain match must be True if base match is set to True')

    is_domain_match = DEFAULT_DOMAIN_MATCH if is_domain_match is None else is_domain_match
    is_base_path_match = DEFAULT_BASE_PATH_MATCH if is_base_path_match is None else is_base_path_match

    if not base_url:
        raise ValueError('❌ Base URL is required')

    if isinstance(target_links, str):
        target_links = target_links.split(',') if ',' in target_links else [target_links]

    if isinstance(target_content, str):
        target_content = target_content.split(',') if ',' in target_content else [target_content]

    if isinstance(valid_paths, str):
        valid_paths = valid_paths.split(',') if ',' in valid_paths else [valid_paths]

    if is_debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug('🐞 Debugging enabled')
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f'🕸️ Crawling {base_url} at ⏬ depth {max_depth} with 🧵 {num_threads} threads')

    # Validate the base URL
    if not is_valid_url(base_url):
        raise ValueError('❌ Invalid base URL')

    # Create base_dir if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create thread-safe URL tracker instead of set
    url_tracker = ThreadSafeURLTracker()

    # Create a queue of URLs to crawl
    q = queue.Queue()

    # Add the base URL to the queue
    q.put((0, base_url))

    threads = []

    # Create a thread for each URL in the queue
    for i in range(num_threads):
        t = threading.Thread(
            target=worker,
            args=(
                q,
                base_url,
                max_depth,
                url_tracker,  # Pass url_tracker instead of already_crawled
                base_dir,
                target_links,
                target_content,
                valid_paths,
                is_domain_match,
                is_base_path_match,
                is_links,
                render_html
            )
        )
        threads.append(t)
        t.start()
        logger.debug(f'Started thread {i+1} of {num_threads}')

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Log final statistics
    total_crawled = url_tracker.get_crawled_count()
    logger.info(f'🏁 All threads finished. Total URLs crawled: {total_crawled}')

# Add after the existing imports and before URL validation
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