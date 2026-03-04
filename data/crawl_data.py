import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from readability import Document
import trafilatura
from playwright.sync_api import sync_playwright
import time


# =========================
# Core document schema
# =========================
def _make_doc(
    text: str,
    source: str,
    url: str,
    method: str,
    confidence: float
) -> Dict:
    return {
        "text": text,
        "source": source,   # domain
        "url": url,
        "method": method,   # requests | playwright | trafilatura
        "confidence": confidence
    }


# =========================
# Fetch HTML
# =========================
def fetch_html(url: str, use_js: bool = False, timeout: int = 15) -> str:
    """
    Fetch raw HTML.
    - use_js=True → dùng Playwright (JS-rendered sites)
    """
    if not use_js:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text

    # JS-rendered fallback
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=timeout * 1000)
        time.sleep(2)  # chờ JS load
        html = page.content()
        browser.close()
        return html


# =========================
# Extract main content
# =========================
def extract_main_text(html: str, url: str) -> str:
    """
    Ưu tiên:
    1. trafilatura (chuẩn research)
    2. readability-lxml
    3. fallback BeautifulSoup
    """

    # --- Layer 1: Trafilatura ---
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=False
    )
    if extracted and len(extracted.strip()) > 200:
        return extracted

    # --- Layer 2: Readability ---
    doc = Document(html)
    readable_html = doc.summary()
    soup = BeautifulSoup(readable_html, "html.parser")
    text = soup.get_text("\n")
    if len(text.strip()) > 200:
        return text

    # --- Layer 3: Brutal fallback ---
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text("\n")


# =========================
# Crawl single URL
# =========================
def crawl_url(url: str, use_js: bool = False) -> Dict | None:
    try:
        html = fetch_html(url, use_js=use_js)
        text = extract_main_text(html, url)

        if not text or len(text.strip()) < 200:
            return None

        source = url.split("//")[-1].split("/")[0]

        return _make_doc(
            text=text,
            source=source,
            url=url,
            method="playwright" if use_js else "requests",
            confidence=0.9 if not use_js else 0.85
        )

    except Exception as e:
        print(f"[ERROR] Crawl failed: {url} → {e}")
        return None


# =========================
# Crawl multiple URLs
# =========================
def crawl_urls(
    urls: List[str],
    use_js: bool = False,
    sleep: float = 1.0
) -> List[Dict]:
    documents = []
    for url in urls:
        doc = crawl_url(url, use_js=use_js)
        if doc:
            documents.append(doc)
        time.sleep(sleep)  # tránh bị block
    return documents
