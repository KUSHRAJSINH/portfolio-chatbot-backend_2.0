import sys
from tools.crawl_page import crawl_page
from tools.open_page import open_page


def smart_scrape(url: str):
    """
    Try Crawl4AI first.
    If it fails, fallback to Playwright.
    """
    try:
        text = crawl_page(url)
        
        if text and len(text) > 500:
            sys.stderr.write("✅ Crawl4AI success\n")
            return text[:3000]

    except Exception as e:
        sys.stderr.write(f"⚠ Crawl4AI failed: {e}\n")

    sys.stderr.write("🔁 Falling back to Playwright...\n")

    try:
        html = open_page(url)
        return html
    except Exception as e:
        sys.stderr.write(f"❌ Playwright also failed: {e}\n")

    return None