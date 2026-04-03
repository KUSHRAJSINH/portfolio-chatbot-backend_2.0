from playwright.sync_api import sync_playwright
from fastmcp import FastMCP


def open_page(url: str):
    """
    Opens a webpage and returns the HTML content.
    """

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        page.goto(url, timeout=60000)

        content = page.content()

        browser.close()

    return content