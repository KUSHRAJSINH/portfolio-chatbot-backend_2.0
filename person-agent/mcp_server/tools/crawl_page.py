import asyncio
import logging
import sys
import contextlib
from crawl4ai import AsyncWebCrawler

# Silence crawl4ai logging to prevent MCP protocol corruption
logging.getLogger("crawl4ai").setLevel(logging.ERROR)

async def crawl(url:str):
    """
    Crawl webpage and return clean markdown text.
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            bypass_cache=True
        )

        if result.success:
            return result.markdown
        else:
            return None


def crawl_page(url: str):
    """
    Sync wrapper for MCP tool usage.
    """
    with contextlib.redirect_stdout(sys.stderr):
        return asyncio.run(crawl(url))
    