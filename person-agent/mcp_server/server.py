from fastmcp import FastMCP
import os
import sys

# Ensure the tools directory is in the path for relative imports within tools
tools_path = os.path.join(os.path.dirname(__file__), "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)

from tools.extract_profile import extract_profile
from tools.smart_scrape import smart_scrape
from tools.search_person import search_person
from tools.github_tools import extract_github_username, get_github_repos


mcp = FastMCP("person-search")


@mcp.tool
def get_github(username: str):
    """Get top GitHub repositories of a user."""
    return get_github_repos(username)


@mcp.tool
def search(name: str) -> list[str]:
    """Search for a person and return relevant links."""
    return search_person(name)

@mcp.tool
def scrape(url: str) -> str:
    """Scrape a webpage and return content."""
    return smart_scrape(url)

@mcp.tool
def extract(html: str) -> str:
    """Extract profile info from text."""
    return extract_profile(html)

if __name__ == "__main__":
    mcp.run()
