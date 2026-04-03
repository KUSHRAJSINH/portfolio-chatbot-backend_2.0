import requests
import os
from dotenv import load_dotenv

import contextlib
import sys

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def search_person(name: str):
    """Search for a person on the internet and return relevant links."""
    url = "https://google.serper.dev/search"

    payload = {
        "q": f"{name} github profile OR linkedin profile"
    }


    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    data = response.json()

    # sys.stderr.write(f"SERPER RESPONSE: {data}\n")

    if "organic" not in data:
        raise Exception("Serper API error. Check API key.")

    links = []

    for item in data["organic"]:
        links.append(item["link"])

    return links[:5]