from fastmcp import FastMCP
import requests
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()




def extract_profile(html: str):

    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text()

    # limit tokens / characters
    text = text[:8000]

    prompt = f"""
Extract structured profile information.

Return ONLY valid JSON:
{{
  "name": "",
  "company": "",
  "role": "",
  "location": "",
  "skills": [],
  "social_links": []
}}

TEXT:
{text}
"""

    llm = ChatGroq(model="llama-3.1-8b-instant")

    response = llm.invoke(prompt,timeout=30)

    return response.content

    #return html[:2000]