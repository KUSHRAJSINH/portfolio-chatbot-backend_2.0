from tools.search_person import search_person
from tools.smart_scrape import smart_scrape
from tools.extract_profile import extract_profile


name=input("enter person name: ")

print("🔍 Searching...")

links = search_person(name)

print(links)

url = links[0]

print("\n🌐 Scraping:", url)

page = smart_scrape(url)

print("\n📄 Page length:", len(page))

print("\n🧠 Extracting profile...")

profile = extract_profile(page)

print("\nResult:\n", profile)