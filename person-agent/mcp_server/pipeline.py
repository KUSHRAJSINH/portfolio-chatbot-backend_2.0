from tools.search_person import search_person
from tools.smart_scrape import smart_scrape
from tools.extract_profile import extract_profile
from tools.github_tools import extract_github_username, get_github_repos, check_github_user_exists

def run_pipeline(name: str):
    print("🔍 Step 1: Searching...")
    links = search_person(name)
    print("Links:", links)

    # STEP 2: GitHub
    print("\n💻 Step 2: Extract GitHub...")
    github_username = extract_github_username(links)
    
    # Verification and Confirmation
    found_correct = False
    if github_username:
        # Check if it exists
        if check_github_user_exists(github_username):
            confirm = input(f"I found this GitHub profile: https://github.com/{github_username}. Is this correct? (y/n/skip): ").strip().lower()
            if confirm == 'y':
                found_correct = True
            elif confirm == 'skip':
                github_username = None
        else:
            print(f"⚠ GitHub user {github_username} not found (404).")

    if not found_correct:
        ui_link = input(f"Please provide the CORRECT GitHub profile link for {name} (or press Enter to skip): ").strip()
        if ui_link:
            github_username = extract_github_username(ui_link)
        else:
            github_username = None

    print("GitHub Username:", github_username)

    github_data = []
    if github_username:
        try:
            github_data = get_github_repos(github_username)
            print("Top repos:", github_data)
        except Exception as e:
            print("❌ GitHub error:", e)

    # STEP 3: Scraping
    print("\n🌐 Step 3: Scraping...")
    
    linkedin_link = next((link for link in links if "linkedin.com/in/" in link), None)
    
    if linkedin_link:
        confirm_li = input(f"I found this LinkedIn profile: {linkedin_link}. Is this correct? (y/n/skip): ").strip().lower()
        if confirm_li == 'n':
            linkedin_link = None
        elif confirm_li == 'skip':
            linkedin_link = "skip"

    if not linkedin_link:
        li_link = input(f"Please provide LinkedIn profile link for {name} (or press Enter to skip): ").strip()
        if li_link:
            linkedin_link = li_link
            links.insert(0, li_link)
    elif linkedin_link == "skip":
        linkedin_link = None
    else:
        # Move the confirmed link to the front
        if linkedin_link in links:
            links.remove(linkedin_link)
        links.insert(0, linkedin_link)

    profile_data = None



    for link in links:
        print(f"\nTrying: {link}")

        try:
            page = smart_scrape(link)

            if not page:
                print("⚠ No content")
                continue

            print("✅ Scraped, length:", len(page))

            print("🧠 Extracting profile...")
            profile_data = extract_profile(page)

            print("✅ Extraction done")
            break

        except Exception as e:
            print("❌ Error:", e)

    print("\n🎯 FINAL OUTPUT:")
    return {
        "name": name,
        "github": github_username,
        "projects": github_data,
        "profile": profile_data
    }

