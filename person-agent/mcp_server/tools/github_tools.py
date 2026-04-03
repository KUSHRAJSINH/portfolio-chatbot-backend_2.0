import requests

def extract_github_username(links):
    if isinstance(links, str):
        links = [links]
    for link in links:
        if "github.com/" in link:
            parts = link.split("github.com/")
            username = parts[1].split("/")[0]
            if username and username != "topics":
                return username
    return None



def check_github_user_exists(username: str):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    return response.status_code == 200


def get_github_repos(username: str):
    url = f"https://api.github.com/users/{username}/repos"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")

    repos = response.json()
    if not isinstance(repos, list):
        return []

    repos = sorted(repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)

    top_repos = []

    for repo in repos[:5]:
        top_repos.append({
            "name": repo.get("name"),
            "description": repo.get("description"),
            "stars": repo.get("stargazers_count"),
            "url": repo.get("html_url"),
            "language": repo.get("language")
        })

    return top_repos



