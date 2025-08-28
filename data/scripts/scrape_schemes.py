import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import os
import re


def fetch_html(url: str) -> str:
    """
    Fetches HTML content from a URL and returns the raw HTML string.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def sanitize_filename(name: str) -> str:
    # Remove invalid filename characters and replace spaces with underscores
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")


def save_scheme_html(scheme: Dict[str, str], html_content: str, directory: str) -> None:
    filename = sanitize_filename(scheme['name']) + ".html"
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)


def load_scheme_urls(filepath: str) -> List[Dict[str, str]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def scrape_and_save_all_schemes_html(schemes: List[Dict[str, str]], directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for scheme in schemes:
        try:
            html_content = fetch_html(scheme['url'])
            save_scheme_html(scheme, html_content, directory)
            print(f"Saved HTML: {scheme['name']}")
        except Exception as e:
            print(f"Error scraping {scheme['name']}: {e}")


def main():
    scheme_urls_path = "data/raw/schema_urls.json"
    output_directory = "data/raw/schemes_html"
    schemes = load_scheme_urls(scheme_urls_path)
    scrape_and_save_all_schemes_html(schemes, output_directory)


if __name__ == "__main__":
    main()

