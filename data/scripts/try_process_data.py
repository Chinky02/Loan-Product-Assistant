import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

BASE_URL = "https://bankofmaharashtra.in/retail-loans"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LoanAssistantBot/1.0; +https://github.com/your-repo)"
}


def fetch(url):
    """Fetch a URL with headers and return BeautifulSoup object"""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def clean_text(text):
    """Clean extra whitespace"""
    return " ".join(text.split())


def extract_links(html):
    """Extract all loan product links from the retail loans page"""
    soup = BeautifulSoup(html, "html.parser")
    product_links = []
    for a in soup.select("a"):  # general selector, refine if needed
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if href and ("loan" in href.lower() or "scheme" in href.lower()):
            product_links.append({
                "name": text,
                "url": urljoin(BASE_URL, href)
            })
    return product_links


def parse_product_page(url, html):
    """Parse loan product page and return structured data"""
    soup = BeautifulSoup(html, "html.parser")
    
    title = soup.find("h1")
    title = clean_text(title.get_text()) if title else "Unknown Loan Product"
    
    # Extract description
    desc_tag = soup.find("p")
    description = clean_text(desc_tag.get_text()) if desc_tag else ""
    
    # Extract all tables
    tables_data = []
    for table in soup.find_all("table"):
        headers = [clean_text(th.get_text()) for th in table.find_all("th")]
        rows = []
        for row in table.find_all("tr"):
            cells = [clean_text(td.get_text()) for td in row.find_all("td")]
            if cells:
                rows.append(cells)
        tables_data.append({"headers": headers, "rows": rows})
    
    data = {
        "product_name": title,
        "source_url": url,
        "crawled_at": datetime.utcnow().isoformat(),
        "description": description,
        "tables": tables_data
    }
    return data


def main():
    print(f"Fetching landing page: {BASE_URL}")
    html = fetch(BASE_URL)
    links = extract_links(html)
    print(f"Found {len(links)} product links")

    for link in links:
        url = link["url"]
        name = link["name"].replace(" ", "_").replace("/", "_")
        print(f"Fetching {name} → {url}")

        try:
            page_html = fetch(url)
            
            # Save raw HTML
            raw_path = os.path.join(RAW_DIR, f"{name}.html")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(page_html)
            
            # Parse + Save processed data
            data = parse_product_page(url, page_html)
            out_path = os.path.join(PROCESSED_DIR, f"{name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {name}")
        
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
        
        time.sleep(1)  # polite delay


if __name__ == "__main__":
    main()
