from bs4 import BeautifulSoup
import json
import os

def extract_scheme_data(html_path: str, scheme_url: str) -> dict:
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Name
    name = ""
    name_tag = soup.find("h1")
    if not name_tag:
        # Try alternative: title tag or h2
        name_tag = soup.find("title") or soup.find("h2")
    if name_tag:
        name = name_tag.get_text(strip=True)

    # Description
    description = ""
    desc_tag = soup.find("div", class_="col-lg maincontent")
    if not desc_tag:
        # Try alternative: first <p> in main container
        main_container = soup.find("div", class_="container")
        if main_container:
            desc_tag = main_container.find("p")
    if desc_tag:
        p_tag = desc_tag.find("p") if desc_tag.find("p") else desc_tag
        description = p_tag.get_text(" ", strip=True) if p_tag else ""

    # Features
    features = []
    for ul in soup.find_all("ul"):
        if "feature" in ul.get("class", []) or "fblist" in ul.get("class", []) or "normlist" in ul.get("class", []):
            features += [li.get_text(strip=True) for li in ul.find_all("li")]
    # Remove duplicates
    features = list(dict.fromkeys(features))

    # Eligibility
    eligibility = []
    elig_div = soup.find("div", id="pane-elig")
    if elig_div:
        normlist_elig = elig_div.find("ul", class_="normlist")
        if normlist_elig:
            eligibility = [li.get_text(strip=True) for li in normlist_elig.find_all("li")]
        else:
            p_elig = elig_div.find("p")
            if p_elig:
                eligibility = [p_elig.get_text(strip=True)]
    else:
        # Try alternative: look for eligibility in features
        eligibility = [f for f in features if "eligib" in f.lower()]

    # Interest Rate
    interest_rate = ""
    ir_div = soup.find("div", id="pane-ir")
    if ir_div:
        h2 = ir_div.find("h2")
        if h2 and "%" in h2.get_text():
            interest_rate = h2.get_text(strip=True)
        else:
            # Try to find any text with %
            for tag in ir_div.find_all(["h2", "p", "span"]):
                if "%" in tag.get_text():
                    interest_rate = tag.get_text(strip=True)
                    break
    if not interest_rate:
        # Try to find in features
        for feat in features:
            if "%" in feat:
                interest_rate = feat
                break

    # Loan Amount & Tenure
    tenure = ""
    loan_amount = ""
    for feat in features:
        if "tenure" in feat.lower():
            tenure = feat
        if "loan amount" in feat.lower():
            loan_amount = feat

    # Processing Fee
    processing_fee = ""
    for feat in features:
        if "processing fee" in feat.lower():
            processing_fee = feat

    # Documents Required
    documents_required = []
    doc_div = soup.find("div", id="pane-dr")
    if doc_div:
        doc_ul = doc_div.find("ul", class_="normlist")
        if doc_ul:
            documents_required = [li.get_text(strip=True) for li in doc_ul.find_all("li")]
            for sublist in doc_ul.find_all("ul", class_="sublist"):
                documents_required += [li.get_text(strip=True) for li in sublist.find_all("li")]
        else:
            # Try all <li> in doc_div
            documents_required = [li.get_text(strip=True) for li in doc_div.find_all("li")]
    else:
        # Try to find in features
        documents_required = [f for f in features if "document" in f.lower()]

    # How to Apply
    how_to_apply = []
    apply_div = soup.find("div", id="pane-hta")
    if apply_div:
        p_tags = apply_div.find_all("p")
        for p in p_tags:
            txt = p.get_text(strip=True)
            if txt:
                how_to_apply.append(txt)
        for a in apply_div.find_all("a"):
            how_to_apply.append(a.get_text(strip=True))
    else:
        # Try to find in features
        how_to_apply = [f for f in features if "apply" in f.lower()]

    # FAQs
    faqs = []
    faq_div = soup.find("div", id="pane-faq")
    if faq_div:
        for card in faq_div.find_all("div", class_="card"):
            q_btn = card.find("button")
            ans_div = card.find("div", class_="card-body")
            if q_btn and ans_div:
                question = q_btn.get_text(strip=True)
                answer = ans_div.get_text(" ", strip=True)
                faqs.append({"question": question, "answer": answer})
    else:
        # Try to find FAQ sections
        for faq in soup.find_all(string=lambda text: "faq" in text.lower()):
            faqs.append({"question": faq, "answer": ""})

    # Compose result
    result = {
        "name": name,
        "description": description,
        "url": scheme_url,
        "features": features,
        "eligibility": eligibility,
        "interest_rate": interest_rate if interest_rate else "Not found",
        "loan_amount": loan_amount if loan_amount else "Not found",
        "tenure": tenure if tenure else "Not found",
        "processing_fee": processing_fee if processing_fee else "Not found",
        "documents_required": documents_required,
        "how_to_apply": how_to_apply,
        "faqs": faqs
    }
    return result

def load_scheme_urls(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_all_html_files(html_dir: str, scheme_urls_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    schemes = load_scheme_urls(scheme_urls_path)
    url_map = {scheme['name'].replace(" ", "_"): scheme['url'] for scheme in schemes}
    for filename in os.listdir(html_dir):
        if filename.endswith('.html'):
            scheme_name = filename.replace('.html', '')
            html_path = os.path.join(html_dir, filename)
            scheme_url = url_map.get(scheme_name, "")
            data = extract_scheme_data(html_path, scheme_url)
            json_path = os.path.join(output_dir, filename.replace('.html', '.json'))
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Processed: {filename}")

# Example usage:
if __name__ == "__main__":
    html_dir = "data/raw/schemes_html"
    scheme_urls_path = "data/raw/schema_urls.json"
    output_dir = "data/raw/schemes_json"
    process_all_html_files(html_dir, scheme_urls_path, output_dir)