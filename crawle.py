import requests
from bs4 import BeautifulSoup
import time
import json
import sys

BASE_URL = "https://github.com/pytorch/pytorch/releases"

def fetch_releases_page(page):
    url = f"{BASE_URL}?page={page}"
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    resp.raise_for_status()
    return resp.text


def parse_release_entries(html):
    soup = BeautifulSoup(html, "html.parser")  

    entries = soup.select("section")
    results = []

    for entry in entries:
        version_link = entry.select_one("a.Link--primary")
        if not version_link:
            continue

        version = version_link.text.strip()

        time_tag = entry.select_one("relative-time")
        published_at = time_tag["datetime"] if time_tag else None

        results.append({
            "version": version,
            "published_at": published_at,
        })

    return results


def main():
    out_file = "pytorch_releases_page1_4.jsonl"

    with open(out_file, "w", encoding="utf-8") as fout:
        for page in range(1, 5):
            html = fetch_releases_page(page)
            # print(html)
            

            # ✅ 修正点：函数名对齐
            records = parse_release_entries(html)
            # print(records)
            # sys.exit()
            for rec in records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            time.sleep(1)

    print(f"Done. Data saved to {out_file}")


if __name__ == "__main__":
    main()
