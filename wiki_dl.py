import argparse
import re
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup


STOP_SECTIONS = {
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "bibliography",
}


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_inputs(input_arg: str) -> list[str]:
    if is_url(input_arg):
        return [input_arg]

    path = Path(input_arg)
    if not path.exists():
        raise FileNotFoundError(f"Input is neither a URL nor an existing file: {input_arg}")

    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)

    return urls


def safe_filename_from_url(url: str) -> str:
    name = unquote(url.rstrip("/").split("/")[-1])
    name = re.sub(r"[^\w\-(). ]+", "_", name)
    name = name.replace(" ", "_")
    return f"{name}.txt"


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)  # remove citation markers like [1]
    return " ".join(text.split())


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "wiki-dl/1.0 (simple dataset builder)"
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def parse_wikipedia(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.find("h1")
    title_text = clean_text(title.get_text()) if title else "Untitled"

    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return ""

    output = [
        f"# {title_text}\n",
        f"Source: {url}\n\n",
    ]

    for element in content.find_all(["h2", "h3", "p", "ul"], recursive=True):
        if element.name in {"h2", "h3"}:
            heading = clean_text(element.get_text().replace("[edit]", ""))
            heading_key = heading.lower()

            if heading_key in STOP_SECTIONS:
                break

            prefix = "##" if element.name == "h2" else "###"
            output.append(f"\n{prefix} {heading}\n\n")

        elif element.name == "p":
            text = clean_text(element.get_text())
            if text:
                output.append(text + "\n\n")

        elif element.name == "ul":
            items = []
            for li in element.find_all("li", recursive=False):
                text = clean_text(li.get_text())
                if text:
                    items.append(f"- {text}")
            if items:
                output.append("\n".join(items) + "\n\n")

    return "".join(output).strip() + "\n"


def download(url: str, output_dir: Path, overwrite: bool = False) -> None:
    filename = safe_filename_from_url(url)
    output_path = output_dir / filename

    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return

    html = fetch_html(url)
    text = parse_wikipedia(html, url)

    if not text.strip():
        print(f"No text extracted: {url}")
        return

    output_path.write_text(text, encoding="utf-8")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia pages as clean text files."
    )

    parser.add_argument(
        "input",
        help="A Wikipedia URL or a text file containing one URL per line",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory to save text files. Defaults to current directory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between downloads when using a URL list. Defaults to 1 second.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = load_inputs(args.input)

    for i, url in enumerate(urls):
        try:
            download(url, output_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f"Failed: {url}")
            print(f"Reason: {e}")

        if i < len(urls) - 1:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
