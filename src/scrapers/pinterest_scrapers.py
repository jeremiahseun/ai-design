import requests
from bs4 import BeautifulSoup
import time

def get_image_url(pin_url: str, max_retries: int = 3, retry_delay: float = 2.0) -> str:
    """
    Extract the direct image URL from a Pinterest pin link.

    Args:
        pin_url: Pinterest pin URL
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)

    Returns:
        Image URL or None if extraction fails
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(pin_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Pinterest always embeds the image URL in og:image meta tag
            meta_img = soup.find("meta", property="og:image")

            if not meta_img or not meta_img.get("content"):
                print(f"⚠️  Could not find image for: {pin_url}")
                return None

            return meta_img["content"]

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"⚠️  Rate limited. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"❌ HTTP Error {e.response.status_code}: {pin_url}")
                return None

        except requests.exceptions.Timeout:
            print(f"⏱️  Timeout on attempt {attempt + 1}/{max_retries}: {pin_url}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        except Exception as e:
            print(f"❌ Failed to load page: {pin_url} — {e}")
            return None

    print(f"❌ Failed after {max_retries} attempts: {pin_url}")
    return None



def batch_extract(file_path: str, rate_limit_delay: float = 2.0):
    """
    Reads a file containing Pinterest URLs (one per line)
    and prints their extracted image URLs.

    Args:
        file_path: Path to file containing Pinterest pin URLs
        rate_limit_delay: Delay between requests in seconds
    """

    with open(file_path, "r") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    print(f"📋 Found {len(urls)} URLs to process")
    print(f"⏱️  Rate limit: {rate_limit_delay}s between requests\n")

    results = []
    success_count = 0

    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] 🔎 Extracting: {url}")
        img_url = get_image_url(url)

        if img_url:
            print(f"         ✅ Image URL: {img_url}")
            results.append(img_url)
            success_count += 1
        else:
            results.append("ERROR")

        # Rate limiting between requests
        if idx < len(urls):  # Don't delay after the last one
            time.sleep(rate_limit_delay)
        print()  # Empty line for readability

    # Save results to output.txt
    with open("output.txt", "w") as f:
        for r in results:
            f.write(r + "\n")

    print("=" * 60)
    print(f"🎉 Done! {success_count}/{len(urls)} images extracted successfully")
    print(f"📁 Results saved to: output.txt")
    print("=" * 60)



if __name__ == "__main__":
    print("Pinterest Image Extractor")
    print("--------------------------")
    print("Provide a text file containing one Pinterest pin URL per line.")
    file_path = input("Enter file name (e.g. pins.txt): ")
    batch_extract(file_path)
