import os
import requests
import concurrent.futures
from urllib.parse import urlparse
import time

def download_image(url, save_dir):
    try:
        # Get filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            return f"Skipped {url}: No filename found"

        save_path = os.path.join(save_dir, filename)

        if os.path.exists(save_path):
            return f"Skipped {url}: Already exists"

        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"Downloaded {url}"
    except Exception as e:
        return f"Error downloading {url}: {e}"

def main():
    urls_file = 'pinterest_urls.txt'
    save_dir = 'images'

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Read URLs
    if not os.path.exists(urls_file):
        print(f"Error: {urls_file} not found.")
        return

    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to download.")

    start_time = time.time()
    downloaded_count = 0
    error_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(download_image, url, save_dir): url for url in urls}

        from tqdm import tqdm
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls), unit="img"):
            url = future_to_url[future]
            try:
                result = future.result()
                if "Error" in result:
                    error_count += 1
                    # print(result) # Optional: print errors
                else:
                    downloaded_count += 1
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')
                error_count += 1

    end_time = time.time()
    duration = end_time - start_time

    print(f"\nDownload complete.")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Downloaded: {downloaded_count}")
    print(f"Errors: {error_count}")
    print(f"Images saved to: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
