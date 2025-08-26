import os, io, csv, time, random, hashlib
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter, Retry
from PIL import Image, UnidentifiedImageError
from ddgs import DDGS

from tqdm import tqdm

# Config
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"}
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20
MAX_WORKERS = 6  # be nice; keep low to moderate
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB hard cap
MIN_WH = 128  # min width/height to keep
TARGET_MAX_SIDE = 1024  # downscale if either side > this (0 to disable)
SAVE_AS_JPEG = True  # convert everything to JPEG (RGB) for consistency
JPEG_QUALITY = 95
ACCEPT_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"}
SAFESEARCH = "Off"  # "Off" | "Moderate" | "Strict" (duckduckgo)
REGION = "wt-wt"  # worldwide

# Pillow resampling fallback (Pillow < 9.1)
try:
    RESAMPLE = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except Exception:
    RESAMPLE = Image.LANCZOS


# Helpers
def make_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update(HEADERS)
    return sess


def safe_name(s: str) -> str:
    s = "".join(c for c in s if c.isalnum() or c in (" ", "-", "_")).strip()
    return s.replace(" ", "_")


def pick_ext_from_headers_or_url(ct: str, url: str) -> str:
    ct = (ct or "").lower()
    if "jpeg" in ct or "jpg" in ct:
        return "jpg"
    if "png" in ct:
        return "png"
    if "gif" in ct:
        return "gif"
    if "webp" in ct:
        return "webp"
    if "bmp" in ct:
        return "bmp"
    if "tiff" in ct:
        return "tiff"
    # fallback: URL path
    path_ext = urlparse(url).path.split(".")[-1].lower()
    return path_ext if path_ext in ACCEPT_EXTS else "jpg"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def validate_and_prepare(image_bytes: bytes):
    """Open, optionally downscale, ensure RGB if saving JPEG, return PIL.Image"""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.load()
        w, h = img.size
        if w < MIN_WH or h < MIN_WH:
            raise ValueError(f"too small: {w}x{h}")
        if TARGET_MAX_SIDE and (w > TARGET_MAX_SIDE or h > TARGET_MAX_SIDE):
            img.thumbnail((TARGET_MAX_SIDE, TARGET_MAX_SIDE), RESAMPLE)
        if img.mode == "P":
            img = img.convert("RGBA")
        if SAVE_AS_JPEG and img.mode != "RGB":
            img = img.convert("RGB")
        return img.copy()  # return a decoupled image


def download_one(
    sess: requests.Session,
    row: dict,
    out_dir: str,
    prefix: str,
    idx: int,
    seen_hashes: set,
    meta_writer,
) -> bool:
    url = row.get("image") or row.get("thumbnail") or row.get("url")
    if not url:
        return False

    try:
        # HEAD first if server supports it (to check size/type)
        head = sess.head(
            url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=True
        )
        if head.status_code >= 400:
            # some servers block HEAD; continue with GET
            pass
        else:
            cl = head.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) > MAX_IMAGE_BYTES:
                return False
            ct = (head.headers.get("Content-Type") or "").lower()
            if ct and not ct.startswith("image/"):
                # We'll still GET; some servers don't send correct CT in HEAD
                pass

        # GET
        r = sess.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True)
        r.raise_for_status()

        ct = r.headers.get("Content-Type", "").lower()
        if ct and not ct.startswith("image/"):
            # Skip known non-image responses (HTML pages, etc.)
            return False

        # Read up to MAX_IMAGE_BYTES
        chunks = []
        total = 0
        for chunk in r.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                return False
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            return False

        # Deduplicate by hash
        h = sha256_bytes(data)
        if h in seen_hashes:
            return False

        # Validate image
        try:
            img = validate_and_prepare(data)
        except (UnidentifiedImageError, OSError, ValueError):
            return False

        # Decide extension and filename
        ext = "jpg" if SAVE_AS_JPEG else pick_ext_from_headers_or_url(ct, url)
        base = f"{prefix}_{idx:04d}"
        fname = f"{base}.{ext}"
        fpath = os.path.join(out_dir, fname)

        # Save
        if SAVE_AS_JPEG:
            img.save(fpath, "JPEG", quality=95, optimize=True)
        else:
            img.save(fpath)

        # Record metadata
        meta_writer.writerow(
            {
                "filename": fname,
                "query": prefix,
                "source_url": url,
                "content_type": ct or "",
                "sha256": h,
                "width": img.width,
                "height": img.height,
            }
        )

        seen_hashes.add(h)
        return True

    except requests.RequestException:
        return False
    except Exception:
        return False


def search_images(query: str, max_results: int):
    # ddg sometimes benefits from tiny stagger to avoid 429s
    time.sleep(random.uniform(0.3, 0.8))
    with DDGS() as ddgs:
        # docs: https://pypi.org/project/duckduckgo-search/
        # We can pass region & safesearch to guide results.
        return list(
            ddgs.images(
                query,
                region=REGION,
                safesearch=SAFESEARCH,
                max_results=max_results,
            )
        )


def download_images(query: str, folder: str, max_images: int = 50) -> int:
    os.makedirs(folder, exist_ok=True)
    # metadata CSV
    meta_path = os.path.join(folder, "_metadata.csv")
    is_new = not os.path.exists(meta_path)
    meta_file = open(meta_path, "a", newline="", encoding="utf-8")
    meta_writer = csv.DictWriter(
        meta_file,
        fieldnames=[
            "filename",
            "query",
            "source_url",
            "content_type",
            "sha256",
            "width",
            "height",
        ],
    )
    if is_new:
        meta_writer.writeheader()

    try:
        results = search_images(
            query, max_results=max_images * 2
        )  # over-fetch to offset skips
    except Exception as e:
        print(f"Error with DDGS search for '{query}': {e}")
        meta_file.close()
        return 0

    print(f"Found {len(results)} candidates for '{query}'")

    sess = make_session()
    prefix = safe_name(query)
    seen_hashes = set()

    # Threaded downloads (gentle)
    tasks = []
    successes = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, row in enumerate(results):
            tasks.append(
                ex.submit(
                    download_one, sess, row, folder, prefix, i, seen_hashes, meta_writer
                )
            )

        for f in tqdm(
            as_completed(tasks), total=len(tasks), desc=f"Downloading {query}"
        ):
            ok = f.result()
            if ok:
                successes += 1
            # stop early if we’ve got enough
            if successes >= max_images:
                break

    meta_file.close()
    print(f"Downloaded {successes} images for '{query}' into {folder}")
    return successes


# ---------- Main ----------
if __name__ == "__main__":
    categories = [
        "dog",
        "cat",
        "elephant",
        "snake",
        "parrot",
        "butterfly",
        "fish",
        "fox",
        "lion",
    ]
    os.makedirs("dataset", exist_ok=True)

    total = 0
    for animal in categories:
        print(f"\n--- Processing {animal} ---")
        got = download_images(
            animal, folder=os.path.join("dataset", animal), max_images=50
        )
        total += got
        time.sleep(1.0)  # polite gap

    print("\n=== Download Complete ===")
    print(f"Total images downloaded: {total}")
    print("Dataset saved in: ./dataset/")
