# input.py
import os
import io
import math
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# CẤU HÌNH
API_URL = "http://localhost:5000/search"
QUERY_IMAGE = (
    "testimage/cat.jpeg"  # chọn ảnh test, có thể thay bằng các ảnh khác trong testimage
)

DATASET_ROOT = "./dataset"
TOP_K = 10  # số ảnh kết quả muốn hiển thị (None = tất cả)
MAX_COLS = 5  # số cột tối đa trong grid


def resolve_local_path(p: str) -> str | None:
    """Chuẩn hoá & ghép đường dẫn tương đối thành tuyệt đối nếu cần."""
    if not p:
        return None
    p = p.replace("\\", "/")
    if DATASET_ROOT is not None and not os.path.isabs(p):
        p = os.path.join(DATASET_ROOT, p)
    return os.path.normpath(p)


def read_local_image(path: str):
    """Đọc ảnh local an toàn; trả về mảng ảnh hoặc None nếu lỗi."""
    try:
        return mpimg.imread(path)
    except Exception:
        # Fallback qua PIL nếu mpimg lỗi (một số định dạng/mode)
        try:
            with Image.open(path) as im:
                return mpimg.pil_to_array(im.convert("RGB"))
        except Exception as e:
            print(f"[WARN] Không thể đọc ảnh: {path} -> {e}")
            return None


def main():
    # Gửi ảnh query lên API
    with open(QUERY_IMAGE, "rb") as f:
        files = {"image": f}
        resp = requests.post(API_URL, files=files)

    print("HTTP Status:", resp.status_code)

    # Thử parse JSON; nếu không phải JSON, in text và thoát
    try:
        data = resp.json()
    except Exception:
        print("Response text:", resp.text[:500])
        return

    print(
        "API response (tóm tắt):",
        str(data)[:300],
        "..." if len(str(data)) > 300 else "",
    )

    # Chuẩn hoá kết quả về dạng list[ {path, score?} ]
    if isinstance(data, list):
        results = data
    elif (
        isinstance(data, dict)
        and "results" in data
        and isinstance(data["results"], list)
    ):
        results = data["results"]
    else:
        print("[WARN] Không nhận diện được cấu trúc kết quả. Mặc định rỗng.")
        results = []

    # Giới hạn Top-K nếu cần
    if isinstance(TOP_K, int) and TOP_K > 0:
        results = results[:TOP_K]

    # Chuẩn bị danh sách item để vẽ: Query + Kết quả
    items = [{"title": "Query", "path": QUERY_IMAGE, "score": None}]

    for r in results:
        # mỗi r kỳ vọng là dict, có thể chỉ có "path", có thể kèm "score"
        if isinstance(r, dict):
            path = (
                r.get("path") or r.get("image") or r.get("file")
            )  # linh động tên khóa
            score = r.get("score", None)
        else:
            # Nếu server trả về chỉ là string path
            path, score = str(r), None

        local_path = resolve_local_path(path)
        title = (
            f"score: {score:.2f}"
            if isinstance(score, (int, float))
            else os.path.basename(local_path or "")
        )
        items.append({"title": title, "path": local_path, "score": score})

    # Vẽ grid
    total = len(items)
    if total == 0:
        print("Không có gì để hiển thị.")
        return

    cols = min(total, MAX_COLS if MAX_COLS >= 1 else 5)
    rows = math.ceil(total / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))

    for idx, it in enumerate(items, start=1):
        ax = plt.subplot(rows, cols, idx)
        img = read_local_image(it["path"]) if it["path"] else None
        if img is not None:
            ax.imshow(img)
            ax.set_title(it["title"], fontsize=11)
        else:
            ax.text(
                0.5,
                0.5,
                f"Không đọc được ảnh\n{os.path.basename(it['path'] or '')}",
                ha="center",
                va="center",
                wrap=True,
            )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
