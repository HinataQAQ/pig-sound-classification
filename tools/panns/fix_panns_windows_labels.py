# tools/fix_panns_windows_labels.py
from pathlib import Path
import urllib.request
import importlib.util

LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

def download_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LABELS_URL, path)
    print(f"[OK] downloaded -> {path}")

def main():
    home = Path.home()

    # ① 常见：用户目录下 panns_data
    targets = [
        home / "panns_data" / "class_labels_indices.csv",
        home / "panns_data" / "metadata" / "class_labels_indices.csv",
    ]

    # ② 保险：把文件也放进 site-packages 的 panns_inference 目录附近
    spec = importlib.util.find_spec("panns_inference")
    if spec and spec.origin:
        pkg_dir = Path(spec.origin).parent
        targets += [
            pkg_dir / "class_labels_indices.csv",
            pkg_dir / "metadata" / "class_labels_indices.csv",
            pkg_dir / "resources" / "class_labels_indices.csv",
        ]
        print(f"[INFO] panns_inference package dir = {pkg_dir}")
    else:
        print("[WARN] find_spec('panns_inference') failed: maybe not installed in this python env.")

    # 去重
    uniq = []
    seen = set()
    for t in targets:
        if str(t) not in seen:
            uniq.append(t)
            seen.add(str(t))

    print("[INFO] will download labels csv to:")
    for t in uniq:
        print(" -", t)

    ok = 0
    for t in uniq:
        try:
            download_to(t)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {t} -> {e}")

    # 下载完尝试 import（如果还是失败，会显示缺哪个路径）
    print("\n[TEST] try import panns_inference ...")
    try:
        import panns_inference
        from panns_inference import labels
        print("[OK] panns_inference import OK")
        print("[OK] labels loaded, len(labels) =", len(labels))
    except Exception as e:
        print("[FAIL] import panns_inference still failed:")
        print(" ", repr(e))
        print("提示：把上面 FAIL 的缺失路径那行完整复制给我。")

    print(f"\n[DONE] labels download attempts = {ok}/{len(uniq)}")

if __name__ == "__main__":
    main()
