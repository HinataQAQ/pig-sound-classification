# tools/fix_panns_windows_ckpt.py
from pathlib import Path
import urllib.request
import hashlib

URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
# Zenodo 页面给出的 md5（可选校验）
MD5_EXPECT = "541141fa2ee191a88f24a3219fff024e"

def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    target = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size > 10_000_000:
        print(f"[OK] already exists: {target} ({target.stat().st_size/1024/1024:.1f} MB)")
    else:
        print("[INFO] downloading checkpoint from Zenodo ...")
        print("      URL   =", URL)
        print("      TO    =", target)
        urllib.request.urlretrieve(URL, target)
        print(f"[OK] downloaded: {target} ({target.stat().st_size/1024/1024:.1f} MB)")

    # optional md5 check
    got = md5sum(target)
    print("[INFO] md5 =", got)
    print("[INFO] md5_expected =", MD5_EXPECT)
    print("[OK] md5_match =", (got == MD5_EXPECT))

if __name__ == "__main__":
    main()
