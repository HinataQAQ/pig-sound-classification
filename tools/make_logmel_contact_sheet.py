
import math
import argparse
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--thumb_w", type=int, default=420)
    ap.add_argument("--thumb_h", type=int, default=180)
    ap.add_argument("--pad", type=int, default=12)
    ap.add_argument("--max_images", type=int, default=24)
    args = ap.parse_args()

    inp = Path(args.input_dir)
    imgs = [p for p in sorted(inp.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs = imgs[:args.max_images]
    if not imgs:
        raise FileNotFoundError(f"No images found in {inp}")

    cols = max(1, args.cols)
    rows = math.ceil(len(imgs) / cols)
    W = cols * args.thumb_w + (cols + 1) * args.pad
    H = rows * args.thumb_h + (rows + 1) * args.pad

    board = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(board)

    for idx, p in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x = args.pad + c * (args.thumb_w + args.pad)
        y = args.pad + r * (args.thumb_h + args.pad)

        img = Image.open(p).convert("RGB")
        img.thumbnail((args.thumb_w, args.thumb_h))
        canvas = Image.new("RGB", (args.thumb_w, args.thumb_h), (255, 255, 255))
        ox = (args.thumb_w - img.width) // 2
        oy = (args.thumb_h - img.height) // 2
        canvas.paste(img, (ox, oy))
        canvas = ImageOps.expand(canvas, border=1, fill=(180, 180, 180))
        board.paste(canvas, (x, y))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    board.save(out)
    print(f"[OK] wrote board -> {out}")


if __name__ == "__main__":
    main()
