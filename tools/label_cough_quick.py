# -*- coding: utf-8 -*-
# tools/label_cough_quick.py
# 功能：快速试听 + 按键分拣（Windows）
# 依赖：pip install simpleaudio
# 热键：C=咳嗽  O=其他  D=丢弃  空格=重放  S=跳过  U=撤销  Q=退出

import argparse
import os
from pathlib import Path
import shutil
import time
import wave
import sys

try:
    import msvcrt  # Windows 单键读取
except ImportError:
    print("本脚本仅支持 Windows（需要 msvcrt）。")
    sys.exit(1)

import simpleaudio as sa


def human_size(bytes_val: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    s = float(bytes_val)
    i = 0
    while s >= 1024 and i < len(units) - 1:
        s /= 1024
        i += 1
    return f"{s:.1f} {units[i]}"


def wav_info(path: Path):
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            ch = wf.getnchannels()
            dur = nframes / float(sr) if sr > 0 else 0.0
        return sr, ch, dur
    except Exception:
        return None, None, None


def play_wav(path: Path):
    try:
        wav_obj = sa.WaveObject.from_wave_file(str(path))
        play_obj = wav_obj.play()
        play_obj.wait_done()
        return True
    except Exception as e:
        print(f"[ERR] 播放失败：{e}")
        return False


def safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem = dst.stem
        suf = dst.suffix
        k = 1
        while True:
            cand = dst_dir / f"{stem}-{k}{suf}"
            if not cand.exists():
                dst = cand
                break
            k += 1
    try:
        # 同盘快捷 rename；跨盘用 move
        src.rename(dst)
    except Exception:
        shutil.move(str(src), str(dst))
    return dst


def get_key(prompt=""):
    if prompt:
        print(prompt, end="", flush=True)
    ch = msvcrt.getch()
    try:
        key = ch.decode("utf-8").lower()
    except Exception:
        key = ""
    print()  # 换行
    return key


def main():
    ap = argparse.ArgumentParser(description="快速试听 + 按键分拣（cough/other/discard）")
    ap.add_argument("--src", type=str, required=True,
                    help="未标注的 wav 根目录，比如 data/processed/soundwel/misc")
    ap.add_argument("--dst_root", type=str, default="data/processed/soundwel",
                    help="目标根目录（会在下面创建 cough/ other/ _discard/）")
    ap.add_argument("--shuffle", action="store_true", help="是否随机顺序")
    ap.add_argument("--ext", type=str, default=".wav", help="音频扩展名（默认 .wav）")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst_root)
    dst_cough = dst_root / "cough"
    dst_other = dst_root / "other"
    dst_discard = dst_root / "_discard"

    files = sorted([p for p in src_root.rglob(f"*{args.ext}") if p.is_file()])
    if args.shuffle:
        import random
        random.shuffle(files)

    if not files:
        print(f"[WARN] 在 {src_root} 下没有找到 {args.ext} 文件。")
        return

    print("=== 快速分拣开始 ===")
    print("热键：C=咳嗽  O=其他  D=丢弃  空格=重放  S=跳过  U=撤销  Q=退出")
    print(f"源目录：{src_root}")
    print(f"目的目录：{dst_cough} / {dst_other} / {dst_discard}")
    print("===============================================")

    undo_stack = []   # 记录 (dst_path, original_parent)
    total = len(files)
    i = 0
    while i < total:
        path = files[i]
        sr, ch, dur = wav_info(path)
        size_str = human_size(path.stat().st_size)

        print(f"[{i+1}/{total}] {path}")
        if sr:
            print(f"  -> {sr} Hz, {ch} ch, {dur:.2f} s, {size_str}")
        else:
            print(f"  -> {size_str}")

        # 播放一遍
        play_wav(path)

        # 等待按键
        while True:
            key = get_key("请选择【C=咳嗽 O=其他 D=丢弃 空格=重放 S=跳过 U=撤销 Q=退出】：")
            if key == " ":
                play_wav(path)
                continue
            elif key == "c":
                newp = safe_move(path, dst_cough)
                undo_stack.append((newp, path.parent))
                print(f"[OK] -> cough: {newp.name}")
                break
            elif key == "o":
                newp = safe_move(path, dst_other)
                undo_stack.append((newp, path.parent))
                print(f"[OK] -> other: {newp.name}")
                break
            elif key == "d":
                newp = safe_move(path, dst_discard)
                undo_stack.append((newp, path.parent))
                print(f"[OK] -> _discard: {newp.name}")
                break
            elif key == "s":
                print("[SKIP] 跳过当前文件")
                break
            elif key == "u":
                if undo_stack:
                    last_dst, orig_parent = undo_stack.pop()
                    back_to = orig_parent / last_dst.name
                    try:
                        last_dst.rename(back_to)
                    except Exception:
                        shutil.move(str(last_dst), str(back_to))
                    print(f"[UNDO] 撤销最近一次移动：{back_to}")
                    # 撤销后，将 files[i] 指回刚撤销的那一项
                    if back_to not in files:
                        files.insert(i, back_to)
                        total += 1
                else:
                    print("[UNDO] 无可撤销操作")
                continue
            elif key == "q":
                print("[QUIT] 已退出。")
                return
            else:
                print("[HINT] 无效按键；请按 C / O / D / 空格 / S / U / Q。")
                continue

        i += 1

    print("=== 分拣完成 ===")
    print("接下来：1）运行 02 生成清单；2）修改 config.yaml 的 labels；3）训练与评估。")


if __name__ == "__main__":
    main()
