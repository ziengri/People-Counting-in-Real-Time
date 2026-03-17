from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from rfzl4 import RawZstdFrameReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read RFZST container sequentially")
    parser.add_argument("--input", required=True, help="Path to input .rfzst file")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    parser.add_argument("--strict", action="store_true", help="Fail on corrupted chunks")
    parser.add_argument("--window-name", default="RFZST Player", help="OpenCV window title")
    return parser.parse_args()


def process_frame(timestamp_ns: int, frame: np.ndarray, window_name: str) -> bool:
    _ = timestamp_ns
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):
        return False
    return True


def main() -> int:
    args = parse_args()
    start = time.time()
    count = 0

    try:
        with RawZstdFrameReader(args.input, skip_corrupted=not args.strict) as reader:
            for timestamp_ns, frame in reader:
                should_continue = process_frame(timestamp_ns, frame, args.window_name)
                count += 1
                if not should_continue:
                    break
                if args.max_frames > 0 and count >= args.max_frames:
                    break
    finally:
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    print(f"[RFZST] Read {count} frames from {args.input} in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
