from __future__ import annotations

import argparse
import time

import cv2

from rfzl4 import RawLZ4FrameWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MP4 stream into RFZL4 container")
    parser.add_argument("--input", required=True, help="Path to input MP4/video file")
    parser.add_argument("--output", required=True, help="Path to output .rfzl4 file")
    parser.add_argument("--chunk-frames", type=int, default=64, help="Frames per chunk")
    parser.add_argument("--fps", type=int, default=25, help="Nominal FPS in file header")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.input}")

    frame_count = 0
    start = time.time()
    writer = RawLZ4FrameWriter(
        args.output,
        width=256,
        height=256,
        channels=3,
        fps=args.fps,
        chunk_frames=args.chunk_frames,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            writer.write_frame(frame, time.time_ns())
            frame_count += 1
    finally:
        cap.release()
        final_path = writer.close()

    elapsed = time.time() - start
    print(f"[RFZL4] Wrote {frame_count} frames to {final_path} in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
