from __future__ import annotations

import argparse

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write raw resized frames to a .bin file")
    parser.add_argument("--input", required=True, help="Path to input MP4/video file")
    parser.add_argument("--output", default="test.bin", help="Output raw binary file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.input}")

    frames = 0
    bytes_written = 0

    with open(args.output, "wb") as out:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            blob = frame.tobytes(order="C")
            out.write(blob)

            frames += 1
            bytes_written += len(blob)

    cap.release()

    print(f"[RAWBIN] Wrote {frames} frames to {args.output}")
    print(f"[RAWBIN] Total bytes: {bytes_written}")
    print(f"[RAWBIN] Total MiB: {bytes_written / (1024 * 1024):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
