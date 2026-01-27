import argparse
import json
import time
from dataclasses import dataclass
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import openvino as ov
from imutils.video import VideoStream, FPS
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment


# -----------------------------
# Benchmark
# -----------------------------
class Bench:
    KEYS = ("read", "resize", "prep", "infer", "post", "kcf", "ct", "draw", "total")

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_n = 0
        self.det_n = 0
        self.trk_n = 0
        self.all_sum = {k: 0.0 for k in self.KEYS}
        self.det_sum = {k: 0.0 for k in self.KEYS}
        self.trk_sum = {k: 0.0 for k in self.KEYS}

    def add(self, is_det: bool, **ms):
        self.all_n += 1
        for k in self.KEYS:
            self.all_sum[k] += ms.get(k, 0.0)

        if is_det:
            self.det_n += 1
            for k in self.KEYS:
                self.det_sum[k] += ms.get(k, 0.0)
        else:
            self.trk_n += 1
            for k in self.KEYS:
                self.trk_sum[k] += ms.get(k, 0.0)

    @staticmethod
    def _fmt(avg):
        return (f"read={avg['read']:.2f} resize={avg['resize']:.2f} prep={avg['prep']:.2f} "
                f"infer={avg['infer']:.2f} post={avg['post']:.2f} kcf={avg['kcf']:.2f} "
                f"ct={avg['ct']:.2f} draw={avg['draw']:.2f} total={avg['total']:.2f}")

    def report_all(self, prefix="[BENCH ALL]"):
        if self.all_n == 0:
            return f"{prefix} no samples"
        avg = {k: self.all_sum[k] / self.all_n for k in self.KEYS}
        return f"{prefix} n={self.all_n} | " + self._fmt(avg)

    def report_det(self, prefix="[BENCH DET]"):
        if self.det_n == 0:
            return f"{prefix} no samples"
        avg = {k: self.det_sum[k] / self.det_n for k in self.KEYS}
        return f"{prefix} n={self.det_n} | " + self._fmt(avg)

    def report_trk(self, prefix="[BENCH TRK]"):
        if self.trk_n == 0:
            return f"{prefix} no samples"
        avg = {k: self.trk_sum[k] / self.trk_n for k in self.KEYS}
        return f"{prefix} n={self.trk_n} | " + self._fmt(avg)



# -----------------------------
# App config
# -----------------------------
@dataclass
class AppArgs:
    model: str
    input: Optional[str]
    confidence: float
    skip_frames: int
    debug: int
    kcf_step: int
    bench: bool
    bench_warmup: int
    bench_every: int
    nms_topk: int


def parse_args() -> AppArgs:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to YOLO12 OpenVINO .xml file")
    ap.add_argument("-i", "--input", type=str, help="path to video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.3)
    ap.add_argument("-s", "--skip-frames", type=int, default=20)
    ap.add_argument("-d", "--debug", type=int, default=0)
    ap.add_argument("--kcf-step", type=int, default=1, help="update KCF every N frames (1=each frame)")

    ap.add_argument("--bench", action="store_true", help="enable benchmark timings")
    ap.add_argument("--bench-warmup", type=int, default=50, help="warmup frames before printing benchmark")
    ap.add_argument("--bench-every", type=int, default=200, help="print benchmark every N frames after warmup")
    ap.add_argument("--nms-topk", type=int, default=200, help="limit candidates before NMS (0=off)")

    a = ap.parse_args()
    return AppArgs(
        model=a.model,
        input=a.input,
        confidence=a.confidence,
        skip_frames=a.skip_frames,
        debug=a.debug,
        kcf_step=a.kcf_step,
        bench=a.bench,
        bench_warmup=a.bench_warmup,
        bench_every=a.bench_every,
        nms_topk=a.nms_topk,
    )


def load_config(path: str = "utils/config.json") -> Dict:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Video source
# -----------------------------
class VideoSource:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class FileVideoSource(VideoSource):
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {path}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        self.cap.release()


class StreamVideoSource(VideoSource):
    def __init__(self, url: str):
        self.vs = VideoStream(url).start()
        time.sleep(2.0)

    def read(self) -> Optional[np.ndarray]:
        return self.vs.read()

    def release(self) -> None:
        # VideoStream не всегда имеет release(), но stop() есть
        try:
            self.vs.stop()
        except Exception:
            pass


# -----------------------------
# Resize helper (precompute new_size once)
# -----------------------------
class FrameResizer:
    def __init__(self, target_w: int = 320):
        self.target_w = target_w
        self.new_size: Optional[Tuple[int, int]] = None  # (w, h)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.new_size is None:
            h0, w0 = frame.shape[:2]
            scale = self.target_w / float(w0)
            new_h = int(h0 * scale)
            self.new_size = (self.target_w, new_h)
        return cv2.resize(frame, self.new_size, interpolation=cv2.INTER_LINEAR)


# -----------------------------
# Detector: OpenVINO (sync) + vectorized postprocess
# -----------------------------
class OpenVINODetector:
    def __init__(self, model_xml_path: str, device: str = "CPU"):
        self.core = ov.Core()
        model = self.core.read_model(model=model_xml_path)
        self.compiled = self.core.compile_model(model, device)
        self.output = self.compiled.output(0)

        self.inp = np.empty((1, 3, 320, 320), dtype=np.float32)

    def preprocess(self, frame: np.ndarray) -> float:
        t0 = time.perf_counter()
        img = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        self.inp[0] = img.astype(np.float32) * (1.0 / 255.0)
        # self.inp = cv2.dnn.blobFromImage(
        #     frame,
        #     scalefactor=1/255.0,
        #     size=(320, 320),
        #     swapRB=False,
        #     crop=False
        # )  # float32 NCHW, shape (1,3,320,320)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

    def infer(self) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        results = self.compiled([self.inp])[self.output]
        t1 = time.perf_counter()
        return results, (t1 - t0) * 1000.0

    def postprocess(self, results: np.ndarray, W: int, H: int,
                    conf_th: float, nms_topk: int, nms_iou: float = 0.45) -> Tuple[List[Tuple[int, int, int, int]], float]:
        t0 = time.perf_counter()

        outputs = np.squeeze(results).T  # (2100, 84)
        sx = (W / 320.0)
        sy = (H / 320.0)

        conf = outputs[:, 4]
        mask = conf > conf_th
        rects: List[Tuple[int, int, int, int]] = []

        if np.any(mask):
            cand = outputs[mask]
            cand_conf = conf[mask]

            if nms_topk and cand.shape[0] > nms_topk:
                idx = np.argpartition(cand_conf, -nms_topk)[-nms_topk:]
                cand = cand[idx]
                cand_conf = cand_conf[idx]

            xc = cand[:, 0]
            yc = cand[:, 1]
            w = cand[:, 2]
            h = cand[:, 3]

            x1 = ((xc - w / 2.0) * sx).astype(np.int32)
            y1 = ((yc - h / 2.0) * sy).astype(np.int32)
            ww = (w * sx).astype(np.int32)
            hh = (h * sy).astype(np.int32)

            boxes = np.stack([x1, y1, ww, hh], axis=1).tolist()
            confs = cand_conf.astype(float).tolist()

            indices = cv2.dnn.NMSBoxes(boxes, confs, conf_th, nms_iou)

            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, bw, bh = boxes[i]
                    rects.append((x, y, x + bw, y + bh))

        t1 = time.perf_counter()
        return rects, (t1 - t0) * 1000.0


# -----------------------------
# KCF tracker manager
# -----------------------------
def create_kcf_tracker():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    return cv2.legacy.TrackerKCF_create()


class KCFTrackerManager:
    def __init__(self):
        self.trackers: List = []
        self.last_rects: List[Tuple[int, int, int, int]] = []

    def reset_from_detections(self, frame: np.ndarray, rects: List[Tuple[int, int, int, int]]) -> None:
        self.trackers = []
        for (x1, y1, x2, y2) in rects:
            w = int(x2 - x1)
            h = int(y2 - y1)
            tracker = create_kcf_tracker()
            tracker.init(frame, (int(x1), int(y1), w, h))
            self.trackers.append(tracker)
        self.last_rects = rects

    def update(self, frame: np.ndarray, do_update: bool) -> Tuple[List[Tuple[int, int, int, int]], float]:
        if not do_update:
            return self.last_rects, 0.0

        t0 = time.perf_counter()
        rects: List[Tuple[int, int, int, int]] = []
        for tracker in self.trackers:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                rects.append((x, y, x + w, y + h))
        t1 = time.perf_counter()

        self.last_rects = rects
        return rects, (t1 - t0) * 1000.0


# -----------------------------
# Counting logic wrapper
# -----------------------------
class PeopleCounter:
    def __init__(self, maxDisappeared=90, maxDistance=100):
        self.ct = CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=maxDistance)
        self.trackableObjects: Dict[int, TrackableObject] = {}
        self.totalDown = 0
        self.totalUp = 0

    def update(self, rects: List[Tuple[int, int, int, int]], H: int, W: int,
               debug: int = 0, frame: Optional[np.ndarray] = None) -> Tuple[float, Dict[int, Tuple[int, int]]]:
        t0 = time.perf_counter()
        objects = self.ct.update(rects)

        line_y = H // 2

        for (objectID, centroid) in objects.items():
            to = self.trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                to.centroids.append(centroid)

                # пересечение линии по факту (между prev и curr)
                if not to.counted and len(to.centroids) >= 2:
                    prev_y = to.centroids[-2][1]
                    curr_y = to.centroids[-1][1]

                    if prev_y < line_y <= curr_y:
                        self.totalDown += 1
                        to.counted = True
                    elif prev_y > line_y >= curr_y:
                        self.totalUp += 1
                        to.counted = True

            self.trackableObjects[objectID] = to

            if debug and frame is not None:
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
                cv2.putText(frame, f"{objectID}", (centroid[0], centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0, objects


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    config = load_config()

    print("[INFO] Загрузка модели OpenVINO...")
    detector = OpenVINODetector(args.model, device="CPU")

    # Source
    is_file = bool(args.input)
    if is_file:
        source: VideoSource = FileVideoSource(args.input)
    else:
        source = StreamVideoSource(config["url"])

    resizer = FrameResizer(target_w=320)
    tracker_mgr = KCFTrackerManager()
    counter = PeopleCounter(maxDisappeared=90, maxDistance=100)

    fps = FPS().start()

    bench = Bench()
    warmup = args.bench_warmup
    every = args.bench_every

    totalFrames = 0
    W = H = None

    try:
        while True:
            t_total0 = time.perf_counter()

            # ---- READ ----
            t0 = time.perf_counter()
            frame = source.read()
            t1 = time.perf_counter()
            if frame is None:
                break
            read_ms = (t1 - t0) * 1000.0

            # ---- RESIZE ----
            t0 = time.perf_counter()
            frame = resizer.apply(frame)
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            t1 = time.perf_counter()
            resize_ms = (t1 - t0) * 1000.0

            is_det = (totalFrames % args.skip_frames == 0)

            prep_ms = infer_ms = post_ms = kcf_ms = 0.0
            rects: List[Tuple[int, int, int, int]] = []

            if is_det:
                # DETECT
                prep_ms = detector.preprocess(frame)
                results, infer_ms = detector.infer()
                rects, post_ms = detector.postprocess(
                    results, W=W, H=H,
                    conf_th=args.confidence,
                    nms_topk=args.nms_topk
                )
                tracker_mgr.reset_from_detections(frame, rects)
            else:
                # TRACK
                do_update = (totalFrames % args.kcf_step == 0)
                rects, kcf_ms = tracker_mgr.update(frame, do_update=do_update)

            # ---- COUNT ----
            ct_ms, _objects = counter.update(rects, H=H, W=W, debug=args.debug, frame=frame)

            # ---- DRAW ----
            draw_ms = 0.0
            if args.debug:
                t0 = time.perf_counter()
                cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
                cv2.putText(frame, f"In: {counter.totalDown} Out: {counter.totalUp}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("OpenVINO INT8 + KCF (sync, refactored)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                t1 = time.perf_counter()
                draw_ms = (t1 - t0) * 1000.0

            totalFrames += 1
            fps.update()

            t_total1 = time.perf_counter()
            total_ms = (t_total1 - t_total0) * 1000.0

            if args.bench:
                bench.add(
                    is_det,
                    read=read_ms, resize=resize_ms, prep=prep_ms, infer=infer_ms, post=post_ms,
                    kcf=kcf_ms, ct=ct_ms, draw=draw_ms, total=total_ms
                )
                if totalFrames > warmup and (totalFrames % every == 0):
                    print(bench.report_all())
                    print(bench.report_det())
                    print(bench.report_trk())

    finally:
        fps.stop()
        source.release()
        if args.debug:
            cv2.destroyAllWindows()

        print(f"[INFO] Итого IN: {counter.totalDown}")
        print(f"[INFO] Итого OUT: {counter.totalUp}")
        print(f"[INFO] Средний FPS: {fps.fps():.2f}")
        return fps.fps()

        if args.bench:
            print(bench.report_all(prefix="[BENCH FINAL ALL]"))
            print(bench.report_det(prefix="[BENCH FINAL DET]"))
            print(bench.report_trk(prefix="[BENCH FINAL TRK]"))


if __name__ == "__main__":
    fpss = np.array([])
    for i in range(20):
        fpss = np.append(fpss, main())
    print(f"Mean fps: {fpss.mean():.2f}")
    
