import argparse
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import openvino as ov
from imutils.video import VideoStream, FPS

# Norfair
from norfair_rs import Detection, Tracker
from norfair_rs import create_normalized_mean_euclidean_distance
from norfair_rs import OptimizedKalmanFilterFactory

# -----------------------------
# Benchmark
# -----------------------------
class Bench:
    KEYS = ("read", "resize", "prep", "infer", "post", "trk", "cnt", "draw", "total")

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
                f"infer={avg['infer']:.2f} post={avg['post']:.2f} trk={avg['trk']:.2f} "
                f"cnt={avg['cnt']:.2f} draw={avg['draw']:.2f} total={avg['total']:.2f}")

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
    bench: bool
    bench_warmup: int
    bench_every: int
    nms_topk: int

    # Norfair params
    distance_threshold: float
    hit_counter_max: int
    initialization_delay: int


def parse_args() -> AppArgs:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to YOLO OpenVINO .xml file")
    ap.add_argument("-i", "--input", type=str, help="path to video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.45)
    ap.add_argument("-s", "--skip-frames", type=int, default=5)
    ap.add_argument("-d", "--debug", type=int, default=0)

    ap.add_argument("--bench", action="store_true", help="enable benchmark timings")
    ap.add_argument("--bench-warmup", type=int, default=50, help="warmup frames before printing benchmark")
    ap.add_argument("--bench-every", type=int, default=200, help="print benchmark every N frames after warmup")
    ap.add_argument("--nms-topk", type=int, default=200, help="limit candidates before NMS (0=off)")

    # Norfair tuning
    ap.add_argument("--dist-th", type=float, default=0.40, help="Norfair distance_threshold (0..1)")
    ap.add_argument("--hit-max", type=int, default=0, help="Norfair hit_counter_max (0=auto from skip_frames)")
    ap.add_argument("--init-delay", type=int, default=1, help="Norfair initialization_delay")

    a = ap.parse_args()
    return AppArgs(
        model=a.model,
        input=a.input,
        confidence=a.confidence,
        skip_frames=a.skip_frames,
        debug=a.debug,
        bench=a.bench,
        bench_warmup=a.bench_warmup,
        bench_every=a.bench_every,
        nms_topk=a.nms_topk,
        distance_threshold=a.dist_th,
        hit_counter_max=a.hit_max,
        initialization_delay=a.init_delay,
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
        try:
            self.vs.stop()
        except Exception:
            pass


# -----------------------------
# Resize helper
# -----------------------------
class FrameResizer:
    def __init__(self, target_w: int = 256):
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
# Detector: OpenVINO (sync) + postprocess
# -----------------------------
class OpenVINODetector:
    def __init__(self, model_xml_path: str, device: str = "CPU"):
        self.core = ov.Core()
        model = self.core.read_model(model=model_xml_path)
        self.compiled = self.core.compile_model(model, device)
        self.output = self.compiled.output(0)

        input_shape = list(self.compiled.input(0).shape)  # [1, 3, H, W]
        self.input_h = int(input_shape[2])
        self.input_w = int(input_shape[3])

        self.inp = np.empty((1, 3, self.input_h, self.input_w), dtype=np.float32)

    def preprocess(self, frame: np.ndarray) -> float:
        t0 = time.perf_counter()

        img = cv2.resize(
            frame,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_LINEAR
        )

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        self.inp[0] = img

        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

    def infer(self) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        results = self.compiled([self.inp])[self.output]
        t1 = time.perf_counter()
        return results, (t1 - t0) * 1000.0

    def postprocess(
        self,
        results: np.ndarray,
        W: int,
        H: int,
        conf_th: float,
        nms_topk: int,
        nms_iou: float = 0.45,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], float]:
        t0 = time.perf_counter()

        outputs = np.squeeze(results)  # ожидаем (300, 6)

        if outputs.ndim != 2 or outputs.shape[1] != 6:
            raise ValueError(f"Unexpected output shape: {results.shape}")

        sx = W / float(self.input_w)
        sy = H / float(self.input_h)

        rects: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []

        # Формат: [x1, y1, x2, y2, score, class_id]
        conf = outputs[:, 4]
        mask = conf > conf_th

        if np.any(mask):
            cand = outputs[mask]

            if nms_topk and cand.shape[0] > nms_topk:
                cand_conf = cand[:, 4]
                idx = np.argpartition(cand_conf, -nms_topk)[-nms_topk:]
                cand = cand[idx]

            x1 = (cand[:, 0] * sx).astype(np.int32)
            y1 = (cand[:, 1] * sy).astype(np.int32)
            x2 = (cand[:, 2] * sx).astype(np.int32)
            y2 = (cand[:, 3] * sy).astype(np.int32)
            confs = cand[:, 4].astype(float)

            for i in range(len(cand)):
                xx1 = max(0, min(W - 1, int(x1[i])))
                yy1 = max(0, min(H - 1, int(y1[i])))
                xx2 = max(0, min(W - 1, int(x2[i])))
                yy2 = max(0, min(H - 1, int(y2[i])))

                if xx2 > xx1 and yy2 > yy1:
                    rects.append((xx1, yy1, xx2, yy2))
                    scores.append(float(confs[i]))

        t1 = time.perf_counter()
        return rects, scores, (t1 - t0) * 1000.0
# -----------------------------
# Counting logic (uses Norfair IDs)
# -----------------------------
@dataclass
class TrackableObject:
    objectID: int
    centroid: Tuple[int, int]
    counted: bool = False

    def __post_init__(self):
        self.centroids: List[Tuple[int, int]] = [self.centroid]


class PeopleCounter:
    def __init__(self):
        self.trackableObjects: Dict[int, TrackableObject] = {}
        self.totalDown = 0
        self.totalUp = 0

    def update_from_ids(
        self,
        id_centroids: List[Tuple[int, Tuple[int, int]]],
        H: int,
        is_det: bool,                 # <-- добавили
        debug: int = 0,
        frame: Optional[np.ndarray] = None,
    ) -> float:
        t0 = time.perf_counter()
        line_y = H // 2

        for objectID, centroid in id_centroids:
            to = self.trackableObjects.get(objectID)

            if to is None:
                to = TrackableObject(objectID, centroid)
                self.trackableObjects[objectID] = to
            else:
                # ВАЖНО: обновляем историю только на детекционных кадрах
                if is_det:
                    to.centroids.append(centroid)

                    if len(to.centroids) >= 2:
                        prev_y = to.centroids[-2][1]
                        curr_y = to.centroids[-1][1]

                        if prev_y < line_y <= curr_y:
                            self.totalDown += 1
                        elif prev_y > line_y >= curr_y:
                            self.totalUp += 1
            if debug and frame is not None:
                cv2.circle(frame, (centroid[0], centroid[1]), 3, (255, 255, 255), -1)

        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

def rects_to_norfair_detections(
    rects: List[Tuple[int, int, int, int]],
    scores: Optional[List[float]] = None,
) -> List[Detection]:
    dets: List[Detection] = []
    if scores is None:
        scores = [1.0] * len(rects)

    for (x1, y1, x2, y2), s in zip(rects, scores):
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        dets.append(
            Detection(
                points=np.array([[cx, cy]], dtype=np.float32),
                scores=np.array([float(s)], dtype=np.float32),
            )
        )
    return dets


def tracked_to_id_centroids(tracked_objects) -> List[Tuple[int, Tuple[int, int]]]:
    out: List[Tuple[int, Tuple[int, int]]] = []
    for t in tracked_objects:
        cx, cy = t.estimate[0]  # shape (1,2)
        out.append((int(t.id), (int(cx), int(cy))))
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    config = load_config()

    print("[INFO] Загрузка модели OpenVINO...")
    detector = OpenVINODetector(args.model, device="CPU")

    # Source
    if args.input:
        source: VideoSource = FileVideoSource(args.input)
    else:
        source = StreamVideoSource(config["url"])

    resizer = FrameResizer(target_w=256)
    counter = PeopleCounter()
    fps = FPS().start()

    bench = Bench()
    warmup = args.bench_warmup
    every = args.bench_every

    totalFrames = 0
    W = H = None

    norfair_tracker: Optional[Tracker] = None
    frames_since_det = 0
    try:
        while True:
            # time.sleep(0.1)
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

            # init Norfair once
            if norfair_tracker is None:
                distance_fn = create_normalized_mean_euclidean_distance(height=H, width=W)
                hit_max = args.hit_counter_max if args.hit_counter_max > 0 else max(120, 4 * args.skip_frames)
                norfair_tracker = Tracker(
                    distance_function=distance_fn,
                    distance_threshold=float(args.distance_threshold),
                    hit_counter_max=int(hit_max),
                    initialization_delay=int(args.initialization_delay),
                    filter_factory=OptimizedKalmanFilterFactory(),
                    past_detections_length=3
                )

            is_det = (totalFrames % args.skip_frames == 0)

            prep_ms = infer_ms = post_ms = trk_ms = 0.0
            rects: List[Tuple[int, int, int, int]] = []
            scores: List[float] = []

            # ---- DETECT ----
            if is_det:
                prep_ms = detector.preprocess(frame)
                results, infer_ms = detector.infer()
                rects, scores, post_ms = detector.postprocess(results, W, H, args.confidence, args.nms_topk)
                detections = rects_to_norfair_detections(rects, scores)
                t0 = time.perf_counter()
                tracked_objects = norfair_tracker.update(detections=detections, period=args.skip_frames)
                frames_since_det = 0
                t1 = time.perf_counter()
                # print("DET frame", totalFrames, "period", max(1, frames_since_det), "DETS", len(rects), "TRK", len(tracked_objects))
            else:
                detections = []
                t0 = time.perf_counter()
                tracked_objects = norfair_tracker.update()
                frames_since_det += 1
                t1 = time.perf_counter()

            # ---- TRACK ----
           
            trk_ms = (t1 - t0) * 1000.0

            id_centroids = tracked_to_id_centroids(tracked_objects)

            # ---- COUNT ----
            cnt_ms = counter.update_from_ids(id_centroids, is_det=is_det, H=H, debug=args.debug, frame=frame)

            # ---- DRAW ----
            draw_ms = 0.0
            if args.debug:
                t0 = time.perf_counter()

                # stats
                cv2.putText(frame, f"In: {counter.totalDown} Out: {counter.totalUp}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"DETS: {len(rects) if is_det else '-'}  TRK: {len(id_centroids)}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # raw det boxes (only on det frame)
                # if is_det:
                #     for (x1, y1, x2, y2) in rects:
                #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # track boxes (every frame)
                for tid, (cx, cy) in id_centroids:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"{tid}", (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # count line
                cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

                cv2.imshow("OpenVINO + Norfair (bbox-2points)", frame)
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
                    read=read_ms,
                    resize=resize_ms,
                    prep=prep_ms,
                    infer=infer_ms,
                    post=post_ms,
                    trk=trk_ms,
                    cnt=cnt_ms,
                    draw=draw_ms,
                    total=total_ms,
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

        if args.bench:
            print(bench.report_all(prefix="[BENCH FINAL ALL]"))
            print(bench.report_det(prefix="[BENCH FINAL DET]"))
            print(bench.report_trk(prefix="[BENCH FINAL TRK]"))


if __name__ == "__main__":
    main()
