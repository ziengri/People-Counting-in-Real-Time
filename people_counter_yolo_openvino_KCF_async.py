from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import openvino as ov
import numpy as np
import argparse
import time
import json
import cv2

# Инициализация конфигурации
with open("utils/config.json", "r") as file:
    config = json.load(file)


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


def people_counter():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to YOLO12 OpenVINO .xml file")
    ap.add_argument("-i", "--input", type=str, help="path to video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.3)
    ap.add_argument("-s", "--skip-frames", type=int, default=20)
    ap.add_argument("-d", "--debug", type=int, default=0)

    ap.add_argument("--kcf-step", type=int, default=2, help="update KCF every N frames (1=each frame)")
    ap.add_argument("--nms-topk", type=int, default=200, help="limit candidates before NMS (0=off)")

    ap.add_argument("--realtime", action="store_true", help="throttle file input to real-time fps (needed for async)")
    ap.add_argument("--realtime-fps", type=float, default=0.0, help="override source fps (0=use CAP_PROP_FPS)")


    # Benchmark options
    ap.add_argument("--bench", action="store_true", help="enable benchmark timings")
    ap.add_argument("--bench-warmup", type=int, default=50, help="warmup frames before printing benchmark")
    ap.add_argument("--bench-every", type=int, default=200, help="print benchmark every N frames after warmup")

    args = vars(ap.parse_args())

    print("[INFO] Загрузка модели OpenVINO...")
    core = ov.Core()
    model_ov = core.read_model(model=args["model"])
    compiled_model = core.compile_model(model_ov, "CPU")

    input_port = compiled_model.input(0)
    output_port = compiled_model.output(0)

    infer_request = compiled_model.create_infer_request()

    # Video input
    is_file = bool(args.get("input", False))
    if not is_file:
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
    else:
        vs = cv2.VideoCapture(args["input"])
        if not vs.isOpened():
            print(f"[ERROR] Не удалось открыть видео: {args['input']}")
            return
    src_fps = 0.0
    if is_file:
        src_fps = float(vs.get(cv2.CAP_PROP_FPS) or 0.0)
        if args["realtime_fps"] > 0:
            src_fps = float(args["realtime_fps"])
        if src_fps <= 1.0:
            src_fps = 25.0  # fallback
        frame_period = 1.0 / src_fps
        t_stream0 = time.perf_counter()

    # Tracker state
    W = H = None
    new_size = None
    ct = CentroidTracker(maxDisappeared=90, maxDistance=100)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalDown = totalUp = 0

    fps = FPS().start()
    inp = np.empty((1, 3, 320, 320), dtype=np.float32)

    # чтобы при kcf-step>1 не передавать пустые rects
    last_rects = []

    # async state
    inflight = False
    inflight_t0 = 0.0
    inflight_W = inflight_H = None
    inflight_frame_for_init = None  # кадр, на котором запускаем детект (для init трекеров)

    # bench
    bench = Bench()
    warmup = args["bench_warmup"]
    every = args["bench_every"]

    while True:
        t_total0 = time.perf_counter()

        # ---- READ ----
        t0 = time.perf_counter()
        frame = vs.read()
        frame = frame[1] if is_file else frame
        t1 = time.perf_counter()
        if frame is None:
            break
        read_ms = (t1 - t0) * 1000.0

        # ---- RESIZE (cv2, one-time size calc) ----
        t0 = time.perf_counter()
        if new_size is None:
            h0, w0 = frame.shape[:2]
            target_w = 320
            scale = target_w / float(w0)
            new_h = int(h0 * scale)
            new_size = (target_w, new_h)  # (width, height)
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        t1 = time.perf_counter()
        resize_ms = (t1 - t0) * 1000.0

        # per-frame stage times
        prep_ms = infer_ms = post_ms = kcf_ms = 0.0
        rects = []
        det_completed_this_frame = False

        # ---- START ASYNC DETECTION on schedule (if not inflight) ----
        if (totalFrames % args["skip_frames"] == 0) and (not inflight):
            t0 = time.perf_counter()

            img = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
            img = img.transpose(2, 0, 1)
            inp[0] = img.astype(np.float32) * (1.0 / 255.0)

            # set tensor and start async
            infer_request.set_tensor(input_port, ov.Tensor(inp))
            infer_request.start_async()

            inflight = True
            inflight_t0 = time.perf_counter()
            inflight_W, inflight_H = W, H

            # сохраняем кадр для init трекеров (он маленький, копия дешёвая)
            inflight_frame_for_init = frame.copy()

            t1 = time.perf_counter()
            prep_ms = (t1 - t0) * 1000.0

        # ---- CHECK ASYNC COMPLETION ----
        if inflight and infer_request.wait_for(0):
            # infer time = from start_async moment to completion
            t_done = time.perf_counter()
            infer_ms = (t_done - inflight_t0) * 1000.0

            # get output tensor
            results = infer_request.get_tensor(output_port).data  # numpy view/copy depending on runtime

            # ---- POST (vectorized) ----
            t0 = time.perf_counter()

            outputs = np.squeeze(results).T  # (2100, 84) expected

            conf_th = args["confidence"]
            sx = (inflight_W / 320.0)
            sy = (inflight_H / 320.0)

            conf = outputs[:, 4]
            mask = conf > conf_th

            boxes = []
            confs = []

            if np.any(mask):
                cand = outputs[mask]
                cand_conf = conf[mask]

                k = args["nms_topk"]
                if k and cand.shape[0] > k:
                    idx = np.argpartition(cand_conf, -k)[-k:]
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

                indices = cv2.dnn.NMSBoxes(boxes, confs, conf_th, 0.45)
            else:
                indices = []

            # rebuild trackers from detection result
            trackers = []
            rects = []

            init_frame = inflight_frame_for_init if inflight_frame_for_init is not None else frame

            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y, w, h) = boxes[i]
                    rects.append((x, y, x + w, y + h))

                    try:
                        tracker = cv2.TrackerKCF_create()
                    except AttributeError:
                        tracker = cv2.legacy.TrackerKCF_create()
                    tracker.init(init_frame, (x, y, w, h))
                    trackers.append(tracker)

            last_rects = rects

            t1 = time.perf_counter()
            post_ms = (t1 - t0) * 1000.0

            # clear inflight
            inflight = False
            inflight_frame_for_init = None
            det_completed_this_frame = True

        # ---- TRACKING (only if we didn't just apply detection this frame) ----
        if not det_completed_this_frame:
            if (totalFrames % args["kcf_step"]) == 0:
                t0 = time.perf_counter()
                rects = []
                for tracker in trackers:
                    success, box = tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in box]
                        rects.append((x, y, x + w, y + h))
                t1 = time.perf_counter()
                kcf_ms = (t1 - t0) * 1000.0
                last_rects = rects
            else:
                rects = last_rects

        # ---- CentroidTracker + counting ----
        t0 = time.perf_counter()
        objects = ct.update(rects)
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y_coords = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y_coords)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2 and len(to.centroids) > 10:
                        totalUp += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > H // 2 and len(to.centroids) > 10:
                        totalDown += 1
                        to.counted = True

            trackableObjects[objectID] = to
            if args["debug"]:
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        t1 = time.perf_counter()
        ct_ms = (t1 - t0) * 1000.0

        # ---- DRAW ----
        draw_ms = 0.0
        if args["debug"]:
            t0 = time.perf_counter()
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            cv2.putText(frame, f"In: {totalDown} Out: {totalUp}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("OpenVINO INT8 + KCF (async)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            t1 = time.perf_counter()
            draw_ms = (t1 - t0) * 1000.0

        if is_file and args["realtime"]:
            target = t_stream0 + (totalFrames + 1) * frame_period
            now = time.perf_counter()
            dt = target - now
            if dt > 0:
                time.sleep(dt)
        totalFrames += 1
        fps.update()

        t_total1 = time.perf_counter()
        total_ms = (t_total1 - t_total0) * 1000.0

        # For benchmark: treat "det" as frames where detection finished (not where it was scheduled)
        if args["bench"]:
            bench.add(
                det_completed_this_frame,
                read=read_ms,
                resize=resize_ms,
                prep=prep_ms,
                infer=infer_ms,
                post=post_ms,
                kcf=kcf_ms,
                ct=ct_ms,
                draw=draw_ms,
                total=total_ms,
            )

            if totalFrames > warmup and (totalFrames % every == 0):
                print(bench.report_all())
                print(bench.report_det())
                print(bench.report_trk())

    fps.stop()
    print(f"[INFO] Итого IN: {totalDown}")
    print(f"[INFO] Итого OUT: {totalUp}")
    print(f"[INFO] Средний FPS: {fps.fps():.2f}")

    if args["bench"]:
        print(bench.report_all(prefix="[BENCH FINAL ALL]"))
        print(bench.report_det(prefix="[BENCH FINAL DET]"))
        print(bench.report_trk(prefix="[BENCH FINAL TRK]"))

    if args["debug"]:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    people_counter()
