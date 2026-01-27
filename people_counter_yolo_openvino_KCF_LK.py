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


def clamp_bbox_xywh(x, y, w, h, W, H):
    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))
    w = int(max(1, min(w, W - x)))
    h = int(max(1, min(h, H - y)))
    return x, y, w, h


def bbox_center_xywh(x, y, w, h):
    return (x + w * 0.5, y + h * 0.5)


def seed_points(gray, x, y, w, h, max_pts=8):
    """
    Пытаемся взять текстурные точки внутри bbox через goodFeaturesToTrack.
    Если не получилось — fallback на "сетку" из 4 точек.
    """
    x, y, w, h = int(x), int(y), int(w), int(h)
    H, W = gray.shape[:2]
    x, y, w, h = clamp_bbox_xywh(x, y, w, h, W, H)

    roi = gray[y:y + h, x:x + w]
    pts = None
    if roi.size > 0 and w >= 8 and h >= 8:
        # maxCorners = max_pts, качество/дистанция подобраны "в среднем"
        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=max_pts,
            qualityLevel=0.01,
            minDistance=max(2, min(w, h) // 6),
            blockSize=3,
            useHarrisDetector=False
        )
        if pts is not None:
            pts = pts.reshape(-1, 2)
            pts[:, 0] += x
            pts[:, 1] += y
            pts = pts.reshape(-1, 1, 2).astype(np.float32)

    if pts is None or len(pts) == 0:
        # fallback: 4 точки сеткой
        xs = [x + 0.3 * w, x + 0.7 * w, x + 0.3 * w, x + 0.7 * w]
        ys = [y + 0.3 * h, y + 0.3 * h, y + 0.7 * h, y + 0.7 * h]
        pts = np.array(list(zip(xs, ys)), dtype=np.float32).reshape(-1, 1, 2)

    return pts


def people_counter():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to YOLO12 OpenVINO .xml file")
    ap.add_argument("-i", "--input", type=str, help="path to video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.3)
    ap.add_argument("-s", "--skip-frames", type=int, default=20)
    ap.add_argument("-d", "--debug", type=int, default=0)
    ap.add_argument("--kcf-step", type=int, default=2, help="update KCF every N frames (1=each frame)")

    # LK options
    ap.add_argument("--lk", action="store_true", help="enable Lucas-Kanade refinement for bboxes/centroids")
    ap.add_argument("--lk-points", type=int, default=8, help="number of points per bbox for LK (4-12 typical)")
    ap.add_argument("--lk-win", type=int, default=21, help="LK window size")
    ap.add_argument("--lk-maxlevel", type=int, default=3, help="LK pyramid maxLevel")
    ap.add_argument("--lk-stuck-th", type=float, default=1.0,
                    help="if KCF moves less than this (px) but LK suggests movement -> use LK shift")

    # Benchmark options
    ap.add_argument("--bench", action="store_true", help="enable benchmark timings")
    ap.add_argument("--bench-warmup", type=int, default=50, help="warmup frames before printing benchmark")
    ap.add_argument("--bench-every", type=int, default=200, help="print benchmark every N frames after warmup")
    ap.add_argument("--nms-topk", type=int, default=200, help="limit candidates before NMS (0=off)")

    args = vars(ap.parse_args())

    print("[INFO] Загрузка модели OpenVINO...")
    core = ov.Core()
    model_ov = core.read_model(model=args["model"])
    compiled_model = core.compile_model(model_ov, "CPU")
    output_layer = compiled_model.output(0)

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

    # Tracker state
    W = H = None
    new_size = None
    ct = CentroidTracker(maxDisappeared=90, maxDistance=100)

    trackers = []            # список KCF объектов
    track_boxes = []         # список bbox (x,y,w,h) для каждого tracker
    lk_pts_list = []         # список точек (Nx1x2) для LK по каждому tracker

    trackableObjects = {}
    totalFrames = 0
    totalDown = totalUp = 0

    fps = FPS().start()
    inp = np.empty((1, 3, 320, 320), dtype=np.float32)

    last_rects = []
    prev_gray = None

    bench = Bench()
    warmup = args["bench_warmup"]
    every = args["bench_every"]

    lk_enabled = bool(args["lk"])

    # LK params
    lk_params = dict(
        winSize=(args["lk_win"], args["lk_win"]),
        maxLevel=args["lk_maxlevel"],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

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

        # ---- gray for LK ----
        gray = None
        if lk_enabled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = []
        is_det = (totalFrames % args["skip_frames"] == 0)

        prep_ms = infer_ms = post_ms = kcf_ms = 0.0

        if is_det:
            trackers = []
            track_boxes = []
            lk_pts_list = []

            # ---- PREP ----
            t0 = time.perf_counter()
            img = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
            img = img.transpose(2, 0, 1)
            inp[0] = img.astype(np.float32) * (1.0 / 255.0)
            t1 = time.perf_counter()
            prep_ms = (t1 - t0) * 1000.0

            # ---- INFER ----
            t0 = time.perf_counter()
            results = compiled_model([inp])[output_layer]
            t1 = time.perf_counter()
            infer_ms = (t1 - t0) * 1000.0

            # ---- POST (vectorized filter + top-k) ----
            t0 = time.perf_counter()
            outputs = np.squeeze(results).T  # (2100, 84)

            conf_th = args["confidence"]
            sx = (W / 320.0)
            sy = (H / 320.0)

            conf = outputs[:, 4]
            mask = conf > conf_th

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
                boxes, confs = [], []
                indices = []

            rects = []
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y, w, h) = boxes[i]
                    x, y, w, h = clamp_bbox_xywh(x, y, w, h, W, H)

                    rects.append((x, y, x + w, y + h))

                    # create KCF tracker
                    try:
                        tracker = cv2.TrackerKCF_create()
                    except AttributeError:
                        tracker = cv2.legacy.TrackerKCF_create()
                    tracker.init(frame, (x, y, w, h))
                    trackers.append(tracker)

                    # store bbox + seed LK points
                    track_boxes.append((x, y, w, h))
                    if lk_enabled:
                        lk_pts_list.append(seed_points(gray, x, y, w, h, max_pts=args["lk_points"]))

            last_rects = rects

            # reset LK previous frame on detection (чтобы не было "скачка")
            if lk_enabled:
                prev_gray = gray

            t1 = time.perf_counter()
            post_ms = (t1 - t0) * 1000.0

        else:
            # ---- TRACKING (KCF + optional LK) ----
            t0 = time.perf_counter()

            # 1) LK предлагаемые сдвиги (на каждом кадре, дёшево)
            lk_suggested = [None] * len(track_boxes)  # (dx,dy) per tracker
            if lk_enabled and prev_gray is not None and gray is not None and len(track_boxes) > 0:
                for i, (bbox, p0) in enumerate(zip(track_boxes, lk_pts_list)):
                    if p0 is None or len(p0) == 0:
                        lk_suggested[i] = None
                        continue

                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                    if p1 is None or st is None:
                        lk_suggested[i] = None
                        continue

                    st = st.reshape(-1)
                    good0 = p0[st == 1].reshape(-1, 2)
                    good1 = p1[st == 1].reshape(-1, 2)

                    if good1.shape[0] < 1:
                        lk_suggested[i] = None
                        continue

                    d = good1 - good0
                    dx, dy = np.median(d, axis=0)
                    lk_suggested[i] = (float(dx), float(dy))

                    # обновим точки на "новые" хорошие
                    lk_pts_list[i] = good1.reshape(-1, 1, 2).astype(np.float32)

            # 2) KCF update по расписанию
            do_kcf = ((totalFrames % args["kcf_step"]) == 0)

            new_boxes = []
            for i, tracker in enumerate(trackers):
                x_prev, y_prev, w_prev, h_prev = track_boxes[i]
                cx_prev, cy_prev = bbox_center_xywh(x_prev, y_prev, w_prev, h_prev)

                # базово: если KCF не делаем - применяем LK (если есть), иначе оставляем как было
                x_use, y_use, w_use, h_use = x_prev, y_prev, w_prev, h_prev

                # LK candidate
                if lk_suggested[i] is not None:
                    dx, dy = lk_suggested[i]
                    x_lk = x_prev + dx
                    y_lk = y_prev + dy
                    x_lk, y_lk, w_lk, h_lk = clamp_bbox_xywh(x_lk, y_lk, w_prev, h_prev, W, H)
                else:
                    x_lk, y_lk, w_lk, h_lk = x_prev, y_prev, w_prev, h_prev

                if do_kcf:
                    success, box = tracker.update(frame)
                    if success:
                        xk, yk, wk, hk = [int(v) for v in box]
                        xk, yk, wk, hk = clamp_bbox_xywh(xk, yk, wk, hk, W, H)
                        cx_k, cy_k = bbox_center_xywh(xk, yk, wk, hk)
                        move_kcf = ((cx_k - cx_prev) ** 2 + (cy_k - cy_prev) ** 2) ** 0.5

                        # если KCF почти не двинулся, но LK даёт сдвиг — берем LK (помогает при "залипании")
                        if lk_enabled and lk_suggested[i] is not None:
                            dx, dy = lk_suggested[i]
                            move_lk = (dx * dx + dy * dy) ** 0.5
                            if move_kcf < args["lk_stuck_th"] and move_lk > args["lk_stuck_th"]:
                                x_use, y_use, w_use, h_use = x_lk, y_lk, w_lk, h_lk
                            else:
                                x_use, y_use, w_use, h_use = xk, yk, wk, hk
                        else:
                            x_use, y_use, w_use, h_use = xk, yk, wk, hk
                    else:
                        # KCF не смог — fallback на LK/prev
                        x_use, y_use, w_use, h_use = x_lk, y_lk, w_lk, h_lk
                else:
                    # между KCF-апдейтами используем LK (если есть)
                    x_use, y_use, w_use, h_use = x_lk, y_lk, w_lk, h_lk

                new_boxes.append((int(x_use), int(y_use), int(w_use), int(h_use)))

            track_boxes = new_boxes

            # 3) reseed LK points внутри текущих bbox (чтобы точки не уехали за границы)
            if lk_enabled and gray is not None:
                lk_pts_list = [
                    seed_points(gray, x, y, w, h, max_pts=args["lk_points"])
                    for (x, y, w, h) in track_boxes
                ]
                prev_gray = gray

            # 4) rects для CentroidTracker
            rects = [(x, y, x + w, y + h) for (x, y, w, h) in track_boxes]
            last_rects = rects

            t1 = time.perf_counter()
            # складываем всё трекинговое время сюда (KCF+LK)
            kcf_ms = (t1 - t0) * 1000.0

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
            cv2.imshow("OpenVINO INT8 + KCF + LK", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            t1 = time.perf_counter()
            draw_ms = (t1 - t0) * 1000.0

        totalFrames += 1
        fps.update()

        t_total1 = time.perf_counter()
        total_ms = (t_total1 - t_total0) * 1000.0

        if args["bench"]:
            bench.add(
                is_det,
                read=read_ms,
                resize=resize_ms,
                prep=prep_ms,
                infer=infer_ms,
                post=post_ms,
                kcf=kcf_ms,   # тут KCF+LK вместе
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