from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import onnxruntime as ort
import numpy as np
import argparse
import datetime
import imutils
import time
import json
import cv2

# Инициализация конфигурации
with open("utils/config.json", "r") as file:
    config = json.load(file)

def people_counter():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to YOLOv8 ONNX model")
    ap.add_argument("-i", "--input", type=str, help="path to video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.3)
    ap.add_argument("-s", "--skip-frames", type=int, default=10)
    args = vars(ap.parse_args())

    # 1. Загрузка YOLOv8 через ONNX Runtime
    session = ort.InferenceSession(args["model"], providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 2. Запуск видеопотока с проверкой
    if not args.get("input", False):
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
    else:
        vs = cv2.VideoCapture(args["input"])
        if not vs.isOpened():
            print(f"[ERROR] Не удалось открыть видео: {args['input']}")
            return

    W, H = None, None
    ct = CentroidTracker(maxDisappeared=60, maxDistance=70)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalDown, totalUp = 0, 0
    move_in, move_out = [], []

    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        if frame is None:
            break

        # Оптимизация размера для J1900
        frame = imutils.resize(frame, width=320)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        rects = []

        # ДЕТЕКЦИЯ (каждые N кадров)
        if totalFrames % args["skip_frames"] == 0:
            trackers = []
            
            # Подготовка для YOLOv8 (320x320)
            blob = cv2.resize(frame, (320, 320))
            blob = blob.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            outputs = session.run(None, {input_name: blob})[0]
            outputs = np.squeeze(outputs).T

            boxes, confs = [], []
            for row in outputs:
                prob = row[4] # Вероятность 'person'
                if prob > args["confidence"]:
                    xc, yc, w, h = row[:4]
                    x1 = int((xc - w/2) * (W / 320))
                    y1 = int((yc - h/2) * (H / 320))
                    boxes.append([x1, y1, int(w * (W / 320)), int(h * (H / 320))])
                    confs.append(float(prob))

            indices = cv2.dnn.NMSBoxes(boxes, confs, args["confidence"], 0.45)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y, w, h) = boxes[i]
                    # ИСПОЛЬЗУЕМ KCF
                    try:
                        tracker = cv2.TrackerKCF_create()
                    except AttributeError:
                        tracker = cv2.legacy.TrackerKCF_create()
                    
                    tracker.init(frame, (x, y, w, h))
                    trackers.append(tracker)
        
        # ТРЕКИНГ (в промежутках)
        else:
            for tracker in trackers:
                (success, box) = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    rects.append((x, y, x + w, y + h))

        # Подсчет объектов
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
                    if direction < 0 and centroid[1] < H // 2:
                        print("+1")
                        totalUp += 1
                        move_out.append(totalUp)
                        to.counted = True
                    elif direction > 0 and centroid[1] > H // 2:
                        print("-1")
                        totalDown += 1
                        move_in.append(totalDown)
                        to.counted = True

            trackableObjects[objectID] = to
            # cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # Отрисовка
        # cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        # cv2.putText(frame, f"In: {totalDown} Out: {totalUp}", (10, 25), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # cv2.imshow("KCF + YOLOv8", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        totalFrames += 1
        fps.update()

    fps.stop()
    if fps.elapsed() > 0:
        print(f"[INFO] FPS: {fps.fps():.2f}")
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    people_counter()