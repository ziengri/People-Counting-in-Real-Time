
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import openvino as ov
import numpy as np
import argparse
import datetime
import imutils
import time
import json
import os
import cv2

# Инициализация конфигурации
with open("utils/config.json", "r") as file:
    config = json.load(file)

def people_counter():
    ap = argparse.ArgumentParser()
    # Теперь передаем путь к .xml файлу OpenVINO
    ap.add_argument("-m", "--model", required=True, help="path to YOLO12 OpenVINO .xml file")
    ap.add_argument("-i", "--input", type=str, help="path to video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.3)
    ap.add_argument("-s", "--skip-frames", type=int, default=20) # Оптимально для J1900
    ap.add_argument("-d", "--debug", type=bool, default=False)

    args = vars(ap.parse_args())

    # 1. Загрузка модели OpenVINO
    print("[INFO] Загрузка модели OpenVINO...")
    core = ov.Core()
    model_ov = core.read_model(model=args["model"])
    compiled_model = core.compile_model(
        model_ov, "CPU",
        {
            "PERFORMANCE_HINT": "THROUGHPUT",
        }
    )    
    output_layer = compiled_model.output(0)

    # 2. Запуск видеопотока
    if not args.get("input", False):
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
    else:
        vs = cv2.VideoCapture(args["input"])
        if not vs.isOpened():
            print(f"[ERROR] Не удалось открыть видео: {args['input']}")
            return

    W, H = None, None
    # Ваши настроенные параметры трекера
    ct = CentroidTracker(maxDisappeared=90, maxDistance=100)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalDown, totalUp = 0, 0

    fps = FPS().start()

    inp = np.empty((1, 3, 320, 320), dtype=np.float32)
    while True:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        if frame is None:
            break

        # Оптимизация размера old
        frame = imutils.resize(frame, width=320)
        if W is None:
            h0, w0 = frame.shape[:2]
            scale = 320 / w0
            new_size = (320, int(h0 * scale))

        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        rects = []

        # ДЕТЕКЦИЯ
        if totalFrames % args["skip_frames"] == 0:
            trackers = []
            
            # Подготовка кадра (BCHW, 320x320, Float32) Старый код
            # blob = cv2.resize(frame, (320, 320))
            # blob = blob.transpose(2, 0, 1) # HWC -> CHW
            # blob = blob.reshape(1, 3, 320, 320).astype(np.float32) / 255.0

            # Подготовка кадра новый код
            img = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
            img = img.transpose(2, 0, 1)  # HWC -> CHW

            # Заполняем уже выделенный буфер inp (без astype и без новых массивов)
            np.multiply(img, 1.0/255.0, out=inp[0], casting="unsafe")

            # Инференс OpenVINO
            results = compiled_model([inp])[output_layer]
            outputs = np.squeeze(results).T # Формат (2100, 84) для YOLO12

            boxes, confs = [], []
            for row in outputs:
                # В YOLO12/v8 индексы 0-3 это box, 4+ это классы. 4 - person.
                prob = row[4] 
                if prob > args["confidence"]:
                    xc, yc, w, h = row[:4]
                    # Масштабируем координаты обратно к размеру кадра (320)
                    x1 = int((xc - w/2) * (W / 320))
                    y1 = int((yc - h/2) * (H / 320))
                    boxes.append([x1, y1, int(w * (W / 320)), int(h * (H / 320))])
                    confs.append(float(prob))

            indices = cv2.dnn.NMSBoxes(boxes, confs, args["confidence"], 0.45)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y, w, h) = boxes[i]

                    # ВАЖНО: rects для CentroidTracker на кадре детекции
                    rects.append((x, y, x + w, y + h))

                    # трекер как и было
                    tracker = cv2.TrackerKCF_create() if hasattr(cv2, "TrackerKCF_create") else cv2.legacy.TrackerKCF_create()
                    tracker.init(frame, (x, y, w, h))
                    trackers.append(tracker)
        
        # ТРЕКИНГ
        else:
            for tracker in trackers:
                (success, box) = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    rects.append((x, y, x + w, y + h))

        # Логика подсчета CentroidTracker
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
                    # Условие: пересечение линии + минимальная история кадров
                    if direction < 0 and centroid[1] < H // 2 and len(to.centroids) > 10:
                        totalUp += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > H // 2 and len(to.centroids) > 10:
                        totalDown += 1
                        to.counted = True

            trackableObjects[objectID] = to
            if args["debug"]:
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # Отрисовка
        if args["debug"]:
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            cv2.putText(frame, f"In: {totalDown} Out: {totalUp}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("OpenVINO INT8 + KCF", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        totalFrames += 1
        fps.update()

    fps.stop()
    print(f"[INFO] Итого IN: {totalDown}")
    print(f"[INFO] Итого OUT: {totalUp}")
    print(f"[INFO] Средний FPS: {fps.fps():.2f}")
    
    if args["debug"]:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    people_counter()