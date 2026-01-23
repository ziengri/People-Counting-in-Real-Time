from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from itertools import zip_longest
from utils.mailer import Mailer
from imutils.video import FPS
from utils import thread
import onnxruntime as ort
import numpy as np
import threading
import argparse
import datetime
import schedule
import logging
import imutils
import time
import json
import csv
import cv2

# Инициализация логгера
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

with open("utils/config.json", "r") as file:
    config = json.load(file)

def parse_arguments():
    ap = argparse.ArgumentParser()
    # Для YOLOv8 нам нужен только путь к .onnx файлу
    ap.add_argument("-m", "--model", required=True, help="path to YOLOv8 ONNX model")
    ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.3, help="min probability")
    ap.add_argument("-s", "--skip-frames", type=int, default=10, help="# of skip frames")
    return vars(ap.parse_args())

def send_mail():
    pass
    # Mailer().send(config["Email_Receive"])

def log_data(move_in, in_time, move_out, out_time):
    pass
    data = [move_in, in_time, move_out, out_time]
    export_data = zip_longest(*data, fillvalue='')
    with open('utils/data/logs/counting_data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
        wr.writerows(export_data)

def people_counter():
    args = parse_arguments()
    start_time = time.time()

    # Инициализация ONNX Runtime для YOLOv8
    # Используем только CPU (так как на J1900 нет GPU)
    session = ort.InferenceSession(args["model"], providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Настройка видеопотока
    if not args.get("input", False):
        logger.info("Starting live stream...")
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
    else:
        logger.info("Starting video file...")
        vs = cv2.VideoCapture(args["input"])

    writer = None
    W, H = None, None

    ct = CentroidTracker(maxDisappeared=60, maxDistance=70)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown, totalUp = 0, 0
    total, move_out, move_in, out_time, in_time = [], [], [], [], []

    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        if args["input"] is not None and frame is None:
            break

        # ОПТИМИЗАЦИЯ: Ресайз до 320 для J1900 (соответствует размеру экспорта ONNX)
        frame = imutils.resize(frame, width=320)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []

        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []

            # Препроцессинг YOLOv8 (320x320)
            blob = cv2.resize(frame, (320, 320))
            blob = blob.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            # Инференс
            outputs = session.run(None, {input_name: blob})[0]
            outputs = np.squeeze(outputs).T  # Результат: (2100, 84)

            boxes, confs = [], []
            for row in outputs:
                prob = row[4] # Вероятность класса 'person' (ID 0)
                if prob > args["confidence"]:
                    xc, yc, w, h = row[:4]
                    # Масштабируем координаты к текущему размеру кадра (W, H)
                    x1 = int((xc - w/2) * (W / 320))
                    y1 = int((yc - h/2) * (H / 320))
                    boxes.append([x1, y1, int(w * (W / 320)), int(h * (H / 320))])
                    confs.append(float(prob))

            # NMS (Фильтрация наложений)
            indices = cv2.dnn.NMSBoxes(boxes, confs, args["confidence"], 0.45)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y, w, h) = boxes[i]
                    # Инициализируем MOSSE трекер (самый быстрый)
                    tracker = cv2.legacy.TrackerMOSSE_create()
                    tracker.init(frame, (x, y, w, h))
                    trackers.append(tracker)
        else:
            # ТРЕКИНГ
            for tracker in trackers:
                status = "Tracking"
                (success, box) = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    rects.append((x, y, x + w, y + h))

        # Линия подсчета (горизонтальная по центру)
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
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
                    # Логика входа/выхода
                    if direction < 0 and centroid[1] < H // 2:
                        print("+1")
                        totalUp += 1
                        move_out.append(totalUp)
                        out_time.append(datetime.datetime.now().strftime("%H:%M:%S"))
                        to.counted = True
                    elif direction > 0 and centroid[1] > H // 2:
                        print("-1")
                        totalDown += 1
                        move_in.append(totalDown)
                        in_time.append(datetime.datetime.now().strftime("%H:%M:%S"))
                        
                        # Лимит людей
                        current_inside = len(move_in) - len(move_out)
                        if current_inside >= config.get("Threshold", 100):
                            if config.get("ALERT"):
                                threading.Thread(target=send_mail, daemon=True).start()
                        to.counted = True

            trackableObjects[objectID] = to
            cv2.putText(frame, f"ID {objectID}", (centroid[0]-10, centroid[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.circle(frame, (centroid[0], centroid[1]), 3, (255, 255, 255), -1)

        # Отрисовка инфо
        total_inside = len(move_in) - len(move_out)
        cv2.putText(frame, f"In: {totalDown}  Out: {totalUp}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Inside: {total_inside}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, H - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if config["Log"]:
            log_data(move_in, in_time, move_out, out_time)
        if writer is not None:
            writer.write(frame)

        cv2.imshow("Bus Counter YOLOv8", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        totalFrames += 1
        fps.update()

        # Таймер автостопа (8 часов)
        if config["Timer"] and (time.time() - start_time) > 28800:
            break

    fps.stop()
    logger.info(f"FPS: {fps.fps():.2f}")
    if config["Thread"]: vs.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    if config["Scheduler"]:
        schedule.every().day.at("09:00").do(people_counter)
        while True:
            schedule.run_pending()
            time.sleep(1)
    else:
        people_counter()