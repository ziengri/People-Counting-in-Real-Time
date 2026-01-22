from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream, FPS
from itertools import zip_longest
from utils.mailer import Mailer
from utils import thread
import numpy as np
import threading
import argparse
import datetime
import logging
import imutils
import time
import json
import csv
import cv2

# setup logger
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

# load config
with open("utils/config.json", "r") as file:
    config = json.load(file)

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False, help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4, help="min probability")
    ap.add_argument("-s", "--skip-frames", type=int, default=30, help="# of skip frames")
    return vars(ap.parse_args())

def send_mail():
    Mailer().send(config["Email_Receive"])

def log_data(move_in, in_time, move_out, out_time):
    # Финальная запись логов в файл
    data = [move_in, in_time, move_out, out_time]
    export_data = zip_longest(*data, fillvalue='')
    with open('utils/data/logs/counting_data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
        wr.writerows(export_data)
    logger.info("Logs saved to CSV.")

def people_counter():
    args = parse_arguments()
    start_time = time.time()
    
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    if not args.get("input", False):
        logger.info("Starting live stream...")
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
    else:
        logger.info("Starting video...")
        vs = cv2.VideoCapture(args["input"])

    writer = None
    W, H = None, None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    
    move_out, out_time = [], []
    move_in, in_time = [], []

    fps = FPS().start()

    try:
        while True:
            frame = vs.read()
            frame = frame[1] if args.get("input", False) else frame
            if frame is None:
                break

            frame = imutils.resize(frame, width=500)
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            if args["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

            status = "Waiting"
            rects = []

            # ДЕТЕКЦИЯ
            if totalFrames % args["skip_frames"] == 0:
                status = "Detecting"
                trackers = []
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > args["confidence"]:
                        idx = int(detections[0, 0, i, 1])
                        if CLASSES[idx] != "person":
                            continue

                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")

                        # ИСПОЛЬЗУЕМ KCF ИЗ OPENCV ВМЕСТО DLIB
                        try:
                            # 1. Пытаемся вызвать через новый API (если доступно)
                            tracker = cv2.TrackerMOSSE.create() 
                        except AttributeError:
                            try:
                                # 2. Пытаемся через legacy (самый частый случай для MOSSE в contrib)
                                tracker = cv2.legacy.TrackerMOSSE_create()
                            except AttributeError:
                                # 3. Если совсем всё плохо, откатываемся на KCF
                                logger.warning("MOSSE не найден, использую KCF")
                                tracker = cv2.TrackerKCF.create()
                        tracker.init(frame, (startX, startY, endX - startX, endY - startY))
                        trackers.append(tracker)

            # ТРЕКИНГ
            else:
                for tracker in trackers:
                    status = "Tracking"
                    (success, box) = tracker.update(frame)
                    if success:
                        (x, y, w, h) = [int(v) for v in box]
                        rects.append((x, y, x + w, y + h))

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
                        # Вверх (Выход)
                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            print("+1")
                            move_out.append(totalUp)
                            out_time.append(datetime.datetime.now().strftime("%H:%M:%S"))
                            to.counted = True
                        # Вниз (Вход)
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # Отрисовка инфо
            current_inside = len(move_in) - len(move_out)
            # cv2.putText(frame, f"In: {totalDown} | Out: {totalUp}", (10, 20), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # cv2.putText(frame, f"Inside: {current_inside}", (10, 50), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if writer is not None:
                writer.write(frame)

            # cv2.imshow("Frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

            totalFrames += 1
            fps.update()

            if config["Timer"] and (time.time() - start_time) > 28800:
                break

    finally:
        fps.stop()
        logger.info(f"FPS: {fps.fps():.2f}")
        
        if config["Log"]:
            log_data(move_in, in_time, move_out, out_time)

        if writer:
            writer.release()
        if hasattr(vs, 'stop'):
            vs.stop()
        else:
            vs.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    people_counter()