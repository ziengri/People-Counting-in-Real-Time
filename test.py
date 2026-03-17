# import argparse
# import time

# import cv2
# import numpy as np


# def get_output_layers(net):
#     ln = net.getLayerNames()
#     return [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--cfg", required=True, help="path to yolov3-tiny-custom.cfg")
#     ap.add_argument("--weights", required=True, help="path to yolov3-tiny-custom_final.weights")
#     ap.add_argument("--input", required=True, help="path to video file (e.g. 4.mp4)")
#     ap.add_argument("--debug", type=int, default=1, help="1=show window, 0=no window")
#     args = ap.parse_args()

#     net = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#     out_layers = get_output_layers(net)

#     cap = cv2.VideoCapture(args.input)
#     if not cap.isOpened():
#         raise RuntimeError(f"Can't open video: {args.input}")

#     INP_W, INP_H = 160, 160
#     CONF_TH = 0.35
#     NMS_TH = 0.45

#     n_frames = 0
#     t_start = time.perf_counter()

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break

#         H, W = frame.shape[:2]

#         blob = cv2.dnn.blobFromImage(
#             frame, 1 / 255.0, (INP_W, INP_H), swapRB=True, crop=False
#         )
#         net.setInput(blob)
#         outs = net.forward(out_layers)

#         boxes, confs = [], []
#         for out in outs:
#             for det in out:
#                 obj = float(det[4])
#                 cls = float(det[5])  # classes=1
#                 conf = obj * cls
#                 if conf < CONF_TH:
#                     continue

#                 cx, cy, bw, bh = det[0:4]
#                 x = int((cx - bw / 2) * W)
#                 y = int((cy - bh / 2) * H)
#                 w = int(bw * W)
#                 h = int(bh * H)

#                 boxes.append([x, y, w, h])
#                 confs.append(conf)

#         idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_TH, NMS_TH)

#         if args.debug == 1:
#             if len(idxs) > 0:
#                 for i in idxs.flatten():
#                     x, y, w, h = boxes[i]
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

#             cv2.imshow("YOLOv3-tiny custom (CPU)", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         n_frames += 1

#     t_end = time.perf_counter()
#     cap.release()
#     if args.debug == 1:
#         cv2.destroyAllWindows()

#     seconds = max(1e-9, (t_end - t_start))
#     fps = n_frames / seconds
#     print(f"FPS: {fps:.2f}")


# if __name__ == "__main__":
#     main()
from ultralytics import YOLO

model = YOLO("yolo26n-head.pt")

model.predict(
    source="4.mp4",
    imgsz=240,      # размер входа 240px
    show=True,
    save=True
)
