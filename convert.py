# Экспортируем с оптимизацией под CPU
# model.export(
#     format="onnx", 
#     imgsz=320,       # Уменьшаем размер до 320 для скорости на J1900
#     half=False,      # J1900 не любит FP16, оставляем FP32
#     simplify=True    # Убирает лишние узлы графа модели
# )
from ultralytics import YOLO

# Загружаем выбранную модель
model = YOLO("yolo26n-head.pt") 

# 2. Запускаем экспорт
# data="coco8.yaml" заставит скрипт сам скачать нужные файлы
# int8=True включит максимальную оптимизацию для Intel
model.export(
    format="openvino", 
    imgsz=256,
    int8=False,
    data="C:/Users/komputer/Desktop/projects/People-Counting-in-Real-Time/datasets/bus/data.yaml"
)