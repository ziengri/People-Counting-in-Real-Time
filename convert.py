from ultralytics import YOLO

# Загружаем выбранную модель
model = YOLO("./detector/yolo12n.pt") 

# Экспортируем с оптимизацией под CPU
# model.export(
#     format="onnx", 
#     imgsz=320,       # Уменьшаем размер до 320 для скорости на J1900
#     half=False,      # J1900 не любит FP16, оставляем FP32
#     simplify=True    # Убирает лишние узлы графа модели
# )


# 2. Запускаем экспорт
# data="coco8.yaml" заставит скрипт сам скачать нужные файлы
# int8=True включит максимальную оптимизацию для Intel
model.export(
    format="openvino", 
    imgsz=320, 
    int8=True, 
    data="coco128.yaml"
)