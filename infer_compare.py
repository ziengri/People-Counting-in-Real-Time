import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        description="Сравнение инференса YOLO .pt и OpenVINO модели на одном видео"
    )
    p.add_argument("--video", required=True, help="Путь к входному видео")
    p.add_argument("--model-a", required=True, help="Путь к первой модели (.pt или OpenVINO export)")
    p.add_argument("--model-b", required=True, help="Путь ко второй модели (.pt или OpenVINO export)")
    p.add_argument("--name-a", default="model_a", help="Имя первой модели в отчёте")
    p.add_argument("--name-b", default="model_b", help="Имя второй модели в отчёте")
    p.add_argument("--imgsz", type=int, default=256, help="Размер входа для модели")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--device", default="cpu", help="Устройство. Для честного сравнения лучше cpu")
    p.add_argument("--save-video", action="store_true", help="Сохранить размеченные видео")
    p.add_argument("--save-csv", action="store_true", help="Сохранить CSV со статистикой по кадрам")
    p.add_argument("--warmup", type=int, default=3, help="Количество прогонов warmup")
    p.add_argument("--max-frames", type=int, default=0, help="Ограничить число кадров, 0 = всё видео")
    p.add_argument("--output-dir", default="runs/compare_infer", help="Папка для результатов")
    return p.parse_args()


def open_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, total_frames


def load_model(path: str):
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")
    return YOLO(str(path_obj))


def warmup_model(model, frame, imgsz, conf, iou, device, repeats=3):
    for _ in range(max(repeats, 0)):
        model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_writer(save_path: Path, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))


def run_benchmark(model_path, model_name, video_path, args):
    print(f"\n{'=' * 80}")
    print(f"Старт: {model_name}")
    print(f"Путь:  {model_path}")

    model = load_model(model_path)

    cap, fps, width, height, total_frames = open_video(video_path)
    ok, first_frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр для warmup")

    warmup_model(model, first_frame, args.imgsz, args.conf, args.iou, args.device, args.warmup)

    cap, fps, width, height, total_frames = open_video(video_path)

    out_dir = ensure_dir(Path(args.output_dir))
    writer = None
    if args.save_video:
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_name)
        save_path = out_dir / f"{safe_name}.mp4"
        writer = make_writer(save_path, fps, width, height)
    else:
        save_path = None

    csv_rows = []
    frame_idx = 0
    measured_times = []
    total_detections = 0

    video_wall_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            break

        t0 = time.perf_counter()
        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        infer_time = time.perf_counter() - t0
        measured_times.append(infer_time)

        result = results[0]
        det_count = 0 if result.boxes is None else len(result.boxes)
        total_detections += det_count

        if writer is not None:
            plotted = result.plot()
            if plotted.shape[1] != width or plotted.shape[0] != height:
                plotted = cv2.resize(plotted, (width, height))
            writer.write(plotted)

        if args.save_csv:
            csv_rows.append([
                frame_idx,
                round(infer_time * 1000, 3),
                det_count,
            ])

    video_wall_total = time.perf_counter() - video_wall_start

    cap.release()
    if writer is not None:
        writer.release()

    if frame_idx == 0:
        raise RuntimeError("Видео не содержит кадров")

    avg_ms = (sum(measured_times) / len(measured_times)) * 1000.0
    pure_fps = len(measured_times) / sum(measured_times)
    end_to_end_fps = frame_idx / video_wall_total

    if args.save_csv:
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_name)
        csv_path = out_dir / f"{safe_name}_stats.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["frame", "infer_ms", "detections"])
            writer_csv.writerows(csv_rows)
    else:
        csv_path = None

    summary = {
        "name": model_name,
        "model_path": str(model_path),
        "frames": frame_idx,
        "source_fps": fps,
        "avg_infer_ms": avg_ms,
        "pure_model_fps": pure_fps,
        "end_to_end_fps": end_to_end_fps,
        "total_detections": total_detections,
        "avg_detections_per_frame": total_detections / frame_idx,
        "saved_video": str(save_path) if save_path else None,
        "saved_csv": str(csv_path) if csv_path else None,
    }

    print(f"Кадров обработано      : {summary['frames']}")
    print(f"Среднее инференс, мс   : {summary['avg_infer_ms']:.2f}")
    print(f"FPS модели             : {summary['pure_model_fps']:.2f}")
    print(f"FPS end-to-end         : {summary['end_to_end_fps']:.2f}")
    print(f"Всего детекций         : {summary['total_detections']}")
    print(f"Среднее детекций/кадр  : {summary['avg_detections_per_frame']:.2f}")
    if summary['saved_video']:
        print(f"Видео сохранено        : {summary['saved_video']}")
    if summary['saved_csv']:
        print(f"CSV сохранён           : {summary['saved_csv']}")

    return summary


def print_comparison(a, b):
    print(f"\n{'#' * 80}")
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print(f"{a['name']}: {a['avg_infer_ms']:.2f} ms | pure FPS {a['pure_model_fps']:.2f} | e2e FPS {a['end_to_end_fps']:.2f}")
    print(f"{b['name']}: {b['avg_infer_ms']:.2f} ms | pure FPS {b['pure_model_fps']:.2f} | e2e FPS {b['end_to_end_fps']:.2f}")

    faster = a if a['avg_infer_ms'] < b['avg_infer_ms'] else b
    slower = b if faster is a else a
    speedup = slower['avg_infer_ms'] / faster['avg_infer_ms'] if faster['avg_infer_ms'] > 0 else 0

    print(f"Быстрее: {faster['name']} примерно в {speedup:.2f}x по среднему времени инференса")

    det_diff = a['avg_detections_per_frame'] - b['avg_detections_per_frame']
    print(
        "Разница по среднему числу детекций/кадр: "
        f"{a['name']} - {b['name']} = {det_diff:+.3f}"
    )
    print("#" * 80)


if __name__ == "__main__":
    args = parse_args()
    summary_a = run_benchmark(args.model_a, args.name_a, args.video, args)
    summary_b = run_benchmark(args.model_b, args.name_b, args.video, args)
    print_comparison(summary_a, summary_b)
