import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Car parking slot occupancy detector.")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0), video file, or image file path.",
    )
    parser.add_argument(
        "--slots",
        default="slots.json",
        help="Path to slot ROI JSON file.",
    )
    parser.add_argument(
        "--occupied-threshold",
        type=int,
        default=900,
        help="White-pixel threshold used to classify a slot as occupied.",
    )
    parser.add_argument(
        "--select-slots",
        action="store_true",
        help="Open interactive slot selector and save ROI rectangles to --slots.",
    )
    return parser.parse_args()


def load_slots(slots_path: Path) -> list[tuple[int, int, int, int]]:
    if not slots_path.exists():
        return []
    with slots_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = data.get("slots", [])
    slots: list[tuple[int, int, int, int]] = []
    for item in data:
        x, y, w, h = int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])
        if w > 0 and h > 0:
            slots.append((x, y, w, h))
    return slots


def save_slots(slots_path: Path, slots: list[tuple[int, int, int, int]]) -> None:
    slots_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [{"x": x, "y": y, "w": w, "h": h} for x, y, w, h in slots]
    with slots_path.open("w", encoding="utf-8") as file:
        json.dump(serialized, file, indent=2)


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit() and not Path(source).exists():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def read_first_frame(source: str) -> np.ndarray:
    capture = open_capture(source)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from source: {source}")
    return frame


def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    threshold = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
    )
    median = cv2.medianBlur(threshold, 5)
    return cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)


def classify_slots(
    frame: np.ndarray,
    processed: np.ndarray,
    slots: list[tuple[int, int, int, int]],
    occupied_threshold: int,
) -> tuple[np.ndarray, int]:
    free_count = 0
    for x, y, w, h in slots:
        if x < 0 or y < 0 or x + w > processed.shape[1] or y + h > processed.shape[0]:
            continue
        slot_crop = processed[y : y + h, x : x + w]
        pixel_count = cv2.countNonZero(slot_crop)
        is_free = pixel_count < occupied_threshold
        color = (0, 255, 0) if is_free else (0, 0, 255)
        thickness = 3 if is_free else 2
        if is_free:
            free_count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            frame,
            str(pixel_count),
            (x + 4, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    cv2.rectangle(frame, (15, 15), (380, 70), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Free: {free_count} / {len(slots)}",
        (24, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return frame, free_count


def select_slots_interactively(source: str, slots_path: Path) -> None:
    frame = read_first_frame(source)
    slots: list[tuple[int, int, int, int]] = []
    start_point: tuple[int, int] | None = None
    current_frame = frame.copy()

    def redraw() -> None:
        nonlocal current_frame
        current_frame = frame.copy()
        for x, y, w, h in slots:
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(
            current_frame,
            "Drag to add slot | s: save | q: quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def mouse_handler(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        nonlocal start_point, current_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and start_point is not None:
            redraw()
            sx, sy = start_point
            cv2.rectangle(current_frame, (sx, sy), (x, y), (0, 255, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP and start_point is not None:
            sx, sy = start_point
            left, top = min(sx, x), min(sy, y)
            width, height = abs(x - sx), abs(y - sy)
            if width > 5 and height > 5:
                slots.append((left, top, width, height))
            start_point = None
            redraw()

    redraw()
    window_name = "Slot Selector"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_handler)
    while True:
        cv2.imshow(window_name, current_frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_slots(slots_path, slots)
            break
    cv2.destroyAllWindows()


def run_image_mode(
    image_path: Path, slots: list[tuple[int, int, int, int]], occupied_threshold: int
) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not load image: {image_path}")
    output, _ = classify_slots(frame, preprocess(frame), slots, occupied_threshold)
    cv2.imshow("Parking Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_mode(
    source: str, slots: list[tuple[int, int, int, int]], occupied_threshold: int
) -> None:
    capture = open_capture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            if source.isdigit() and not Path(source).exists():
                continue
            break
        output, _ = classify_slots(frame, preprocess(frame), slots, occupied_threshold)
        cv2.imshow("Parking Detection", output)
        if (cv2.waitKey(20) & 0xFF) == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    slots_path = Path(args.slots)

    if args.select_slots:
        select_slots_interactively(args.source, slots_path)

    slots = load_slots(slots_path)
    if not slots:
        raise RuntimeError(
            "No parking slots configured. Use --select-slots or provide a slots JSON file."
        )

    source_path = Path(args.source)
    if source_path.suffix.lower() in IMAGE_EXTENSIONS and source_path.exists():
        run_image_mode(source_path, slots, args.occupied_threshold)
    else:
        run_video_mode(args.source, slots, args.occupied_threshold)


if __name__ == "__main__":
    main()
