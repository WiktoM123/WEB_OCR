from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_emnist import EMNIST_BALANCED_LABELS, EMNISTCNN, NUM_CLASSES


EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def load_model(checkpoint_path: Path, device: torch.device) -> EMNISTCNN:
    model = EMNISTCNN(num_classes=NUM_CLASSES)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    # Supports both plain state_dict and checkpoint dictionaries.
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_character(roi: np.ndarray) -> np.ndarray:
    """Converts a single binary ROI into centered 28x28 EMNIST-like input."""
    h, w = roi.shape
    max_dim = max(h, w)
    pad = int(max_dim * 0.2)

    square = np.zeros((max_dim + 2 * pad, max_dim + 2 * pad), dtype=np.uint8)
    start_y = pad + (max_dim - h) // 2
    start_x = pad + (max_dim - w) // 2
    square[start_y : start_y + h, start_x : start_x + w] = roi

    return cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )
    binary = cv2.bitwise_or(binary_otsu, binary_adaptive)

    # Closing reconnects weak pen strokes; opening trims tiny speckle noise.
    close_kernel = np.ones((2, 2), dtype=np.uint8)
    open_kernel = np.ones((2, 2), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return binary


def find_bounding_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    contours_result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

    raw_boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w > 3 and h > 8 and area > 20:
            raw_boxes.append((x, y, w, h))

    if not raw_boxes:
        return []

    heights = np.array([h for _, _, _, h in raw_boxes], dtype=np.float32)
    areas = np.array([w * h for _, _, w, h in raw_boxes], dtype=np.float32)
    median_h = float(np.median(heights))
    median_area = float(np.median(areas))

    min_h = max(6.0, median_h * 0.30)
    max_h = median_h * 3.0
    min_area = max(12.0, median_area * 0.10)

    boxes: List[Tuple[int, int, int, int]] = []
    for x, y, w, h in raw_boxes:
        box_area = w * h
        aspect = w / max(h, 1)
        if h < min_h or h > max_h:
            continue
        if box_area < min_area:
            continue
        if aspect > 3.0:
            continue
        boxes.append((x, y, w, h))

    # If dynamic thresholds drop too many glyphs (mixed-size handwriting), prefer coarse detections.
    min_kept = max(1, int(len(raw_boxes) * 0.6))
    if len(boxes) >= min_kept:
        return boxes

    # Fallback: dynamic filtering too strict for this sample.
    return raw_boxes


def sort_boxes_left_to_right(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    return sorted(boxes, key=lambda b: b[0])


def compute_space_threshold(boxes: List[Tuple[int, int, int, int]]) -> float:
    if not boxes:
        return float("inf")
    avg_width = sum(w for _, _, w, _ in boxes) / len(boxes)
    return avg_width * 0.75


def recognize_binary_text(
    binary: np.ndarray,
    model: EMNISTCNN,
    device: torch.device,
    eval_transform=EVAL_TRANSFORM,
) -> Tuple[str, List[Tuple[int, int, int, int]], List[float]]:
    boxes = sort_boxes_left_to_right(find_bounding_boxes(binary))
    if not boxes:
        return "", [], []

    space_threshold = compute_space_threshold(boxes)
    text_parts: List[str] = []
    confidences: List[float] = []

    for i, (x, y, w, h) in enumerate(boxes):
        if i > 0:
            prev_x, _, prev_w, _ = boxes[i - 1]
            gap = x - (prev_x + prev_w)
            if gap > space_threshold:
                text_parts.append(" ")

        roi = binary[y : y + h, x : x + w]
        char_img = prepare_character(roi)
        tensor = eval_transform(Image.fromarray(char_img)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)

        predicted_char = EMNIST_BALANCED_LABELS[int(pred_idx.item())]
        confidences.append(float(conf.item()))
        text_parts.append(predicted_char)

    predicted_text = "".join(text_parts)
    return predicted_text, boxes, confidences

