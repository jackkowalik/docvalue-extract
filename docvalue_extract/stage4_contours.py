import cv2
import numpy as np

C_MAX = 12  # cap on characters per field


def detect_char_boxes(image, field_bbox):
    """Character-level contour detection within a field region.

    Crops with asymmetric padding (5% horizontal, 20% vertical),
    adaptive-thresholds, morphologically closes, and extracts
    external contours as candidate character boxes.
    """
    x1, y1, x2, y2 = field_bbox
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(0.05 * w), int(0.20 * h)

    H, W = image.shape[:2]
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(W, x2 + pad_x), min(H, y2 + pad_y)
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return []

    crop_h, crop_w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10,
    )
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if not (0.15 * crop_h <= bh <= 0.95 * crop_h):
            continue
        if bw < 2 or bw > 0.5 * crop_w:
            continue
        if bw * bh < 10:
            continue
        boxes.append([bx, by, bx + bw, by + bh])

    boxes.sort(key=lambda b: b[0])
    boxes = _merge_overlapping(boxes)[:C_MAX]

    # Map back to full-image coordinates.
    return [(bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1)
            for bx1, by1, bx2, by2 in boxes]


def _merge_overlapping(boxes):
    """Merge left-to-right pairs whose horizontal overlap exceeds
    50% of the narrower box's width."""
    if not boxes:
        return []
    merged = [boxes[0]]
    for cur in boxes[1:]:
        prev = merged[-1]
        overlap = min(prev[2], cur[2]) - max(prev[0], cur[0])
        narrower = min(prev[2] - prev[0], cur[2] - cur[0])
        if narrower > 0 and overlap > 0.5 * narrower:
            merged[-1] = [
                min(prev[0], cur[0]), min(prev[1], cur[1]),
                max(prev[2], cur[2]), max(prev[3], cur[3]),
            ]
        else:
            merged.append(cur)
    return merged