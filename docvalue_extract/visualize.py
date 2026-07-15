import cv2

_COLORS = {
    "name": (0, 180, 0),
    "date": (0, 140, 255),
    "address": (200, 0, 0),
    "id_number": (0, 0, 220),
    "other": (128, 128, 128),
    "unknown": (128, 128, 128),
}


def draw_overlay(image_path: str, fields, out_path: str,
                 show_chars: bool = False):
    """Draw field boxes (and optionally character boxes) on the
    source image and write the result to out_path."""
    image = cv2.imread(image_path)

    for f in fields:
        x1, y1, x2, y2 = f.bbox
        color = _COLORS.get(f.field_type, _COLORS["other"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, f.field_type, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        if show_chars:
            for cb in f.char_boxes:
                cx1, cy1, cx2, cy2 = cb.bbox
                cv2.rectangle(image, (cx1, cy1), (cx2, cy2),
                              (180, 180, 180), 1)

    cv2.imwrite(out_path, image)
    return out_path