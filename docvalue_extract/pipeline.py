import cv2

from .stage1_ocr import OcrLocalizer
from .stage2_vlm import VlmValueExtractor
from .stage3_align import align
from .stage4_contours import detect_char_boxes
from .field_types import normalize_field_type
from .types import Field, CharBox


class ExtractionPipeline:
    """Four-stage field extraction: OCR localization, VLM value
    identification, fuzzy alignment, character contour detection.

    Output is a list of Field records, each with an OCR-grounded
    bounding box, a normalized semantic type, the recognized text,
    and per-character boxes. No document-level layout is produced;
    the output is a set of independent field records."""

    def __init__(self, languages=None, device="cuda", detect_chars=True):
        self.ocr = OcrLocalizer(languages=languages, gpu=(device == "cuda"))
        self.vlm = VlmValueExtractor(device=device)
        self.detect_chars = detect_chars

    def run(self, image_path: str) -> list[Field]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"could not read image: {image_path}")

        # Stage 1: spatial localization.
        regions = self.ocr.localize(image)

        # Stage 2: semantic value identification. The VLM reads the
        # source image, not the OCR output; it decides which text is
        # a value worth analyzing.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        values = self.vlm.extract(rgb)

        # Stage 3: fuzzy alignment. Match each VLM value to its
        # OCR-grounded box. Returns (bbox, field_type, text) tuples.
        aligned = align(values, regions)

        # Stage 4: per-character contour boxes within each field.
        fields = []
        for bbox, field_type, text in aligned:
            char_boxes = []
            if self.detect_chars:
                char_boxes = [
                    CharBox(bbox=cb)
                    for cb in detect_char_boxes(image, bbox)
                ]
            fields.append(Field(
                bbox=bbox,
                field_type=normalize_field_type(field_type),
                text=text,
                char_boxes=char_boxes,
            ))
        return fields