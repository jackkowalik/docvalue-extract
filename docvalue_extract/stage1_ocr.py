import easyocr

from .types import OcrRegion

_CONF_THRESHOLD = 0.3


class OcrLocalizer:
    def __init__(self, languages=None, gpu=True):
        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    def localize(self, image) -> list[OcrRegion]:
        """Run OCR for spatial localization only. Returns every text
        region above the confidence floor as an axis-aligned box.
        Semantics (value vs label) are decided downstream in Stage 2."""
        results = self.reader.readtext(image, paragraph=False)

        regions = []
        for quad, text, conf in results:
            if conf < _CONF_THRESHOLD:
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            regions.append(OcrRegion(bbox=bbox, text=text, conf=float(conf)))
        return regions