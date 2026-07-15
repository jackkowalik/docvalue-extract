import argparse
import json
from dataclasses import asdict

from docvalue_extract.pipeline import ExtractionPipeline
from docvalue_extract.visualize import draw_overlay


def main():
    ap = argparse.ArgumentParser(
        description="Extract value fields from an identity document."
    )
    ap.add_argument("image", help="path to the document image")
    ap.add_argument("-o", "--out", help="write field JSON to this path")
    ap.add_argument("--overlay", help="write a box overlay image to this path")
    ap.add_argument("--show-chars", action="store_true",
                    help="draw character boxes on the overlay")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-chars", action="store_true",
                    help="skip Stage 4 character contour detection")
    ap.add_argument("--lang", nargs="+", default=["en"],
                    help="EasyOCR language codes")
    args = ap.parse_args()

    pipeline = ExtractionPipeline(
        languages=args.lang,
        device=args.device,
        detect_chars=not args.no_chars,
    )
    fields = pipeline.run(args.image)

    payload = [asdict(f) for f in fields]
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)

    if args.overlay:
        path = draw_overlay(args.image, fields, args.overlay,
                            show_chars=args.show_chars)
        print(f"overlay written to {path}")


if __name__ == "__main__":
    main()
