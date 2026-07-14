# docvalue-extract

Field-level value extraction for identity documents. Given a document
image, it returns the filled-in value fields like name, date of birth,
document number, address, each with a bounding box, a semantic type,
the recognized text, and per character bounding boxes.

## Why value fields and not the whole document?

This pipeline prepares a document for a downstream fraud classifier.
Consider a legitimate government issued document that an individual has 
altered to hide or change their identity. The edits are in the values, 
like a date pushed forward by two years, or a swapped name, and not the 
printed template labels, the document layout, or other substrate characteristics.
Working at the field level, rather than on the whole-document image, is what lets
downstream classifiers run across issuing authorities, and reduces the
input variation they have to handle.

## How it works

Four stages:

1. **Localization (EasyOCR).** Finds every text region and returns
   axis-aligned boxes. This is very accurate for smartphone captured
   identity documents, even under relatively bad conditions.
2. **Semantic value identification (Qwen2.5-VL-7B).** Reads the image
   and decides which text is a filled-in value versus a template label,
   returning `{text, field_type}` pairs. A heavy vision-language model is
   used deliberately here because I know that it generalizes
   across issuing authorities and scripts without per-jurisdiction rules. 
   It can easily recognize that `niebieskie` is an eye-color *value* and
   `DATA URODZENIA` is a *label*.
4. **Fuzzy alignment.** Matches each VLM value to its OCR-grounded bbox
   (three tiers: exact, substring, character-overlap).
5. **Character contour detection.** Locates individual character bboxes
   within each field. These can be used by systems ingesting fields and
   characters separately, through attention between them, or other
   mechanisms.

The output is a set of independent field records.

## Benchmark

The Qwen model, under a similar prompt and value-identification configuration
to the one this pipeline uses, achieves a 97.3% field extraction success rate
across 6,407 document samples spanning 223 countries, territories, and 49
document types. Full per-country and per-document-type results: 
https://authorize.earth/products/benchmarks/

## Install

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # needs matching CUDA/torch
```

## Run

```bash
python examples/run.py document.jpg --out fields.json --overlay boxes.png
```

`fields.json` contains the extracted fields; `boxes.png` draws the boxes
on the source image. Add `--show-chars` to draw character boxes,
`--no-chars` to skip Stage 4, `--lang en pl de` for multilingual OCR.

## Hardware

Qwen2.5-VL-7B in bf16 needs roughly 18 GB of VRAM. It fits comfortably on
a 24 GB+ card. On 16 GB cards, cap the input resolution (via the
processor's `max_pixels`). I have not tested quantization.

## License

This project is MIT-licensed. Qwen2.5-VL-7B has an Apache 2.0 license.
EasyOCR has an Apache 2.0 license.
