def _normalize(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _score(vlm_text: str, ocr_text: str) -> float:
    a, b = _normalize(vlm_text), _normalize(ocr_text)
    if not a or not b:
        return 0.0

    # Tier 1: exact match on normalized strings.
    if a == b:
        return 1.0

    # Tier 2: substring containment, scored by length ratio.
    if a in b or b in a:
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        return len(shorter) / len(longer)

    # Tier 3: character overlap, fraction of the shorter string's
    # characters present anywhere in the longer, scaled by length.
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    present = sum(1 for c in shorter if c in longer)
    overlap = present / len(shorter)
    if overlap > 0.7:
        return overlap * (len(shorter) / len(longer))
    return 0.0


def align(vlm_values, ocr_regions):
    """Greedy one-to-one assignment of VLM values to OCR regions.

    For each VLM value, pick the highest-scoring unclaimed OCR
    region above 0.5. Once an OCR region is claimed it cannot be
    reused. Values that match nothing are dropped.
    """
    claimed = set()
    fields = []

    for value in vlm_values:
        best_idx = None
        best_score = 0.5  # acceptance floor

        for idx, region in enumerate(ocr_regions):
            if idx in claimed:
                continue
            s = _score(value.text, region.text)
            if s > best_score:
                best_score = s
                best_idx = idx

        if best_idx is not None:
            claimed.add(best_idx)
            region = ocr_regions[best_idx]
            fields.append((region.bbox, value.field_type, region.text))

    return fields