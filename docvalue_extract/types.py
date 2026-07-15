from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OcrRegion:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 axis-aligned
    text: str
    conf: float


@dataclass
class VlmValue:
    text: str
    field_type: str  # name | date | address | id_number | other


@dataclass
class CharBox:
    bbox: tuple[int, int, int, int]


@dataclass
class Field:
    bbox: tuple[int, int, int, int]
    field_type: str
    text: str
    char_boxes: list[CharBox] = field(default_factory=list)