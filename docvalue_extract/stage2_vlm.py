import json
import threading

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

from .types import VlmValue
from .field_types import is_value_field

_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Serializes GPU inference within a process; the model is not safe
# to call concurrently from multiple threads on one device.
_inference_lock = threading.Lock()

PROMPT = """This is an identity document. List all the VALUE fields (the actual filled-in data).

DO NOT include static template labels like "NAME", "DATE OF BIRTH", "ADDRESS", "LICENSE NUMBER", "SEX", "EYES", "HAIR", "HGT", "WGT", "CLASS", "EXPIRES", "DONOR", "REST", "END", etc.

ONLY list the actual values like personal names, dates (in MM-DD-YYYY format), addresses, ID numbers, etc.

Return a JSON array with the text content and field type:
[{"text": "actual value text", "field_type": "name|date|address|id_number|other"}, ...]

Output ONLY the JSON array."""


class VlmValueExtractor:
    def __init__(self, model_id: str = _MODEL_ID, device: str = "cuda"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=1280*28*28)

    def extract(self, image) -> list[VlmValue]:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with _inference_lock:
            with torch.inference_mode():
                ids = self.model.generate(**inputs, max_new_tokens=2048)

        response = self.processor.batch_decode(
            [o[len(i):] for i, o in zip(inputs.input_ids, ids)],
            skip_special_tokens=True,
        )[0]

        del inputs, ids
        torch.cuda.empty_cache()

        return self._parse(response)

    @staticmethod
    def _parse(response: str) -> list[VlmValue]:
        """Defensive parse: tolerate code fences, surrounding prose,
        and trailing junk; return an empty list on any failure."""
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end <= start:
            return []

        try:
            raw = json.loads(cleaned[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return []

        values = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            ftype = str(item.get("field_type", "other")).strip().lower()
            if not is_value_field(ftype):
                continue
            values.append(VlmValue(text=text, field_type=ftype))
        return values
