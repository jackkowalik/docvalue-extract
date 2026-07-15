TEMPLATE_LABELS = {
    "name", "date of birth", "address", "license number", "sex",
    "eyes", "hair", "hgt", "wgt", "class", "expires", "donor",
    "rest", "end",
}

TYPE_ALIASES = {
    "dob": "date",
    "date_of_birth": "date",
    "birthdate": "date",
    "date": "date",
    "name": "name",
    "full_name": "name",
    "address": "address",
    "id_number": "id_number",
    "license_number": "id_number",
    "document_number": "id_number",
}

# Field types dropped as non-value: not consumed downstream for
# reconciliation or verification, so including them only inflates
# the false-positive surface.
NON_VALUE_TYPES = {
    "signature", "donor", "emergency_contact", "restriction",
    "hair", "weight", "eyes", "height",
}


def normalize_field_type(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    return TYPE_ALIASES.get(key, "unknown")


def is_value_field(field_type: str) -> bool:
    return field_type not in NON_VALUE_TYPES