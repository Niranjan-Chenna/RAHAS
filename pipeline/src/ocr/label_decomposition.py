"""Decompose RAHAS character folder labels into OCR metadata targets.

This module uses folder-label text only. It does not inspect, binarize,
skeletonize, thin, or otherwise process character images.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import unicodedata


NONE = "none"

VOWEL_SIGNS: tuple[str, ...] = ("ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ")
NASAL_SIGNS: tuple[str, ...] = ("ं", "ँ")

_DEVANAGARI_BLOCK_START = ord("\u0900")
_DEVANAGARI_BLOCK_END = ord("\u097f")


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """Parsed metadata for one character folder label."""

    full_label: str
    base_glyph: str
    modifier: str = NONE
    nasal: str = NONE

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _normalize_label(label: str) -> str:
    normalized = unicodedata.normalize("NFC", label.strip())
    if not normalized:
        raise ValueError("label is empty")
    return normalized


def _contains_devanagari(label: str) -> bool:
    return any(_DEVANAGARI_BLOCK_START <= ord(char) <= _DEVANAGARI_BLOCK_END for char in label)


def parse_label(label: str) -> LabelRecord:
    """Parse a full folder label into base glyph, vowel modifier, and nasal mark.

    Examples:
    - क, का, कि, कु, के, को, कं all map to base_glyph क.
    - ब्रा maps to base_glyph ब्र and modifier ा.
    - ध्रू maps to base_glyph ध्र and modifier ू.
    - ह्वे maps to base_glyph ह्व and modifier े.

    Independent vowels such as अ and ओ remain their own base glyphs unless
    a trailing vowel sign is present in a future label.
    """

    full_label = _normalize_label(label)
    if not _contains_devanagari(full_label):
        raise ValueError(f"label is not Devanagari: {full_label!r}")

    remaining = full_label
    nasal = NONE
    if remaining[-1] in NASAL_SIGNS:
        nasal = remaining[-1]
        remaining = remaining[:-1]
        if not remaining:
            raise ValueError(f"label has nasal mark without base glyph: {full_label!r}")

    modifier = NONE
    if remaining[-1] in VOWEL_SIGNS:
        modifier = remaining[-1]
        remaining = remaining[:-1]
        if not remaining:
            raise ValueError(f"label has vowel sign without base glyph: {full_label!r}")

    base_glyph = remaining
    return LabelRecord(
        full_label=full_label,
        base_glyph=base_glyph,
        modifier=modifier,
        nasal=nasal,
    )


def list_character_labels(character_root: Path | str) -> list[str]:
    """Return deterministic character folder labels from a characters/ root."""

    root = Path(character_root)
    if not root.exists():
        raise FileNotFoundError(f"character root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"character root is not a directory: {root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _ordered_modifiers(records: list[LabelRecord]) -> list[str]:
    present = {record.modifier for record in records}
    return [NONE] + [modifier for modifier in VOWEL_SIGNS if modifier in present]


def _ordered_nasals(records: list[LabelRecord]) -> list[str]:
    present = {record.nasal for record in records}
    return [NONE] + [nasal for nasal in NASAL_SIGNS if nasal in present]


def build_label_maps(character_root: Path | str) -> dict[str, Any]:
    """Build deterministic label metadata maps for OCR training.

    Returns a dictionary containing sorted label lists, parsed records, and
    index maps for full labels, base glyphs, modifiers, and nasal marks.
    """

    full_labels = list_character_labels(character_root)
    records = [parse_label(label) for label in full_labels]

    base_glyphs = sorted({record.base_glyph for record in records})
    modifiers = _ordered_modifiers(records)
    nasals = _ordered_nasals(records)

    full_label_to_index = {label: index for index, label in enumerate(full_labels)}
    base_glyph_to_index = {label: index for index, label in enumerate(base_glyphs)}
    modifier_to_index = {label: index for index, label in enumerate(modifiers)}
    nasal_to_index = {label: index for index, label in enumerate(nasals)}

    return {
        "full_labels": full_labels,
        "base_glyphs": base_glyphs,
        "modifiers": modifiers,
        "nasals": nasals,
        "records": records,
        "label_to_record": {record.full_label: record for record in records},
        "full_label_to_index": full_label_to_index,
        "base_glyph_to_index": base_glyph_to_index,
        "modifier_to_index": modifier_to_index,
        "nasal_to_index": nasal_to_index,
    }
