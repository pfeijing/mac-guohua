from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


PerspectiveType = Literal[
    "high-distance",
    "deep-distance",
    "level-distance",
    "mixed",
    "unspecified",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PaintingBrief(StrictModel):
    subject: str
    mood: str
    season: str = "unspecified"
    time_of_day: str = "unspecified"
    perspective: PerspectiveType = "unspecified"
    visual_hierarchy: list[str] = Field(default_factory=list)
    intended_regions: list[str] = Field(default_factory=list)
    cultural_motifs: list[str] = Field(default_factory=list)
    color_ink_strategy: str
    composition_summary: str
    negative_space_target: float = Field(default=0.30, ge=0.0, le=0.85)


class Region(StrictModel):
    id: str
    label: str
    polygon: list[list[float]]
    depth_order: int = Field(default=0, ge=0, le=100)
    constraint: str = ""
    description: str = ""
    style_prompt: str = ""

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) < 3:
            raise ValueError("A polygon must contain at least three vertices.")

        cleaned: list[list[float]] = []
        for point in value:
            if len(point) != 2:
                raise ValueError("Each polygon vertex must be [x, y].")
            x = min(1.0, max(0.0, float(point[0])))
            y = min(1.0, max(0.0, float(point[1])))
            cleaned.append([x, y])

        return cleaned


class LayoutPlan(StrictModel):
    canvas_description: str
    regions: list[Region] = Field(min_length=1)
    reasoning_summary: str = ""


class StyleResult(StrictModel):
    region_id: str
    prompt: str
    negative_prompt: str = ""


class GenerationRecord(StrictModel):
    user_prompt: str
    brief: PaintingBrief
    layout: LayoutPlan
    blank_ratio: float
    overlap_penalty: float
    iterations: int
    image_path: str