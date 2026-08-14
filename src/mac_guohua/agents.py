from __future__ import annotations

import json
import os
import time
from typing import TypeVar, Type

from openai import OpenAI
from pydantic import BaseModel

from .retrieval import KnowledgeRetriever, format_documents
from .schemas import LayoutPlan, PaintingBrief, Region, StyleResult


T = TypeVar("T", bound=BaseModel)


class StructuredLLM:
    def __init__(
        self,
        model: str,
        max_retries: int = 3,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        base_url = os.getenv("OPENAI_BASE_URL") or None
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float,
    ) -> T:
        schema = response_model.model_json_schema()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_model.__name__,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )

                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("The LLM returned empty content.")

                return response_model.model_validate(json.loads(content))

            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)

        raise RuntimeError(
            f"Structured LLM generation failed: {last_error}"
        ) from last_error


class DirectorAgent:
    def __init__(
        self,
        llm: StructuredLLM,
        retriever: KnowledgeRetriever,
        temperature: float = 0.7,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.temperature = temperature
        self.top_k = top_k

    def run(self, user_prompt: str) -> PaintingBrief:
        docs = self.retriever.query(
            user_prompt,
            category="symbolism",
            top_k=self.top_k,
        )

        system = """
You are the Director Agent of a Traditional Chinese Painting studio.
Convert the user's request into a culturally grounded painting brief.

You must:
1. Identify subject, mood, season and perspective.
2. Select high-distance, deep-distance, level-distance, mixed, or unspecified.
3. Define a clear visual hierarchy.
4. List intended semantic regions.
5. Treat blank space as an active element.
6. Avoid claiming historical facts not present in the supplied knowledge.
7. Return only data conforming to the requested JSON schema.
""".strip()

        user = f"""
USER REQUEST:
{user_prompt}

RETRIEVED SYMBOLIC KNOWLEDGE:
{format_documents(docs)}

Create a painting brief. Use a negative-space target near 0.30 unless the
request clearly calls for a denser or more minimal composition.
""".strip()

        return self.llm.generate(
            system,
            user,
            PaintingBrief,
            self.temperature,
        )


class CompositionAgent:
    def __init__(
        self,
        llm: StructuredLLM,
        retriever: KnowledgeRetriever,
        temperature: float = 0.2,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.temperature = temperature
        self.top_k = top_k

    def run(
        self,
        brief: PaintingBrief,
        correction_feedback: str = "",
    ) -> LayoutPlan:
        query = (
            f"{brief.subject}; {brief.perspective}; "
            f"{brief.composition_summary}; negative space"
        )
        docs = self.retriever.query(
            query,
            category="composition",
            top_k=self.top_k,
        )

        system = """
You are the Composition Agent for Traditional Chinese Painting.

Generate a normalized polygon layout in the [0,1] x [0,1] coordinate system.

Rules:
1. Every region must have a unique short id.
2. Use semantic labels such as sky, fog, mist, water, mountain, rock,
   vegetation, pine, tree, architecture, figure, boat, bird, flower or ground.
3. Explicitly represent functional blank regions as sky, fog, mist, water,
   or blank.
4. A polygon must contain at least three vertices.
5. Keep all coordinates inside [0,1].
6. Lower depth_order values are rendered first.
7. Preserve visual hierarchy and the selected Three-Distances perspective.
8. Avoid excessive overlap between non-blank foreground objects.
9. Cover the canvas intentionally; uncovered pixels are interpreted as blank.
10. Return only the requested JSON object.
""".strip()

        user = f"""
PAINTING BRIEF:
{brief.model_dump_json(indent=2)}

COMPOSITION KNOWLEDGE:
{format_documents(docs)}

CORRECTION FEEDBACK:
{correction_feedback or "This is the first layout attempt."}

Generate a semantic polygon layout satisfying the brief.
""".strip()

        return self.llm.generate(
            system,
            user,
            LayoutPlan,
            self.temperature,
        )


class StyleAgent:
    def __init__(
        self,
        llm: StructuredLLM,
        retriever: KnowledgeRetriever,
        temperature: float = 0.7,
        top_k: int = 4,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.temperature = temperature
        self.top_k = top_k

    def run(
        self,
        region: Region,
        brief: PaintingBrief,
    ) -> StyleResult:
        query = (
            f"{region.label}; {region.description}; "
            f"{brief.mood}; Chinese ink brushwork"
        )
        docs = self.retriever.query(
            query,
            category="brushwork",
            top_k=self.top_k,
        )

        system = """
You are the Style Agent for Traditional Chinese Painting.

For one semantic region, create a concise English diffusion prompt that
specifies:
- suitable Cun-fa or brushstroke method;
- dry/wet brush behavior;
- ink density;
- edge quality;
- paper and wash behavior;
- regional atmosphere.

Do not assign Axe-cut Cun to soft water or fog. Do not densely texture
functional blank regions. Return only the requested JSON object.
""".strip()

        user = f"""
PAINTING BRIEF:
{brief.model_dump_json(indent=2)}

REGION:
{region.model_dump_json(indent=2)}

RETRIEVED BRUSHWORK KNOWLEDGE:
{format_documents(docs)}

Create a region-specific style prompt.
""".strip()

        result = self.llm.generate(
            system,
            user,
            StyleResult,
            self.temperature,
        )

        if result.region_id != region.id:
            result = result.model_copy(update={"region_id": region.id})

        return result