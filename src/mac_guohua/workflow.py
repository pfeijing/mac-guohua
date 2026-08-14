from __future__ import annotations

import json
from pathlib import Path

from .agents import (
    CompositionAgent,
    DirectorAgent,
    StructuredLLM,
    StyleAgent,
)
from .layout import (
    RasterizedLayout,
    image_negative_space_ratio,
    layout_score,
    rasterize_layout,
)
from .retrieval import KnowledgeRetriever
from .schemas import GenerationRecord, LayoutPlan
from .synthesizer import GuohuaSynthesizer


class MACGuohuaWorkflow:
    def __init__(self, config: dict) -> None:
        self.config = config

        retrieval_cfg = config["retrieval"]
        self.retriever = KnowledgeRetriever(
            knowledge_dir=retrieval_cfg["knowledge_dir"],
            index_dir=retrieval_cfg["index_dir"],
            embedding_model=retrieval_cfg["embedding_model"],
        )
        self.retriever.load()

        openai_cfg = config["openai"]
        llm = StructuredLLM(
            model=openai_cfg["model"],
            max_retries=int(openai_cfg.get("max_retries", 3)),
        )

        top_k = int(retrieval_cfg.get("top_k", 5))

        self.director = DirectorAgent(
            llm,
            self.retriever,
            temperature=float(
                openai_cfg["director_temperature"]
            ),
            top_k=top_k,
        )
        self.composition = CompositionAgent(
            llm,
            self.retriever,
            temperature=float(
                openai_cfg["composition_temperature"]
            ),
            top_k=top_k,
        )
        self.style = StyleAgent(
            llm,
            self.retriever,
            temperature=float(
                openai_cfg["style_temperature"]
            ),
            top_k=top_k,
        )

        self.synthesizer: GuohuaSynthesizer | None = None

    def _rasterize(
        self,
        layout: LayoutPlan,
    ) -> RasterizedLayout:
        layout_cfg = self.config["layout"]

        return rasterize_layout(
            layout=layout,
            width=int(layout_cfg["width"]),
            height=int(layout_cfg["height"]),
            palette=self.config["palette"],
            blank_labels={
                str(x).lower()
                for x in layout_cfg["blank_labels"]
            },
        )

    def generate(
        self,
        prompt: str,
        output_dir: str | Path,
        seed: int | None = None,
    ) -> GenerationRecord:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if seed is None:
            seed = int(self.config.get("seed", 42))

        brief = self.director.run(prompt)

        layout_cfg = self.config["layout"]
        target = float(layout_cfg["liubai_threshold"])
        max_iterations = int(layout_cfg["max_iterations"])
        overlap_lambda = float(layout_cfg["overlap_lambda"])

        # 论文实验固定目标为 0.30。
        brief = brief.model_copy(
            update={"negative_space_target": target}
        )

        best_layout: LayoutPlan | None = None
        best_raster: RasterizedLayout | None = None
        best_score = float("-inf")
        feedback = ""
        used_iterations = 0

        for iteration in range(1, max_iterations + 1):
            used_iterations = iteration

            candidate = self.composition.run(
                brief,
                correction_feedback=feedback,
            )
            raster = self._rasterize(candidate)

            score = layout_score(
                blank_ratio=raster.blank_ratio,
                target=target,
                overlap_penalty=raster.overlap_penalty,
                overlap_lambda=overlap_lambda,
            )

            if score > best_score:
                best_score = score
                best_layout = candidate
                best_raster = raster

            if raster.blank_ratio >= target:
                break

            deficit = target - raster.blank_ratio
            feedback = f"""
The previous functional blank-space ratio was
{raster.blank_ratio:.4f}, below the required target {target:.4f}.

Increase blank sky, mist, fog, water or untouched-paper regions by
approximately {deficit:.4f} of the canvas. Reduce or compress secondary
objects while preserving the main subject. Avoid increasing invalid
non-blank polygon overlap. Generate a corrected full layout.
""".strip()

        if best_layout is None or best_raster is None:
            raise RuntimeError("Composition generation produced no layout.")

        styled_regions = []
        style_negative_prompts: list[str] = []

        for region in best_layout.regions:
            style_result = self.style.run(region, brief)
            styled_regions.append(
                region.model_copy(
                    update={"style_prompt": style_result.prompt}
                )
            )
            if style_result.negative_prompt:
                style_negative_prompts.append(
                    style_result.negative_prompt
                )

        best_layout = best_layout.model_copy(
            update={"regions": styled_regions}
        )
        best_raster = self._rasterize(best_layout)

        if style_negative_prompts:
            generation_cfg = self.config["generation"]
            generation_cfg["negative_prompt"] = (
                generation_cfg["negative_prompt"]
                + ", "
                + ", ".join(style_negative_prompts)
            )

        with (output_dir / "brief.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                brief.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        with (output_dir / "layout.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                best_layout.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        best_raster.semantic_image.save(
            output_dir / "segmentation.png"
        )

        mask_dir = output_dir / "region_masks"
        mask_dir.mkdir(exist_ok=True)
        for region_id, mask in best_raster.region_masks.items():
            mask.save(mask_dir / f"{region_id}.png")

        if self.synthesizer is None:
            self.synthesizer = GuohuaSynthesizer(
                self.config["generation"]
            )

        image = self.synthesizer.generate(
            brief=brief,
            layout=best_layout,
            raster=best_raster,
            seed=seed,
        )

        image_path = output_dir / "generated.png"
        image.save(image_path)

        record = GenerationRecord(
            user_prompt=prompt,
            brief=brief,
            layout=best_layout,
            blank_ratio=best_raster.blank_ratio,
            overlap_penalty=best_raster.overlap_penalty,
            iterations=used_iterations,
            image_path=str(image_path),
        )

        metadata = record.model_dump()
        metadata["output_image_nsr"] = image_negative_space_ratio(
            image,
            threshold=0.90,
        )
        metadata["seed"] = seed

        with (output_dir / "metadata.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return record