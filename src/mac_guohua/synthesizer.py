from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image
from tqdm import tqdm

from .layout import RasterizedLayout
from .schemas import LayoutPlan, PaintingBrief


@dataclass
class PromptCondition:
    prompt_embeds: torch.Tensor
    pooled_embeds: torch.Tensor


class GuohuaSynthesizer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.device.type == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        controlnet = ControlNetModel.from_pretrained(
            config["controlnet_model"],
            torch_dtype=self.dtype,
        )

        vae_name = config.get("vae_model")
        vae = None
        if vae_name:
            vae = AutoencoderKL.from_pretrained(
                vae_name,
                torch_dtype=self.dtype,
            )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            config["base_model"],
            controlnet=controlnet,
            vae=vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=False)

        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

        lora_path = config.get("lora_path", "")
        if lora_path:
            self.pipe.load_lora_weights(lora_path, adapter_name="ink")
            self.pipe.set_adapters(
                ["ink"],
                adapter_weights=[float(config.get("lora_scale", 0.8))],
            )

    def _global_prompt(
        self,
        brief: PaintingBrief,
        layout: LayoutPlan,
    ) -> str:
        regional = []
        for region in layout.regions:
            if region.style_prompt:
                regional.append(
                    f"{region.label} region: {region.style_prompt}"
                )

        return (
            "authentic traditional Chinese ink painting, Guohua, Xieyi, "
            "rice paper texture, controlled ink diffusion, intentional "
            f"negative space. Subject: {brief.subject}. "
            f"Mood: {brief.mood}. Season: {brief.season}. "
            f"Perspective: {brief.perspective}. "
            f"Composition: {brief.composition_summary}. "
            + " ".join(regional)
        )

    @torch.inference_mode()
    def generate(
        self,
        brief: PaintingBrief,
        layout: LayoutPlan,
        raster: RasterizedLayout,
        seed: int,
    ) -> Image.Image:
        if self.config.get("regional_prompting", True):
            return self._generate_regional(
                brief,
                layout,
                raster,
                seed,
            )
        return self._generate_standard(
            brief,
            layout,
            raster,
            seed,
        )

    @torch.inference_mode()
    def _generate_standard(
        self,
        brief: PaintingBrief,
        layout: LayoutPlan,
        raster: RasterizedLayout,
        seed: int,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=self._global_prompt(brief, layout),
            negative_prompt=self.config["negative_prompt"],
            image=raster.semantic_image,
            width=int(self.config["width"]),
            height=int(self.config["height"]),
            num_inference_steps=int(self.config["steps"]),
            guidance_scale=float(self.config["guidance_scale"]),
            controlnet_conditioning_scale=float(
                self.config["controlnet_scale"]
            ),
            generator=generator,
        )

        return result.images[0]

    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str,
    ) -> PromptCondition:
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
        )

        combined_prompt = torch.cat(
            [negative_prompt_embeds, prompt_embeds],
            dim=0,
        )
        combined_pooled = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds],
            dim=0,
        )

        return PromptCondition(
            prompt_embeds=combined_prompt,
            pooled_embeds=combined_pooled,
        )

    def _predict_unet(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        condition: PromptCondition,
        time_ids: torch.Tensor,
        down_residuals: tuple[torch.Tensor, ...],
        mid_residual: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        noise_pred = self.pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=condition.prompt_embeds,
            added_cond_kwargs={
                "text_embeds": condition.pooled_embeds,
                "time_ids": time_ids,
            },
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )[0]

        noise_uncond, noise_text = noise_pred.chunk(2)

        return noise_uncond + guidance_scale * (
            noise_text - noise_uncond
        )

    def _prepare_latent_masks(
        self,
        layout: LayoutPlan,
        raster: RasterizedLayout,
        latent_height: int,
        latent_width: int,
    ) -> list[torch.Tensor]:
        masks: list[torch.Tensor] = []

        for region in layout.regions:
            image = raster.region_masks[region.id]
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array)[None, None]
            tensor = F.interpolate(
                tensor,
                size=(latent_height, latent_width),
                mode="nearest",
            )
            masks.append(
                tensor.to(
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        return masks

    @torch.inference_mode()
    def _generate_regional(
        self,
        brief: PaintingBrief,
        layout: LayoutPlan,
        raster: RasterizedLayout,
        seed: int,
    ) -> Image.Image:
        width = int(self.config["width"])
        height = int(self.config["height"])
        steps = int(self.config["steps"])
        guidance_scale = float(self.config["guidance_scale"])
        control_scale = float(self.config["controlnet_scale"])
        regional_strength = float(
            self.config.get("regional_strength", 0.75)
        )
        negative_prompt = self.config["negative_prompt"]

        generator = torch.Generator(device=self.device).manual_seed(seed)

        global_condition = self._encode_prompt(
            self._global_prompt(brief, layout),
            negative_prompt,
        )

        region_conditions: list[PromptCondition] = []
        for region in layout.regions:
            prompt = (
                f"traditional Chinese ink painting, {region.label}, "
                f"{region.description}, {region.style_prompt}, "
                f"overall mood {brief.mood}"
            )
            region_conditions.append(
                self._encode_prompt(prompt, negative_prompt)
            )

        control_image = self.pipe.prepare_image(
            image=raster.semantic_image,
            width=width,
            height=height,
            batch_size=1,
            num_images_per_prompt=1,
            device=self.device,
            dtype=next(self.pipe.controlnet.parameters()).dtype,
            do_classifier_free_guidance=True,
            guess_mode=False,
        )

        self.pipe.scheduler.set_timesteps(
            steps,
            device=self.device,
        )
        timesteps = self.pipe.scheduler.timesteps

        latents = self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=self.pipe.unet.config.in_channels,
            height=height,
            width=width,
            dtype=global_condition.prompt_embeds.dtype,
            device=self.device,
            generator=generator,
            latents=None,
        )

        projection_dim = self.pipe.text_encoder_2.config.projection_dim
        add_time_ids = self.pipe._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            dtype=global_condition.prompt_embeds.dtype,
            text_encoder_projection_dim=projection_dim,
        )
        time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        time_ids = time_ids.to(self.device)

        latent_masks = self._prepare_latent_masks(
            layout,
            raster,
            latents.shape[-2],
            latents.shape[-1],
        )

        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(
            generator,
            eta=0.0,
        )

        for timestep in tqdm(timesteps, desc="Diffusion"):
            latent_input = torch.cat([latents, latents], dim=0)
            latent_input = self.pipe.scheduler.scale_model_input(
                latent_input,
                timestep,
            )

            down_residuals, mid_residual = self.pipe.controlnet(
                latent_input,
                timestep,
                encoder_hidden_states=global_condition.prompt_embeds,
                controlnet_cond=control_image,
                conditioning_scale=control_scale,
                guess_mode=False,
                added_cond_kwargs={
                    "text_embeds": global_condition.pooled_embeds,
                    "time_ids": time_ids,
                },
                return_dict=False,
            )

            global_noise = self._predict_unet(
                latent_input,
                timestep,
                global_condition,
                time_ids,
                down_residuals,
                mid_residual,
                guidance_scale,
            )

            regional_sum = torch.zeros_like(global_noise)
            mask_sum = torch.zeros_like(global_noise[:, :1])

            for condition, mask in zip(
                region_conditions,
                latent_masks,
            ):
                if float(mask.max()) <= 0:
                    continue

                region_noise = self._predict_unet(
                    latent_input,
                    timestep,
                    condition,
                    time_ids,
                    down_residuals,
                    mid_residual,
                    guidance_scale,
                )
                regional_sum += region_noise * mask
                mask_sum += mask

            regional_noise = regional_sum / mask_sum.clamp(min=1e-6)
            uncovered = mask_sum <= 1e-6
            regional_noise = torch.where(
                uncovered.expand_as(regional_noise),
                global_noise,
                regional_noise,
            )

            noise_pred = (
                (1.0 - regional_strength) * global_noise
                + regional_strength * regional_noise
            )

            latents = self.pipe.scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=False,
                **extra_step_kwargs,
            )[0]

        latents = latents / self.pipe.vae.config.scaling_factor
        latents = latents.to(dtype=self.pipe.vae.dtype)

        image = self.pipe.vae.decode(
            latents,
            return_dict=False,
        )[0]

        images = self.pipe.image_processor.postprocess(
            image,
            output_type="pil",
        )

        return images[0]