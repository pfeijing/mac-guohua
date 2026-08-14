from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)


class PaintingDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        resolution: int,
    ) -> None:
        self.df = pd.read_csv(manifest)
        if "caption" not in self.df.columns:
            raise ValueError("Manifest must contain a caption column.")

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")

        return {
            "pixel_values": self.transform(image),
            "caption": str(row["caption"]),
        }


def encode_prompts(
    captions: list[str],
    tokenizers: list,
    text_encoders: list,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for tokenizer, text_encoder in zip(
        tokenizers,
        text_encoders,
    ):
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(device)

        output = text_encoder(
            input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        prompt_embeds = output.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            pooled_prompt_embeds = output.text_embeds
        else:
            pooled_prompt_embeds = output[0]

    prompt_embeds = torch.cat(
        prompt_embeds_list,
        dim=-1,
    )

    assert pooled_prompt_embeds is not None
    return prompt_embeds, pooled_prompt_embeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--base-model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--vae-model",
        default="madebyollin/sdxl-vae-fp16-fix",
    )
    parser.add_argument("--output", default="outputs/lora")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", default="fp16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
    )

    weight_dtype = (
        torch.float16
        if args.mixed_precision == "fp16"
        else torch.bfloat16
        if args.mixed_precision == "bf16"
        else torch.float32
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model,
        subfolder="scheduler",
    )

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.base_model,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.base_model,
        subfolder="tokenizer_2",
        use_fast=False,
    )

    text_encoder_one = CLIPTextModel.from_pretrained(
        args.base_model,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.base_model,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
    )

    vae = AutoencoderKL.from_pretrained(
        args.vae_model,
        torch_dtype=weight_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model,
        subfolder="unet",
    )

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ],
    )
    unet.add_adapter(lora_config)

    for parameter in unet.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    trainable_parameters = [
        p for p in unet.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    dataset = PaintingDataset(
        args.manifest,
        args.resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    unet, optimizer, dataloader = accelerator.prepare(
        unet,
        optimizer,
        dataloader,
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(
        accelerator.device,
        dtype=weight_dtype,
    )
    text_encoder_two.to(
        accelerator.device,
        dtype=weight_dtype,
    )

    text_encoder_one.eval()
    text_encoder_two.eval()
    vae.eval()
    unet.train()

    progress = tqdm(
        range(args.max_steps),
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0

    while global_step < args.max_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                )

                with torch.no_grad():
                    latents = vae.encode(
                        pixel_values
                    ).latent_dist.sample()
                    latents = (
                        latents * vae.config.scaling_factor
                    )

                    prompt_embeds, pooled_embeds = encode_prompts(
                        batch["caption"],
                        [tokenizer_one, tokenizer_two],
                        [text_encoder_one, text_encoder_two],
                        accelerator.device,
                    )
                    prompt_embeds = prompt_embeds.to(
                        dtype=weight_dtype
                    )
                    pooled_embeds = pooled_embeds.to(
                        dtype=weight_dtype
                    )

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(
                    latents,
                    noise,
                    timesteps,
                )

                time_ids = torch.tensor(
                    [
                        args.resolution,
                        args.resolution,
                        0,
                        0,
                        args.resolution,
                        args.resolution,
                    ],
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                time_ids = time_ids.unsqueeze(0).repeat(
                    latents.shape[0],
                    1,
                )

                model_prediction = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds,
                        "time_ids": time_ids,
                    },
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif (
                    noise_scheduler.config.prediction_type
                    == "v_prediction"
                ):
                    target = noise_scheduler.get_velocity(
                        latents,
                        noise,
                        timesteps,
                    )
                else:
                    raise ValueError(
                        "Unsupported prediction type: "
                        f"{noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(
                    model_prediction.float(),
                    target.float(),
                    reduction="mean",
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_parameters,
                        1.0,
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(
                    loss=float(loss.detach().item())
                )

            if global_step >= args.max_steps:
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)

        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_state_dict = get_peft_model_state_dict(
            unwrapped_unet
        )
        lora_state_dict = convert_state_dict_to_diffusers(
            lora_state_dict
        )

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=output,
            unet_lora_layers=lora_state_dict,
            safe_serialization=True,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()