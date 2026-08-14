from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="Salesforce/blip2-opt-2.7b",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Describe this traditional Chinese painting, including "
            "subject, composition, negative space, brushwork and mood:"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    df = pd.read_csv(args.manifest)
    captions = []

    for path in tqdm(df["image_path"], desc="Captioning"):
        image = Image.open(path).convert("RGB")
        inputs = processor(
            images=image,
            text=args.prompt,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
            )

        caption = processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        captions.append(caption)

    df["caption"] = captions
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()