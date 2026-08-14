from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import pdist
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    )


def load_for_fid(path: Path, size: int = 299) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(
        (size, size),
        Image.Resampling.BICUBIC,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def negative_space_ratio(
    path: Path,
    threshold: float = 0.90,
) -> float:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return float((array > threshold).mean())


def compute_fid(
    real_paths: list[Path],
    fake_paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> float:
    metric = FrechetInceptionDistance(
        feature=2048,
        normalize=True,
    ).to(device)

    for start in tqdm(
        range(0, len(real_paths), batch_size),
        desc="FID real",
    ):
        batch = torch.stack(
            [
                load_for_fid(p)
                for p in real_paths[start : start + batch_size]
            ]
        ).to(device)
        metric.update(batch, real=True)

    for start in tqdm(
        range(0, len(fake_paths), batch_size),
        desc="FID fake",
    ):
        batch = torch.stack(
            [
                load_for_fid(p)
                for p in fake_paths[start : start + batch_size]
            ]
        ).to(device)
        metric.update(batch, real=False)

    return float(metric.compute().item())


class CLIPEncoder:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-L-14",
                pretrained="openai",
            )
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model = self.model.to(device).eval()

    @torch.inference_mode()
    def encode_images(
        self,
        paths: list[Path],
        batch_size: int,
    ) -> np.ndarray:
        all_embeddings = []

        for start in tqdm(
            range(0, len(paths), batch_size),
            desc="CLIP images",
        ):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack(
                [
                    self.preprocess(
                        Image.open(p).convert("RGB")
                    )
                    for p in batch_paths
                ]
            ).to(self.device)

            embeddings = self.model.encode_image(images)
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(
                embeddings.cpu().float().numpy()
            )

        return np.concatenate(all_embeddings, axis=0)

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int,
    ) -> np.ndarray:
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)
            embeddings = self.model.encode_text(tokens)
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(
                embeddings.cpu().float().numpy()
            )

        return np.concatenate(all_embeddings, axis=0)


def rbf_mmd(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float | None = None,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if sigma is None:
        combined = np.concatenate([x, y], axis=0)
        distances = pdist(combined, metric="sqeuclidean")
        positive = distances[distances > 0]
        median = np.median(positive) if len(positive) else 1.0
        sigma = float(np.sqrt(median / 2.0))
        sigma = max(sigma, 1e-6)

    gamma = 1.0 / (2.0 * sigma**2)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = np.sum(a * a, axis=1, keepdims=True)
        b2 = np.sum(b * b, axis=1, keepdims=True).T
        distance = np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)
        return np.exp(-gamma * distance)

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)

    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    nx = len(x)
    ny = len(y)

    xx = k_xx.sum() / max(nx * (nx - 1), 1)
    yy = k_yy.sum() / max(ny * (ny - 1), 1)
    xy = k_xy.mean()

    return float(max(0.0, xx + yy - 2.0 * xy))


def load_prompt_mapping(
    manifest: str | None,
    fake_paths: list[Path],
) -> list[str] | None:
    if not manifest:
        return None

    df = pd.read_csv(manifest)

    if "filename" not in df.columns or "prompt" not in df.columns:
        raise ValueError(
            "Prompt manifest must contain filename and prompt columns."
        )

    mapping = dict(zip(df["filename"], df["prompt"]))
    return [str(mapping[p.name]) for p in fake_paths]


def run_optional_predictor(
    model_path: str | None,
    inputs: torch.Tensor,
    device: torch.device,
) -> float | None:
    if not model_path:
        return None

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    with torch.inference_mode():
        values = model(inputs.to(device))

    return float(values.float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--prompt-manifest")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nsr-threshold", type=float, default=0.90)
    parser.add_argument("--aesthetic-model")
    parser.add_argument("--output", default="evaluation.json")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    real_paths = collect_images(args.real_dir)
    fake_paths = collect_images(args.fake_dir)

    if len(real_paths) < 2 or len(fake_paths) < 2:
        raise ValueError("At least two real and fake images are required.")

    fid = compute_fid(
        real_paths,
        fake_paths,
        device,
        args.batch_size,
    )

    clip = CLIPEncoder(device)
    real_embeddings = clip.encode_images(
        real_paths,
        args.batch_size,
    )
    fake_embeddings = clip.encode_images(
        fake_paths,
        args.batch_size,
    )

    cmmd = rbf_mmd(real_embeddings, fake_embeddings)

    prompts = load_prompt_mapping(
        args.prompt_manifest,
        fake_paths,
    )
    clip_score = None

    if prompts is not None:
        text_embeddings = clip.encode_texts(
            prompts,
            args.batch_size,
        )
        count = min(
            len(text_embeddings),
            len(fake_embeddings),
        )
        similarities = np.sum(
            text_embeddings[:count] * fake_embeddings[:count],
            axis=1,
        )
        clip_score = float(similarities.mean() * 100.0)

    real_nsr = np.mean(
        [
            negative_space_ratio(p, args.nsr_threshold)
            for p in tqdm(real_paths, desc="NSR real")
        ]
    )
    fake_nsr = np.mean(
        [
            negative_space_ratio(p, args.nsr_threshold)
            for p in tqdm(fake_paths, desc="NSR fake")
        ]
    )
    nsr_error = float(abs(fake_nsr - real_nsr) * 100.0)

    aesthetic_score = run_optional_predictor(
        args.aesthetic_model,
        torch.from_numpy(fake_embeddings).float(),
        device,
    )

    result = {
        "num_real": len(real_paths),
        "num_fake": len(fake_paths),
        "fid": fid,
        "clip_score": clip_score,
        "cmmd_rbf_clip": cmmd,
        "real_mean_nsr": float(real_nsr),
        "fake_mean_nsr": float(fake_nsr),
        "nsr_error_percent": nsr_error,
        "aesthetic_score": aesthetic_score,
        "nsr_threshold": args.nsr_threshold,
    }

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()