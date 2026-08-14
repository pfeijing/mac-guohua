from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def remove_duplicates(
    dataframe: pd.DataFrame,
    phash_threshold: int,
) -> pd.DataFrame:
    accepted_rows = []
    exact_hashes: set[str] = set()
    perceptual_hashes: list[imagehash.ImageHash] = []

    for _, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc="Deduplicating",
    ):
        path = Path(row["image_path"])
        if not path.exists() or not validate_image(path):
            continue

        file_hash = sha256_file(path)
        if file_hash in exact_hashes:
            continue

        with Image.open(path) as image:
            phash = imagehash.phash(image.convert("RGB"))

        near_duplicate = any(
            phash - old_hash <= phash_threshold
            for old_hash in perceptual_hashes
        )
        if near_duplicate:
            continue

        row = row.copy()
        row["sha256"] = file_hash
        row["phash"] = str(phash)

        exact_hashes.add(file_hash)
        perceptual_hashes.append(phash)
        accepted_rows.append(row)

    return pd.DataFrame(accepted_rows)


def stratification_key(df: pd.DataFrame) -> pd.Series:
    source = df["source"].fillna("unknown").astype(str)
    genre = df["genre"].fillna("unknown").astype(str)
    key = source + "__" + genre

    counts = key.value_counts()
    rare = set(counts[counts < 3].index)
    return key.map(lambda x: "rare" if x in rare else x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/cp5k")
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-images",
        action="store_true",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    required = {
        "image_path",
        "source",
        "source_url",
        "license",
        "collection_id",
        "genre",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = remove_duplicates(df, args.phash_threshold)
    df = df.reset_index(drop=True)

    stratify = stratification_key(df)

    train, temp = train_test_split(
        df,
        test_size=0.20,
        random_state=args.seed,
        stratify=stratify,
    )

    temp_stratify = stratification_key(temp)
    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_stratify,
    )

    split_map = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for split_name, split_df in split_map.items():
        split_df = split_df.copy()
        split_df["split"] = split_name

        if args.copy_images:
            split_dir = output / "images" / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            new_paths = []
            for index, row in split_df.iterrows():
                source = Path(row["image_path"])
                destination = (
                    split_dir
                    / f"{row['sha256'][:16]}{source.suffix.lower()}"
                )
                shutil.copy2(source, destination)
                new_paths.append(str(destination))

            split_df["image_path"] = new_paths

        split_df.to_csv(
            output / f"{split_name}.csv",
            index=False,
        )

    full = pd.concat(
        [
            split_map["train"].assign(split="train"),
            split_map["validation"].assign(split="validation"),
            split_map["test"].assign(split="test"),
        ],
        ignore_index=True,
    )
    full.to_csv(output / "manifest.csv", index=False)

    stats = {
        "total": len(full),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "licenses": full["license"].value_counts().to_dict(),
        "sources": full["source"].value_counts().to_dict(),
        "genres": full["genre"].value_counts().to_dict(),
    }

    with (output / "statistics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()