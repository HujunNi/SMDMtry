#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import sentencepiece as spm

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset

def prepare_wikitext2(
    data_dir: Path,
    tokenizer_model_path: Path,
    destination_path: Path,
    split: str = "train",
    chunk_size: int = 2049 * 1024
) -> None:
    data_file = data_dir / f"wiki.{split}.tokens"
    assert data_file.exists(), f"File not found: {data_file}"

    destination_path.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_model_path))
    print(f"Saving to: {destination_path}/{split}_slim_*.bin")

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_slim",
        chunk_size=chunk_size,
        sep_token=sp.bos_id(),  # Use BOS token if needed
        dtype="auto",
        vocab_size=sp.get_piece_size(),
    )

    with open(data_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing {split}"):
            text = line.strip()
            if not text:
                continue
            text_ids = sp.encode(text, out_type=int)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))

    if builder._idx > 0:
        builder._write_chunk()

    print(f"Done! Output saved to: {destination_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to wikitext2 .tokens files")
    parser.add_argument("--tokenizer_model_path", type=Path, required=True, help="Path to tokenizer.model")
    parser.add_argument("--destination_path", type=Path, required=True, help="Output directory for .bin files")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="Dataset split")
    args = parser.parse_args()

    prepare_wikitext2(
        data_dir=args.data_dir,
        tokenizer_model_path=args.tokenizer_model_path,
        destination_path=args.destination_path,
        split=args.split,
    )
