import argparse
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import zstandard as zstd

# 支持运行时加载 lit_gpt
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from lit_gpt.packed_dataset import PackedDatasetBuilder
from lit_gpt import Tokenizer


def main(source_file: Path, tokenizer_path: Path, destination_dir: Path, split: str, max_lines: int = -1):
    destination_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    progress_dir = destination_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    progress_file = progress_dir / f"{source_file.name}.progress"
    done_file = progress_dir / f"{source_file.name}.done"
    hash_prefix = source_file.stem.replace(".", "_")

    if done_file.exists():
        print(f"[Skip] Already completed: {source_file.name}")
        return

    output_file = destination_dir / f"{split}_resume_{hash_prefix}.bin"
    builder = PackedDatasetBuilder(
        outdir=destination_dir,
        prefix=f"{split}_resume_{hash_prefix}",
        chunk_size=2049 * 1024,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    # Resume position
    resume_from = 0
    if progress_file.exists():
        with open(progress_file, "r") as pf:
            resume_from = int(pf.read().strip())
        print(f"[Resume] Resuming from line {resume_from}")

    with zstd.open(open(source_file, "rb"), "rt", encoding="utf-8") as f:
        for i, row in enumerate(f):
            if i < resume_from:
                continue
            if max_lines != -1 and i >= resume_from + max_lines:
                print(f"[Break] Reached max_lines={max_lines}, stopping early.")
                break

            try:
                record = json.loads(row)
                if record.get("meta", {}).get("redpajama_set_name") == "RedPajamaGithub":
                    continue
                text_ids = tokenizer.encode(record["text"])
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
            except Exception as e:
                print(f"[Error] Skipping row {i}: {e}")

            if i % 10000 == 0 and i > 0:
                with open(progress_file, "w") as pf:
                    pf.write(str(i))

    # 如果已经到文件末尾，写 .done
    if max_lines == -1 or i < resume_from + max_lines - 1:
        done_file.touch()
        print(f"[Done ✅] Finished full file: {source_file.name}")
    else:
        print(f"[Partial ⏸️] Stopped after {max_lines} lines: {source_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--destination_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_lines", type=int, default=-1, help="Maximum number of lines to process per run (-1 = no limit)")
    args = parser.parse_args()

    main(
        source_file=Path(args.source_file),
        tokenizer_path=Path(args.tokenizer_path),
        destination_dir=Path(args.destination_dir),
        split=args.split,
        max_lines=args.max_lines,
    )
