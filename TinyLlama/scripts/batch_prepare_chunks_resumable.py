import os
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count

# === 参数配置 ===
SCRIPT_PATH = "/yinyongjing/SMDM/TinyLlama/scripts/prepare_single_chunk_resumable.py"
CHUNK_DIR = "/ssdwork/yinyongjing/SlimPajama-62B/train"  # 可替换为 chunk0~5 的任意目录
TOKENIZER_PATH = "/yinyongjing/SMDM/data/llama"
DESTINATION_DIR = "/ssdwork/yinyongjing/slimpajama62b_prepared/resumable"
SPLIT = "train"
MAX_LINES = 6000000           # 每次最多处理的行数
NUM_PROCESSES = min(cpu_count(), 6)  # 控制并发数，避免内存爆

def process_file(file_path: Path):
    progress_dir = Path(DESTINATION_DIR) / "progress"
    done_file = progress_dir / f"{file_path.name}.done"
    if done_file.exists():
        return f"[Skip ✅] {file_path.name} already done."

    cmd = [
        "python", SCRIPT_PATH,
        "--source_file", str(file_path),
        "--tokenizer_path", TOKENIZER_PATH,
        "--destination_dir", DESTINATION_DIR,
        "--split", SPLIT,
        "--max_lines", str(MAX_LINES),
    ]

    try:
        subprocess.run(cmd, check=True)
        return f"[Done ✅] {file_path.name}"
    except subprocess.CalledProcessError:
        return f"[Fail ❌] {file_path.name}"


def main():
    progress_path = Path(DESTINATION_DIR) / "progress"
    progress_path.mkdir(parents=True, exist_ok=True)

    chunk_dir = Path(CHUNK_DIR)
    files = sorted(chunk_dir.rglob("*.jsonl.zst"))
    print(f"[Info] Found {len(files)} chunk files.")

    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(process_file, files)

    print("\n[Summary]")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
