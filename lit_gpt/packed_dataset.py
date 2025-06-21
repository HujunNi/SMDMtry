# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py

import os
import random
import struct

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def code(dtype):
    for k, v in dtypes.items():
        if v == dtype:
            return k
    raise ValueError(f"Unsupported dtype: {dtype}")


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self,
        filenames,
        n_chunks,
        orig_block_size,
        seed=12345,
        shuffle=True,
        wrap=False,
        num_processes=1,
        process_rank=0,
        tokenizer=None,              # added
        latent_size=64,              # added
        use_latent_tokens=False,     # added
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._orig_block_size = orig_block_size  # added
        # total_block_size = original + latent
        self._total_block_size = orig_block_size + latent_size  # added
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

        # latent token support
        self.use_latent_tokens = use_latent_tokens  # added
        self.latent_token_tensor = None             # added
        if self.use_latent_tokens:                  # added
            ids = tokenizer.convert_tokens_to_ids([f"<LATENT_{i}>" for i in range(latent_size)])  # added
            self.latent_token_tensor = torch.tensor(ids, dtype=torch.long)  # added

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_files:num_shards]
        if not filenames:
            return iter([])

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            orig_block_size=self._orig_block_size,        # added
            total_block_size=self._total_block_size,      # added
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
            use_latent_tokens=self.use_latent_tokens,     # added
            latent_token_tensor=self.latent_token_tensor, # added
        )


class PackedDatasetBuilder:
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            self._dtype = np.uint16 if vocab_size < 65500 else np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.full(self._chunk_size, self._sep_token, dtype=self._dtype)
        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        fname = f"{self._prefix}_{self._counter:010d}.bin"
        path = os.path.join(self._outdir, fname)
        with open(path, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))
        self._filenames.append(path)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return list(self._filenames)

    def add_array(self, arr: np.ndarray):
        while self._idx + arr.shape[0] > self._chunk_size:
            part = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part] = arr[:part]
            self._write_chunk()
            arr = arr[part:]
        self._arr[self._idx : self._idx + arr.shape[0]] = arr
        self._idx += arr.shape[0]

    def write_reminder(self):
        self._write_chunk()


class PackedDatasetIterator:
    def __init__(
        self,
        filenames,
        n_chunks,
        orig_block_size,
        total_block_size,
        seed,
        shuffle,
        wrap,
        use_latent_tokens=False,     # added
        latent_token_tensor=None,    # added
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._orig_block_size = orig_block_size        # added
        self._total_block_size = total_block_size      # added
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self.use_latent_tokens = use_latent_tokens     # added
        self.latent_token_tensor = latent_token_tensor # added

        self._rng = np.random.default_rng(seed) if shuffle else None
        self._file_idx = 0
        self._dtype = None
        self._chunk_size = None
        self._n_blocks = None
        self._mmaps = []
        self._buffers = []
        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            assert f.read(len(HDR_MAGIC)) == HDR_MAGIC
            version, = struct.unpack("<Q", f.read(8))
            assert version == 1
            dtype_code, = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            chunk_size, = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mp in self._mmaps:
            mp._mmap.close()

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps.clear()
        self._buffers.clear()

        if self._file_idx + self._n_chunks > len(self._filenames):
            self._file_idx = 0

        for i in range(self._n_chunks):
            path = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(path)
                self._n_blocks = self._chunk_size // self._orig_block_size
            mp = np.memmap(path, mode="r", offset=HDR_SIZE, dtype=self._dtype, shape=(self._chunk_size,))
            self._mmaps.append(mp)
            self._buffers.append(mp)

        self._file_idx += self._n_chunks
        total_blocks = self._n_chunks * self._n_blocks
        indices = self._rng.permutation(total_blocks) if self._shuffle else np.arange(total_blocks)
        self._block_idxs = indices
        self._curr_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
        block_idx = int(self._block_idxs[self._curr_idx])
        chunk_id = block_idx // self._n_blocks
        elem_id = (block_idx % self._n_blocks) * self._orig_block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        buf = self._buffers[chunk_id]
        arr = np.frombuffer(buf, dtype=self._dtype, count=self._orig_block_size, offset=offset)
        self._curr_idx += 1

        block_tensor = torch.from_numpy(arr.astype(np.int64))
        if self.use_latent_tokens and self.latent_token_tensor is not None:  # added
            out = torch.cat([self.latent_token_tensor, block_tensor], dim=0)  # added
            assert out.size(0) == self._total_block_size, (               # added
                f"Expected length {self._total_block_size}, got {out.size(0)}"
            )                                                              # added
            return out                                                     # added
        return block_tensor

    def __del__(self):
        self._close_mmaps()


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._datasets = datasets
        self._seed = seed
        self._weights = weights or [1 / len(datasets)] * len(datasets)

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._iters = [iter(ds) for ds in datasets]
        self._rng = random.Random(seed)
        self._weights = weights

    def __next__(self):
        ds = self._rng.choices(self._iters, weights=self._weights, k=1)[0]
        return next(ds)
