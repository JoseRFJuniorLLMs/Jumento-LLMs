"""
Downloads and tokenizes the TinyStories dataset using the Llama 3.1 Tokenizer.

- Downloads the dataset from HuggingFace.
- Tokenizes the dataset using the Llama 3.1 Tokenizer with tiktoken.
- Outputs the tokenized data to a newly created 'tinystories/' directory.
- Prints the number of shards and the sizes of tokenized files.

The .bin files are raw byte streams of uint32 numbers indicating token IDs.

File sizes:
- Train: ~3.4G
- Validation: ~72M
"""

import os
import glob
import json
import random
import requests
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tokenizer import Tokenizer

def download_file(url: str, fname: str, chunk_size=1024):
    """Download a file from a URL and save it to a local file."""
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(fname, "wb") as file, tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        raise

def write_datafile(filename, toks):
    """Save token data as a .bin file."""
    assert len(toks) < 2**31, "Token count too large"  # ~2.1B tokens
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240801  # Magic number
    header[1] = 7  # Version
    header[2] = len(toks)  # Number of tokens
    toks_np = np.array(toks, dtype=np.uint32)
    print(f"Writing {len(toks):,} tokens to {filename}")
    try:
        with open(filename, "wb") as f:
            f.write(header.tobytes())
            f.write(toks_np.tobytes())
    except IOError as e:
        print(f"Error writing file: {e}")
        raise

def download_data(url: str, data_cache_dir: str):
    """Download and unpack the TinyStories dataset."""
    os.makedirs(data_cache_dir, exist_ok=True)
    data_filename = os.path.join(data_cache_dir, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {url} to {data_filename}...")
        download_file(url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")
    
    data_dir = os.path.join(data_cache_dir, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        try:
            os.system(f"tar -xzf {data_filename} -C {data_dir}")
        except Exception as e:
            print(f"Error unpacking file: {e}")
            raise
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print("Download and unpack done.")
    print(f"Number of shards: {len(shard_filenames)}")

def process_shard(shard_index, shard_filename, encode):
    """Process and tokenize a single shard."""
    try:
        with open(shard_filename, "r") as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading shard {shard_filename}: {e}")
        return []

    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    for example in data:
        text = example.get("story", "").strip()
        tokens = encode(text)
        all_tokens.extend(tokens)
    return all_tokens

def tokenize_data(data_cache_dir: str, encode):
    """Tokenize the TinyStories dataset and save to .bin files."""
    data_dir = os.path.join(data_cache_dir, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]

    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:
        print(f"Tokenizing {split_name} split...")
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_shard, shard_index, shard_filename, encode)
                       for shard_index, shard_filename in enumerate(split_shards)]
            for future in as_completed(futures):
                all_tokens.extend(future.result())

        split_filename = os.path.join(data_cache_dir, f"TinyStories_{split_name}.bin")
        write_datafile(split_filename, all_tokens)

def main():
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinystories")
    tokenizer = Tokenizer("llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model")
    encode = lambda x: tokenizer.encode(x, bos=True, eos=False)

    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    
    download_data(data_url, DATA_CACHE_DIR)
    tokenize_data(DATA_CACHE_DIR, encode)

if __name__ == "__main__":
    main()
