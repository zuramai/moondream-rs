import struct
import gzip
from typing import BinaryIO, Tuple, Iterator, Union
from pathlib import Path
import requests
import os
from tqdm import tqdm
import sys

MOON_MAGIC = b"MOON"
MOON_VERSION = 1


class MoonReader:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def _get_file_handle(self) -> Union[BinaryIO, gzip.GzipFile]:
        """Returns appropriate file handle based on extension"""
        if self.input_path.endswith(".gz"):
            return gzip.open(self.input_path, "rb")
        return open(self.input_path, "rb")

    def _validate_header(self, f: Union[BinaryIO, gzip.GzipFile]) -> None:
        """Validate magic bytes and version"""
        magic = f.read(4)
        if magic != MOON_MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")

        version = struct.unpack("!B", f.read(1))[0]
        if version != MOON_VERSION:
            raise ValueError(f"Unsupported version: {version}")

    def read_files(self) -> Iterator[Tuple[str, bytes]]:
        """Read and yield (filename, content) pairs from the archive"""
        with self._get_file_handle() as f:
            self._validate_header(f)

            while True:
                # Try to read filename length
                filename_len_bytes = f.read(4)
                if not filename_len_bytes:
                    break  # End of file

                filename_len = struct.unpack("!I", filename_len_bytes)[0]

                # Read filename
                filename = f.read(filename_len).decode("utf-8")

                # Read content length and content
                content_len = struct.unpack("!Q", f.read(8))[0]
                content = f.read(content_len)

                yield filename, content


def unpack(input_path: str) -> Iterator[Tuple[str, bytes]]:
    """Unpack a .gz file"""
    # Create a directory for the unpacked files
    output_dir = Path(input_path).parent 
    for filename, content in MoonReader(input_path).read_files():
        yield filename, content


def download_model(model_type: str):
    links = {
        "0.5b": "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz?download=true",
        "2b": "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz?download=true"
    }
    url = links[model_type]
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    output_path = f"./model/{model_type}.mf.gz"
    model_dir = Path("./model")
    if not model_dir.exists():
        os.makedirs(model_dir)
    output_path = model_dir / f"{model_type}.mf.gz"

    with open(output_path, "wb") as f, tqdm(
        desc=output_path,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)
    print(f"Model downloaded and saved to {output_path}")
    print(f"Unpacking model files from: {output_path}")
    return output_path.__str__()


def download():
    model_dir = Path("./model")
    if not model_dir.exists():
        raise FileNotFoundError("Model directory not found")
    
    args = sys.argv[1:]
    if not args:
        raise ValueError("No model type provided. Should be [0.5b, 2b]")
    
    model_type = args[0]

    mf_files = list(model_dir.glob("*.mf.gz"))
    if mf_files:
        print(f"Model found, unpacking model files from: {mf_files[0]}")
        result = unpack(mf_files[0].__str__())

    else:
        print(f"No model found, downloading model: {model_type}")
        output_path = download_model(model_type)
        result = unpack(output_path)

    for filename, content in result:
        print(f"Unpacked {filename}")
        with open(model_dir / filename, "wb") as f:
            f.write(content)

if __name__ == "__main__":
    download()