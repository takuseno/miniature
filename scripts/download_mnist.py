import os
import tempfile
import gzip
from urllib import request

DATASET_DIR = "datasets"


def download_and_uncompress(url, name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_data_path = os.path.join(tmp_dir, name + ".gz")

        print(f"Downloading {url}...")
        request.urlretrieve(url, tmp_data_path)
        with gzip.open(tmp_data_path, "rb") as fp:
            data = fp.read()

        uncompressed_path = os.path.join(DATASET_DIR, name)
        print(f"Writing uncompressed data to {uncompressed_path}")
        os.makedirs(DATASET_DIR, exist_ok=True)
        with open(uncompressed_path, "wb") as f:
            f.write(data)

def main():
    download_and_uncompress("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "mnist-train-images")
    download_and_uncompress("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "mnist-train-labels")
    download_and_uncompress("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "mnist-test-images")
    download_and_uncompress("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "mnist-test-labels")


if __name__ == "__main__":
    main()
