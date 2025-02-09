import os
import sys
from huggingface_hub import hf_hub_download


class FinewebCachedDataLoader:
    '''
        A class for downloading pre-tokenized Fineweb datasets from Hugging Face Hub.
        The cache saves about an hour of startup time compared to regenerating them using fineweb.py
    '''

    DATASETS = {
        "fineweb10B": {  # Each chunk is 100M tokens
            "repo_id": "kjj0/fineweb10B-gpt2",
            "val_file": "fineweb_val_%06d.bin",
            "train_file": "fineweb_train_%06d.bin"
        },
        "finewebedu10B": {  # Each chunk is 100M tokens
            "repo_id": "kjj0/finewebedu10B-gpt2",
            "val_file": "finewebedu_val_%06d.bin",
            "train_file": "finewebedu_train_%06d.bin"
        },
    }

    def __init__(self, dataset_name):
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}, should be in {self.DATASETS.keys()}")

        self.dataset_name = dataset_name
        self.repo_id = self.DATASETS[self.dataset_name]["repo_id"]
        self.local_dir = os.path.join(os.path.dirname(__file__), self.dataset_name)
        self.val_file = self.DATASETS[self.dataset_name]["val_file"]
        self.train_file = self.DATASETS[self.dataset_name]["train_file"]

    def _get_file(self, fname):
        hf_hub_download(repo_id=self.repo_id, filename=fname, repo_type="dataset", local_dir=self.local_dir)

    def download(self, num_chunks):
        print(f"Downloading {self.dataset_name} to {self.local_dir}")
        self._get_file(self.val_file % 0)

        for i in range(1, num_chunks + 1):
            self._get_file(self.train_file % i)


if __name__ == "__main__":
    # we can pass an argument to download less
    num_chunks = int(sys.argv[1]) if len(sys.argv) >= 2 else 8

    # Default dataset
    data_loader = FinewebCachedDataLoader("fineweb10B")
    data_loader.download(num_chunks)
