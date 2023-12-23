import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizerFast
import datasets
# dataset.py

class GPT2Dataset(Dataset):
    def __init__(self):
        self.ds  = datasets.Dataset.load_from_disk("../../dataset/Tokenized_dataset.hf")

    def __len__(self):
        """Returns the number of examples in the dataset."""
        return len(self.ds)

    def __getitem__(self, idx):
        input_id = self.ds[idx]['input_ids']
        attention_mask = self.ds[idx]['attention_mask']
        targets = input_id[1:]
        targets = torch.cat([targets, torch.tensor([1])], dim=0)
        return input_id, attention_mask, targets