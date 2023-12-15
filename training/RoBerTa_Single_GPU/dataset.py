import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizerFast
import datasets

# dataset.py

class GPT2Dataset(Dataset):
    def __init__(self):
        self.ds = datasets.Dataset.load_from_disk("../../dataset/Tokenized_dataset.hf")

    def __len__(self):
        """Returns the number of examples in the dataset."""
        return len(self.ds)

    def __getitem__(self, idx):
        input_id = self.ds[idx]['input_ids']
        attention_mask = self.ds[idx]['attention_mask']
        targets = input_id[1:]
        targets = torch.cat([targets, torch.tensor([1])], dim=0)
        return input_id, attention_mask, targets

def get_loaders(batch_size, split_ratio):
    full_dataset = GPT2Dataset()
    train_size = int(split_ratio * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader
