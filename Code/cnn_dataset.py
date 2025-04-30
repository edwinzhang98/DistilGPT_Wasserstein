from datasets import load_from_disk
import torch
from torch.utils.data import Dataset

class CNNDMDistillationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        article = sample["article"]
        
        # 将 article 转为模型输入的 token IDs
        # Convert article to token IDs for model input
        tokenized = self.tokenizer(
            article,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        labels = input_ids.clone()  # 为了计算损失，将 labels 设置为 input_ids # Set labels to input_ids for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100  # 忽略 padding 部分的损失 # Ignore loss for padding parts
        
        return {
            "article": article,  # 返回原始文本 # Return original text
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels
        }



from torch.utils.data import Dataset
import torch

class OpenWebTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        """
        dataset: Hugging Face datasets.Dataset object, expected to have a 'text' column.
        tokenizer: Tokenizer to use.
        max_length: Maximum sequence length.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample.get("text", "") # Get text, default to empty string if key missing

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length", # Pad to max_length
            truncation=True,
            return_tensors="pt" # Return PyTorch tensors
        )

        input_ids = encoding["input_ids"].squeeze(0) # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze(0) # Remove batch dimension

        # Create labels for language modeling (same as input_ids, mask padding)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100 # Use -100 index for ignored tokens in loss calculation

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
