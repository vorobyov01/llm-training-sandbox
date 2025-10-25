import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tiktoken import get_encoding
import torch.distributed as dist


class TinyStoriesDataset(Dataset):
    def __init__(self, tokenizer_name="cl100k_base", max_length=512):
        self.tokenizer = get_encoding(tokenizer_name)
        self.max_length = max_length
        
        # Load Tiny Stories dataset
        self.dataset = load_dataset("roneneldan/TinyStories", split="train")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input and target (shifted by 1)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids
        }


def collate_fn(batch):
    """Collate function to handle variable length sequences"""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], 
        batch_first=True, 
        padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch], 
        batch_first=True, 
        padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch], 
        batch_first=True, 
        padding_value=-100
    )
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def get_dataloader(batch_size=8, num_workers=4, max_length=512):
    dataset = TinyStoriesDataset(max_length=max_length)
    
    # Use DistributedSampler for DDP
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader
