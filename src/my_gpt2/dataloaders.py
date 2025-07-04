from .Datasets import GPTDatasetV1
import tiktoken
from torch.utils.data import DataLoader

def create_dataloader_v1(txt,
                         tokenizer = None,
                         batch_size = 4, 
                         max_length = 256, 
                         stride = 128, 
                         shuffle = True,
                         drop_last = True,
                         num_workers = 0):
    if tokenizer == None: 
        tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader