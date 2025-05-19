from importlib.metadata import version

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire text, why not (while at it lets load datageddon - gavin belson - into it): 
        # use a sliding window to chunk the data into overlapping sequences of max_length
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 0 where we start
        # len - max length otherwise we fall hard over the edge
        # stride aka steps
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # returns the total number of INPUT rows
    def __len__(self):
        return len(self.input_ids)
    
    # return a single row from both input and target
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # using OpenAPI encoder
    tokenizer = tiktoken.get_encoding("gpt2")

    
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,      
        shuffle=shuffle,            # shuffle or not
        drop_last=drop_last,        # be careful here, drops the last batch if it is shorter than the specified batch to prevent loss spikes
        num_workers=num_workers     # thats fun, multithreading
    )

    return dataloader



if __name__ == "__main__":

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text, batch_size = 8, max_length = 4, stride = 1, shuffle = False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
   # print(f"input: {first_batch[0]}")
   # print(f"target:{first_batch[1]}\n")
    
    second_batch = next(data_iter)
   # print(f"{second_batch[0]}")
    
    third_batch = next(data_iter)
   # print(f"{third_batch[0]}")
   
    # don't ask, accept it. your faith
#    vocab_size = 50257
#    output_dim = 256
#    context_length = 1024


 #   token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
 #   pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

#    batch_size = 8
#    max_length = 4
#    dataloader = create_dataloader_v1(
#        raw_text,
#        batch_size=batch_size,
#        max_length=max_length,
#        stride=max_length
#    )

    vocab_size = 6
    output_dim = 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)

    input_ids = ([2, 3, 5, 1])
    print(embedding_layer(torch.tensor([3])))
    
