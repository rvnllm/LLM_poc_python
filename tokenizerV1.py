import urllib.request
import re

#url = ( "https://raw.githubusercontent.com/rasbt/"
#        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#        "the-verdict.txt")
#file_path = "the-verdict.txt"
#urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
#print("Total number of characters:", len(raw_text))
#print(raw_text[:99])

#text = "Hello, world. Is this-- a test?"
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(result))
#for integer, token in enumerate(all_words):
  #  print(integer, token)
vocab = {token:integer for integer, token in enumerate(all_words)}
#for i, item in enumerate(vocab.items()):
#    print(item)
#    if i >= 50:
#        break



####################
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items() }
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # 1st pass strip naked
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


## special tokens
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items() }
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # 1st pass strip naked
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        # 2nd pass sanitize, robustness using unknowns
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text



if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    #print(preprocessed[:30])
    # building tokens
    all_tokens = sorted(set(preprocessed))
    # adding special control tokens to the vocabulary so that LLM training can understand missing tokens or other things
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer, token in enumerate(all_tokens)}

    print(len(vocab.items()))
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)


    tokenizer = SimpleTokenizerV2(vocab)
    text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    text_decode = tokenizer.decode(ids)
    print(text_decode)

    # non existent token problem
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))

    print(text)
    
    print(tokenizer.encode(text))


    
