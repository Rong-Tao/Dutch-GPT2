from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer

#tokenizer = AutoTokenizer.from_pretrained("./gpt")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=ByteLevelBPETokenizer()
    )

ds = Dataset.load_from_disk("../dataset/dutch.hf")
ds = ds.with_format("torch")

def batch_iterator():
    for batch in ds.select_columns("text").iter(batch_size=64):
        yield batch["text"]
        
tokenizer.train_new_from_iterator(batch_iterator(), 
                                  vocab_size=52_000)

tokenizer.save_pretrained("bpe-post")