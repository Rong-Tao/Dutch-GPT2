from transformers import RobertaTokenizerFast
from datasets import Dataset
import numpy as np

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer/bpe-post", max_len=512)
def tokenize_function(examples):
    text_column_name = 'text'
    return tokenizer(
        examples[text_column_name],
        padding = 'max_length',
        truncation = True,
        max_length = 512
    )
dataset = Dataset.load_from_disk("../dataset/dutch.hf")
dataset = dataset.with_format("torch",columns='text')

tokenized_dataset = dataset.map(
    tokenize_function,
    batched = True,
    num_proc = 24
)

tokenized_dataset.save_to_disk('Tokenized_dataset.hf')