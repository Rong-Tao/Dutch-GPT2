# model.py
from transformers import GPT2Config, GPT2LMHeadModel

def get_model():
    config = GPT2Config(
        vocab_size = 52_000,
        n_positions = 512
    )
    model = GPT2LMHeadModel(config=config)
    print(model.num_parameters())
    return model