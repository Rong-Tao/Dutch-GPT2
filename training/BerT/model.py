# model.py
import torch
from transformers import BertConfig, BertForMaskedLM

def get_model():
    config = BertConfig(
        vocab_size=52_000,
        max_position_embeddings=512,
        num_attention_heads=6,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = BertForMaskedLM(config=config)
    print(model.num_parameters())
    return model