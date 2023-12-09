# model.py
import torch
from transformers import RobertaConfig, RobertaForMaskedLM

def get_model():
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config=config)
    return model