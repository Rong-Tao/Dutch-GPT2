import torch
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import TextDataset, LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained("../tokenizer/bpe-post", max_len=512)
model = RobertaForMaskedLM(config=config)

print(model.num_parameters())


dataset = Dataset.load_from_disk("../dataset/dutch.hf")
dataset = dataset.with_format("torch",columns='text')
tokenized_dataset = dataset.map(
    tokenizer,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./DutchBERTo",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_gpu_train_batch_size=64,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./DutchBERTo")