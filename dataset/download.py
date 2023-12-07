# Importing the load_dataset function from the 'datasets' library
from datasets import load_dataset

# Loading the Dutch subset of the OSCAR corpus dataset
# 'oscar-corpus/OSCAR-2109' refers to the specific version of the OSCAR dataset
# 'deduplicated_nl' specifies the Dutch language subset
# 'split="train"' indicates that we are loading the training split of the dataset
dataset = load_dataset("oscar-corpus/OSCAR-2109", "deduplicated_nl", split="train")

# Saving the loaded dataset to disk
# This creates a file 'dutch.hf' which contains the dataset in a format
# that can be easily reloaded by the 'datasets' library
dataset.save_to_disk("dutch.hf")
