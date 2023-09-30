from datasets import load_dataset
import random

# Load the dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")

# Access the train and validation splits
train_dataset = dataset["train"]


# Sample 5 random examples from train_dataset
train_sample_indices = random.sample(range(len(train_dataset)), 5)
train_samples = [train_dataset[i] for i in train_sample_indices]


# Print the samples
print("Train samples:")
for sample in train_samples:
    print(sample)
