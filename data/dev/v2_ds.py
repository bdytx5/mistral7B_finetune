from datasets import Dataset
from datasets.features import Features, Value
import json
import random
from datasets import load_dataset
# Step 1: Write 100 JSON entries to a file

data_to_write = [{"text": f"Example text {i}"} for i in range(100)]

with open("example_data.jsonl", "w") as f:
    for entry in data_to_write:
        json.dump(entry, f)
        f.write("\n")


# Step 2: Create a generator function to yield entries one-by-one

def data_generator():
    with open("example_data.jsonl", "r") as f:
        for line in f:
            yield json.loads(line)


# Step 3: Load the data into Hugging Face Dataset


dataset = Dataset.from_generator(data_generator)

# Now you have a Hugging Face Dataset object, 'dataset', containing the data.
# You can access its elements in a streaming manner.

print(dataset[0])  # Prints the first example in the dataset



# Load as streaming dataset
train_dataset = load_dataset('json', data_files='example_data.jsonl', split='train', streaming=True)
print(train_dataset[0])  # Prints the first example in the dataset