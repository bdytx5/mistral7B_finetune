from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch 
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


##############  test data 
# # Data with instruction-response pairs
# data = [
#         "<s>[INST] How do you make coffee? [/INST] To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder cup, reduce the steeping time. To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder</s>",
#         "<s>[INST] How do you make coffee? [/INST] To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder cup, reduce the steeping time. To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder</s>",
#         "<s>[INST] How do you make coffee? [/INST] To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder cup, reduce the steeping time. To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder</s>",
#         "<s>[INST] How do you make coffee? [/INST] To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder cup, reduce the steeping time. To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder To make the perfect cup of coffee, you'll need some essential tools: a coffee grinder, a coffee maker or French press, fresh coffee beans, and filtered water. Start by grinding the coffee beans to a medium coarseness. If you grind it too fine, the coffee may turn out bitter; if it's too coarse, you'll end up with a weak cup. Use about 1 ounce of coffee for every 16 ounces of water. Boil the water to around 200°F, then let it sit for 30 seconds to cool slightly. Add the coffee grounds to the coffee maker or French press, then pour in the hot water. Let it steep for about 4 minutes before serving. You can adjust the brewing time to taste. If you like your coffee strong, let it steep a bit longer; for a milder</s>",
# ]

# # Create a text file from the data array
# with open("train_dataset.txt", "w") as f:
#     for item in data:
#         f.write(item + "\n")

##############  end of test data 

# for smaller compute 
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16)
MAX_SEQ_LEN = 2048

# Prepare the dataset and data collator
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="./output_examples_train.txt",
    block_size=MAX_SEQ_LEN
)

valid_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="./output_examples_val.txt",  # Change this path to your validation dataset
    block_size=MAX_SEQ_LEN
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)


training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=1,
    save_steps=20,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    report_to='wandb',  # Add this line to enable Wandb logging
    logging_steps=10, # log every 10 steps 
    gradient_checkpointing=True,
    deepspeed="./zero2_deepspeed.json"  # Add this line for DeepSpeed
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,  # Added for validation
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    tokenizer=tokenizer,
)
trainer.train()

trainer.save_model("./output/final_model")