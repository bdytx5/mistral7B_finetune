from transformers import Conversation, AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# Add padding token if it's missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Create a Conversation object or list of dictionaries
conv = Conversation([
    {"role": "user", "content": "Hello, who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Hello, who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}
])

# Applying chat template with tokenization and padding
token_ids_with_padding = tokenizer.apply_chat_template(
    conversation=conv,
    tokenize=True,
    padding=False,
    max_length=50
)

# Applying chat template without tokenization (will return a string)
string_output = tokenizer.apply_chat_template(
    conversation=conv,
    tokenize=False
)

# Applying chat template with custom Jinja template
token_ids_custom_template = tokenizer.apply_chat_template(
    conversation=conv,
    chat_template="{{role}}: {{content}}",
    tokenize=True
)

# Print the token IDs with padding
print("Token IDs with padding:", token_ids_with_padding)

# Print the string output
print("String output:", string_output)

# Print the token IDs with a custom template
print("Token IDs with custom template:", token_ids_custom_template)
with open('./string_output.txt', 'w') as f:
    f.write(string_output)