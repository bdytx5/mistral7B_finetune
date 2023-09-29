import json
from collections import deque

# Limit of tokens
TOKEN_LIMIT = 4096

def count_tokens(text):
    return len(text.split())

def get_conversation_messages(conversation):
    messages = []
    current_node = conversation['current_node']
    while current_node is not None:
        node = conversation['mapping'][current_node]
        if (
            node.get('message') is not None and
            'content' in node['message'] and
            node['message']['content']['content_type'] == 'text' and
            len(node['message']['content']['parts']) > 0 and
            len(node['message']['content']['parts'][0]) > 0 and
            (node['message']['author']['role'] != 'system' or node['message']['metadata']['is_user_system_message'])
        ):
            author = node['message']['author']['role']
            if author == 'assistant':
                author = 'ChatGPT'
            elif author == 'system' and node['message']['metadata']['is_user_system_message']:
                author = 'Custom user info'
            messages.append({'author': author, 'text': node['message']['content']['parts'][0]})
        current_node = node.get('parent', None)
    return list(reversed(messages))

def generate_examples(messages):
    examples = []
    tokens = deque()
    token_count = 0

    for message in messages:
        author = message['author']
        text = message['text']
        if author == 'user':
            current_tokens = [f"<s>[INST] {text} [/INST]</s>"]
        else:
            current_tokens = [f"<s>{text}</s>"]
        
        current_token_count = sum(count_tokens(t) for t in current_tokens)

        if token_count + current_token_count <= TOKEN_LIMIT:
            tokens.extend(current_tokens)
            token_count += current_token_count
        else:
            # Ensure last Q&A is included
            while token_count + current_token_count > TOKEN_LIMIT:
                removed_token = tokens.popleft()
                token_count -= count_tokens(removed_token)
            tokens.extend(current_tokens)
            token_count += current_token_count

        if author == 'ChatGPT':  # Add to examples after each assistant's message
            examples.append(" ".join(tokens))

    return examples

if __name__ == "__main__":
    with open('/Users/brettyoung/Desktop/1510/data/small_chatgptDs.json', 'r') as f:
        jsonData = json.load(f)

    for conversation in jsonData:
        messages = get_conversation_messages(conversation)
        examples = generate_examples(messages)
        for i, example in enumerate(examples[:10]):
            print(f"Example {i + 1} for {conversation['title']}:")
            print(example)
            print("#" * 50)
            print("#" * 50)
            print("#" * 50)
            print("#" * 50)