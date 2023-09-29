import json
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling
import argparse
import json
import random

# Limit of tokens
TOKEN_LIMIT = 2048
beg = "<s>[INST]"
en_inst = "[/INST]"
eos = "<s>"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

def count_tokens(text):
    return len(tokenizer.encode(text))


fixed_token_count = count_tokens(beg) + count_tokens(en_inst) + count_tokens(eos) + (count_tokens(" ")*4) + 1 # add a token for good measure :)
TOKEN_LIMIT -= fixed_token_count



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



def generate_examples_max_context(messages, fill_context=True):
    global eos, beg, en_inst
    examples = []
    convo_sample = ""
    token_count = 0
    full_cntxt = False
    for message in messages:
        author = message['author']
        text = message['text']
        # current_msg = f"{author}: {text} "
        current_msg = f"{text} "

        current_msg_token_count = count_tokens(current_msg)

        if token_count + current_msg_token_count <= TOKEN_LIMIT:
            full_cntxt = False
            convo_sample += current_msg
            token_count += current_msg_token_count
        else:
            if author == 'ChatGPT':  # trim down context starting at the beginning
                full_cntxt = True
                
                if current_msg_token_count >= TOKEN_LIMIT: # single message is longer than max context 
                    full_cntxt = False 
                    convo_sample = current_msg
                    continue 

                while token_count + current_msg_token_count > TOKEN_LIMIT:

                    if len(convo_sample) > 50:
                        convo_sample = convo_sample[50:] # trim down prev context 
                        # Recalculate token_count based on the modified convo_sample
                        token_count = count_tokens(convo_sample)
                    else:
                        full_cntxt = False # since sample is just message
                        break 
            

        if author == 'ChatGPT' and full_cntxt:  # Add to examples after each assistant's message

            formatted_example = f"{beg} {convo_sample} {en_inst} {current_msg.strip()} {eos}"
            examples.append(formatted_example.strip())
            convo_sample += current_msg
            token_count += current_msg_token_count
        else:
            # create tokens for user chatgpt or not 
            convo_sample += current_msg
            token_count += current_msg_token_count

    return examples


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('val_pct', type=float, help='Validation percentage')
    # parser.add_argument('gpt_data', type=str, help='Path to GPT data JSON')
    # args = parser.parse_args()


    # For Debugging 
    class ManualArgs:
        def __init__(self, val_pct, gpt_data):
            self.val_pct = val_pct
            self.gpt_data = gpt_data

    args = ManualArgs(val_pct=0.3, gpt_data="/Users/brettyoung/Desktop/mistral7b/data/conversations.json")



    val_pct = args.val_pct
    gpt_data = args.gpt_data

    with open(gpt_data, 'r') as f:
        jsonData = json.load(f)
    
    with open('./output_examples_train.txt', 'w') as train_f, \
         open('./output_examples_val.txt', 'w') as val_f:
        for conversation in jsonData:
            messages = get_conversation_messages(conversation)
            examples = generate_examples_max_context(messages)
            for i, example in enumerate(examples):
                if random.random() < val_pct:
                    val_f.write(example + '\n')
                else:
                    train_f.write(example + '\n')



