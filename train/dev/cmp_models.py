from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from mistral.cache import RotatingBufferCache
from pathlib import Path
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_mistral_embedding(model: Transformer, tokenizer: Tokenizer):
    prompt = "This is a test"
    encoded_prompt = tokenizer.encode(prompt, bos=True)
    print(f"Mistral encoded prompt: {encoded_prompt}")

    seqlen = len(encoded_prompt)

    cache = RotatingBufferCache(model.args.n_layers, model.args.max_batch_size, model.args.sliding_window, model.args.n_kv_heads, model.args.head_dim)
    cache.to(device=model.device, dtype=model.dtype)
    print("mistral type {}".format(model.dtype))
    cache.reset()

    prelogits = model.get_last_hidden_state(
        torch.tensor(encoded_prompt, device=model.device, dtype=torch.long),
        cache,
        seqlens=[seqlen]
    )
    return prelogits


def generate_huggingface_embedding():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, device_map='cuda:0')

    model = model.to("cuda")  # Move to CUDA
    model.eval()

    prompt = "This is a test"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move input tensors to CUDA

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

    return last_hidden_state

    
# if __name__ == "__main__":

#     # running into cuda error when trying to do both .. Just comment out the code for 
#     model_path = "/root/mistral-src/mistral-7B-v0.1/"  # Replace with your model path
    
#     # Using Mistral
#     tokenizer_mistral = Tokenizer(str(Path(model_path) / "tokenizer.model"))
#     transformer = Transformer.from_folder(Path(model_path), max_batch_size=1)
#     mistral_embedding = generate_mistral_embedding(transformer, tokenizer_mistral)

#     with open("mistral_embedding.pkl", "wb") as f:
#         pickle.dump(mistral_embedding.cpu().detach().numpy(), f)
        
#     # Unload Mistral model from GPU
#     del transformer
#     torch.cuda.empty_cache()
    
#     # Using Hugging Face
#     huggingface_embedding = generate_huggingface_embedding()

#     with open("huggingface_embedding.pkl", "wb") as f:
#         pickle.dump(huggingface_embedding.cpu().detach().numpy(), f)

#     # Compare embeddings
#     mistral_numpy = mistral_embedding.cpu().detach().numpy()
#     huggingface_numpy = huggingface_embedding.cpu().detach().numpy()
    
#     difference = mistral_numpy - huggingface_numpy

#     print(f"Embedding difference (L2 norm): {torch.norm(torch.tensor(difference))}")
    
#     # Load and compare tokenized inputs from saved pickle files
#     with open("mistral_embedding.pkl", "rb") as f:
#         mistral_pickle = pickle.load(f)
        
#     with open("huggingface_embedding.pkl", "rb") as f:
#         huggingface_pickle = pickle.load(f)
        
#     token_diff = mistral_pickle - huggingface_pickle

#     print(f"Tokenized Input difference (L2 norm): {torch.norm(torch.tensor(token_diff))}")


import pickle
import numpy as np
from pathlib import Path

def main():
    # Load and compare saved embeddings
    with open("mistral_embedding.pkl", "rb") as f:
        mistral_pickle = pickle.load(f)

    with open("huggingface_embedding.pkl", "rb") as f:
        huggingface_pickle = pickle.load(f)

    # Sample a few values from each embedding
    mistral_sample = mistral_pickle[:3, :3]
    huggingface_sample = huggingface_pickle[:3, :3]
    print("Shape of Mistral embedding:", mistral_pickle.shape)
    print("Shape of Hugging Face embedding:", huggingface_pickle.shape)


    print("Sampled values from Mistral embedding:", mistral_sample)
    print("Sampled values from Hugging Face embedding:", huggingface_sample)

    embedding_diff = mistral_pickle - huggingface_pickle
    print(f"Embedding difference (L2 norm): {torch.norm(torch.tensor(embedding_diff))}")

    # Clear CUDA memory to be sure
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()



# from transformers import AutoTokenizer
# from pathlib import Path
# from mistral.tokenizer import Tokenizer

# if __name__ == "__main__":
#     model_path = "/root/mistral-src/mistral-7B-v0.1/"

#     # Using Mistral
#     tokenizer_mistral = Tokenizer(str(Path(model_path) / "tokenizer.model"))
#     mistral_tokens = tokenizer_mistral.encode("This is a test")
#     print(f"Mistral tokens: {mistral_tokens}")

#     # Using Hugging Face
#     tokenizer_huggingface = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#     huggingface_tokens = tokenizer_huggingface.encode("This is a test")
#     print(f"Hugging Face tokens: {huggingface_tokens}")

#     # Compare Tokens
#     print(f"Are tokens same: {mistral_tokens == huggingface_tokens}")
