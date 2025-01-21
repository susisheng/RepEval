#%% set up environment
import os
from dotenv import load_dotenv

# load_dotenv() 
# cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES')
# print(f'CUDA Device: {cuda_devices}')
#%%
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
#%% 
from utils import load_file
import importlib
import get_prompt_label
importlib.reload(get_prompt_label)
from get_prompt_label import get_prompts_labels
#%% model & tokenizer type
model_context_dict = {
    "mistralai/Mistral-7B-v0.1": 8192,
    "meta-llama/Llama-2-7b-hf": 4096,
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "chavinlo/alpaca-native": 2048,
    "eachadea/vicuna-7b-1.1": 2048,
    "tiiuae/falcon-7b": 2048
}
model_template_dict = {
    # 'mistralai/Mistral-7B-Instruct-v0.2': "Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-2-7b-chat-hf": "llama-2",
    "chavinlo/alpaca-native": "alpaca",
    "eachadea/vicuna-7b-1.1": "vicuna_v1.1",
    "tiiuae/falcon-7b": "falcon"
}

#%% Load model and tokenizer
def get_model_tokenizer(model_name_or_path, max_length=2048):
    print("Load Model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto", 
        torch_dtype=torch.float16
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length=max_length)
    tokenizer.pad_token = tokenizer.eos_token
    print("Load Model Done")
    return model, tokenizer

#%% tokenization
def get_prompt_input_by_tokenizer(tokenizer, prompt):
    # identify whether the model is a chat model
    if tokenizer.name_or_path in model_template_dict:
        if "system" in tokenizer.chat_template:
            messages = prompt
        else:
            content = " ".join([m["content"] for m in prompt])
            messages = [
                {"role": "user", "content": content}
            ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    # else concatenate the prompt
    else:
        prompt = " ".join([m["content"] for m in prompt])
    return prompt

def get_tokenized_prompts(tokenizer, prompts, max_length=2048, batch_size=-1):
    tokenized_inputs = []
    exceed_max_length = [False] * len(prompts)
    if batch_size == -1:
        batch_size = len(prompts)
    
    # process prompts
    if not isinstance(prompts[0], str):
        prompts = [get_prompt_input_by_tokenizer(tokenizer, prompt) for prompt in prompts]
    # tokenization
    for i in tqdm(range(0, len(prompts), batch_size)):
        input_texts = prompts[i: i + batch_size]
        inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
        if inputs["input_ids"].size(-1) > max_length:
            exceed_max_length[i: i + batch_size] = [True] * len(input_texts)
        tokenized_inputs.append(inputs)
    
    # warning if some inputs exceed max_length
    if any(exceed_max_length):
        print(f"Warning: {sum(exceed_max_length)} samples exceed max_length.")
    return tokenized_inputs, exceed_max_length

#%% Get embedding
def get_embedding(model, tokenizer, tokenized_inputs=None, prompts=None, batch_size=-1, token_range=[-1,None]):
    assert tokenized_inputs is not None or prompts is not None
    if tokenized_inputs is None:
        if batch_size == -1:
            batch_size = len(prompts)
        tokenized_inputs, _ = get_tokenized_prompts(tokenizer, prompts, batch_size=batch_size)
    embeddings = []
    with torch.no_grad():
        print("calcu embedding...")
        for inputs in tqdm(tokenized_inputs):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states: (layer, batch_size, seq_len, hidden_size)
            hidden_states = torch.stack(outputs["hidden_states"][1:]).transpose(0,1)
            embeddings.extend(hidden_states[:, :, token_range[0]:token_range[1], :].cpu())
    embeddings_to_save = torch.stack(embeddings)
    return embeddings_to_save.numpy()

#%% Prepare representation and labels
def get_embedding_and_labels(model, tokenizer, prompts, labels, eval_type, batch_size=-1, max_length=2048, token_range=[-1,None]):
    tokenized_inputs, exceed_max_length = get_tokenized_prompts(tokenizer, prompts, max_length=max_length, batch_size=batch_size)
    # drop samples that exceed max_length
    if eval_type == "absolute":
        delete_index = [i for i, exceed in enumerate(exceed_max_length) if
                        exceed]
    elif eval_type == "pairwise":
        for i in range(0, len(exceed_max_length), 2):
            if exceed_max_length[i] or exceed_max_length[i+1]:
                exceed_max_length[i], exceed_max_length[i+1] = True, True
        delete_index = [i for i, exceed in enumerate(exceed_max_length) if exceed]
    else:
        raise ValueError("Invalid eval_type.")
    
    tokenized_inputs = [tokenized_inputs[i] for i in range(len(tokenized_inputs)) if i not in delete_index]
    labels = [label for i, label in enumerate(labels) if i not in delete_index]
    embeddings = get_embedding(model, tokenizer, tokenized_inputs=tokenized_inputs, token_range=token_range)
    return embeddings, labels

#%% Save representation and labels
def save_representations_and_labels(representations, labels, rep_label_dir, config=None):
    if not os.path.exists(rep_label_dir):
        os.makedirs(rep_label_dir)
    rep_file_path = os.path.join(rep_label_dir, "embedding.pkl")
    label_file_path = os.path.join(rep_label_dir, "labels.pkl")
    # save rep and label
    print(f"save embedding to {rep_file_path}...")
    pickle.dump(representations, open(rep_file_path, "wb"))
    print(f"save labels to {label_file_path}")
    pickle.dump(labels, open(label_file_path, "wb"))
    # save config
    if config is not None:
        config_path = os.path.join(rep_label_dir, "config.json")
        print(f"save config to {config_path}")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

#%% Get Representation
def load_rep_label(config, model=None, tokenizer=None):
    # path config
    rep_label_dir = config.get("rep_label_dir", "")
    rep_file_path, label_file_path = "", ""
    if rep_label_dir != "" :
        if not os.path.exists(rep_label_dir):
            os.makedirs(rep_label_dir)
        rep_file_path = os.path.join(rep_label_dir, "embedding.pkl")
        label_file_path = os.path.join(rep_label_dir, "labels.pkl")
    
        # Try load previous embedding and labels
        replace_embedding = config.get("replace_embedding", False)
        if os.path.exists(rep_file_path) and not replace_embedding:
            print(f"load labels from {label_file_path}...")
            labels = pickle.load(open(label_file_path, "rb"))
            if isinstance(labels, list):
                labels = np.array(labels)
            print(f"load embedding from {rep_file_path}...")
            return pickle.load(open(rep_file_path, "rb")), labels
        # if could not find embedding.pkl and labels.pkl, try add prompt template and model_name to file path
        else:
            if config.get("prompt_template_name") is None:
                prompt_template_file = config.get("prompt_template_file", "")
                prompt_template_name = os.path.basename(prompt_template_file).split(".")[0]
            else:
                prompt_template_name = config["prompt_template_name"]
            rep_label_dir = os.path.join(rep_label_dir, prompt_template_name, config["model_name_or_path"].replace("/", "_"))
            if not os.path.exists(rep_label_dir):
                os.makedirs(rep_label_dir)
            rep_file_path = os.path.join(rep_label_dir, "embedding.pkl")
            label_file_path = os.path.join(rep_label_dir, "labels.pkl")
            
            if os.path.exists(rep_file_path) and not replace_embedding:
                print(f"load labels from {label_file_path}...")
                labels = pickle.load(open(label_file_path, "rb"))
                if isinstance(labels, list):
                    labels = np.array(labels)
                print(f"load embedding from {rep_file_path}...")
                return pickle.load(open(rep_file_path, "rb")), labels
            # print the save path
            print(f"The embedding and labels will be saved to {rep_label_dir}")
    
    # Else, load model and tokenizer
    if model is None or tokenizer is None or model.name_or_path != config["model_name_or_path"]:
        if model is not None and model.name_or_path != config["model_name_or_path"]:
            print(f"Warning: model name mismatch. Load model from {config['model_name_or_path']}")
        model_name_or_path = config["model_name_or_path"]
        max_length = config.get("max_length", 2048)
        model, tokenizer = get_model_tokenizer(
            model_name_or_path=model_name_or_path, max_length=max_length)
    
    # Evaluation type
    eval_type = config["eval_type"]
    assert eval_type in ["pairwise", "absolute"]
    
    # Load prompts and labels
    prompts, labels = get_prompts_labels(
        eval_type=eval_type, 
        content_file=config["content_file"],
        prompt_template_file=config["prompt_template_file"]
    )
            
    # Get Representation
    token_range = config.get("token_range", [-1, None])
    batch_size = config.get("batch_size", 1)
    representations, labels = get_embedding_and_labels(
        model, tokenizer, prompts, labels, eval_type, batch_size=batch_size, token_range=token_range)
    
    if rep_label_dir != "":
        # save embedding and labels
        print(f"save embedding to {rep_file_path}...")
        pickle.dump(representations, open(rep_file_path, "wb"))
        print(f"save labels to {label_file_path}")
        pickle.dump(labels, open(label_file_path, "wb"))
        
        # save config
        config_path = os.path.join(rep_label_dir, "config.json")
        print(f"save config to {config_path}")
        with open(config_path, "w") as f:
            json.dump(config, f)
    
    return_embedding = config.get("return_embedding", rep_file_path == "")
    if return_embedding:
        return representations, labels

#%%
if __name__ == "__main__":
    # from config import config
    config = {
        "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "eval_type": "absolute",
        "content_file": "data/example.json",
        "prompt_template": "prompts/entail.txt",
        "rep_label_dir": "./data/representations/entailment",
        "return_embedding": True
    }
    model, tokenizer = get_model_tokenizer("mistralai/Mistral-7B-Instruct-v0.2")
    results = load_rep_label(config, model, tokenizer)
