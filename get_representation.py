#%% set up environment
import os
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()  # 加载.env文件

# 现在可以使用CUDA_VISIBLE_DEVICES
cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES')
print(f'CUDA Device: {cuda_devices}')

#%%
import json
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
#%% Utils
def load_file(file_path):
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".pkl"):
        return pickle.load(open(file_path, "rb"))
#%% Get Prompt
def load_prompt_template(prompt_template_file):
    with open(prompt_template_file, "r") as f:
        prompt_template = f.read()
    return prompt_template

def is_prompt_template_valid(eval_type, prompt_template):
    """Check if the prompt template is valid."""
    if eval_type == "pairwise":
        if "{response_A}" not in prompt_template or "{response_B}" not in prompt_template:
            return False
    return True 

def get_pairwise_prompt(prompt_template, contents):
    """contents contains response_A and response_B. The function switch the order of response_A and response_B and get two prompts. Therefore, in training, make sure A is better than B."""
    prompt_AB = prompt_template.format(**contents)
    # switch A and B
    contents["response_A"], contents["response_B"] = contents["response_B"], contents["response_A"]
    prompt_BA = prompt_template.format(**contents)
    return prompt_AB, prompt_BA

def get_absolute_prompt(prompt_template, contents):
    return prompt_template.format(**contents)

def get_prompt(eval_type, prompt_template, contents):
    if eval_type == "pairwise":
        return get_pairwise_prompt(prompt_template, contents)
    elif eval_type == "absolute":
        return get_absolute_prompt(prompt_template, contents)
    else:
        raise ValueError("Invalid eval_type.")

def get_prompts(eval_type, prompt_template, contents_list):
    prompts = []
    for contents in contents_list:
        if eval_type == "pairwise":
            prompts.extend(get_prompt("pairwise", prompt_template, contents))
        elif eval_type == "absolute":
            prompts.append(get_prompt("absolute", prompt_template, contents))
        else:
            raise ValueError("Invalid eval_type.")
    return prompts

#%% Utils
def get_embedding_and_labels(
    model, tokenizer, prompts, labels, batch_size=-1, max_length=2048, token_range=[-1,None]
    ):
    if batch_size == -1:
        batch_size = len(prompts)
    embeddings = []
    delete_index = []
    with torch.no_grad():
        print("calcu embedding...")
        cnt = 0
        for i in tqdm(range(0, len(prompts), batch_size)):
            cnt += 1
            input_texts = prompts[i: i + batch_size]
            inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
            if inputs["input_ids"].size(-1) > max_length:
                delete_index.append(i)
                continue
            outputs = model(**inputs, output_hidden_states=True)
            #hidden_states: (layer, batch_size, seq_len, hidden_size)
            hidden_states = torch.stack(outputs["hidden_states"][1:]).transpose(0,1)
            # embeddings.extend(hidden_states[layer_range[0]:layer_range[1], :, token_range[0]:token_range[1], :].cpu())
            embeddings.extend(hidden_states[:, :, token_range[0]:token_range[1], :].cpu())

    embeddings_to_save = torch.stack(embeddings)
    embeddings_to_save = embeddings_to_save.transpose(0,1)
    labels = [label for i, label in enumerate(labels) if i not in delete_index]
    print(f"delete {len(delete_index)} samples, with index: {delete_index}")
    # if file_to_save != "":
    #     print(f"save embedding to {file_to_save}...")
    #     pickle.dump(embeddings_to_save, open(file_to_save, "wb"))
        
    return embeddings_to_save, labels

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

#%% Get Representation
def get_representation_labels(config, model=None, tokenizer=None):
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
    
    # Load model and tokenizer
    if model is None or tokenizer is None:
        model_name_or_path = config["model_name_or_path"]
        max_length = config.get("max_length", 2048)
        model, tokenizer = get_model_tokenizer(
            model_name_or_path=model_name_or_path, max_length=max_length)
    
    # Evaluation type
    eval_type = config["eval_type"]
    assert eval_type in ["pairwise", "absolute"]
    
    # Prompt template
    prompt_template_file = config["prompt_template"]
    prompt_template = load_prompt_template(prompt_template_file)
    if not is_prompt_template_valid(eval_type, prompt_template):
        raise ValueError("Prompt template is not valid.")
    
    # Load contents
    content_file = config["content_file"]
    content_list = load_file(content_file)
    prompts = get_prompts(eval_type, prompt_template, content_list)
    
    # Prepare labels
    if eval_type == "absolute":
        labels = [content["label"] for content in content_list]
    elif eval_type == "pairwise":
        labels = []
        for content in content_list:
            # here label should be 'A' or 'B'
            assert content["label"] in ["A", "B"]
            labels.extend([1,0] if content["label"] == "A" else [0,1])
            
    # Get Representation
    # layer_range = config.get("layer_range", [-1, None])
    token_range = config.get("token_range", [-1, None])
    
    batch_size = config.get("batch_size", 1)
    representations, labels = get_embedding_and_labels(
        model, tokenizer, prompts, labels, 
        # layer_range=layer_range,
        token_range=token_range, 
        batch_size=batch_size)
    
    if rep_label_dir != "":
        print(f"save embedding to {rep_file_path}...")
        pickle.dump(representations, open(rep_file_path, "wb"))
        print(f"save labels to {label_file_path}")
        pickle.dump(labels, open(label_file_path, "wb"))
    
    return_embedding = config.get("return_embedding", rep_file_path == "")
    if return_embedding:
        return representations, labels

#%%
if __name__ == "__main__":
    # from config import config
    config = {
        "model_name_or_path": "meta-llama/Meta-Llama-3-8B",
        "eval_type": "absolute",
        "content_file": "data/example.json",
        "prompt_template": "prompts/entail.txt",
        "rep_label_dir": "./data/representations/entailment",
        "return_embedding": True
    }
    model, tokenizer = get_model_tokenizer("meta-llama/Meta-Llama-3-8B")
    results = get_representation_labels(config, model, tokenizer)
