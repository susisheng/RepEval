from utils import load_file

def load_prompt_template(prompt_template_file):
    if prompt_template_file.endswith(".txt"):
        with open(prompt_template_file, "r") as f:
            prompt_template = f.read()
    elif prompt_template_file.endswith(".json"):
        prompt_template = load_file(prompt_template_file)
    else:
        raise ValueError("Invalid prompt template file.")
    return prompt_template

def is_prompt_template_valid(eval_type, prompt_template):
    """Check if the prompt template is valid."""
    if eval_type == "pairwise":
        if "{response_A}" not in prompt_template or "{response_B}" not in prompt_template:
            return False
    return True 

def get_absolute_prompt(prompt_template, contents):
    return prompt_template.format(**contents)

def get_pairwise_prompt(prompt_template, contents):
    """contents contains response_A and response_B. The function switch the order of response_A and response_B and get two prompts. Therefore, in training, make sure A is better than B."""
    if isinstance(prompt_template, str):
        prompt_AB = prompt_template.format(**contents)
        # switch A and B
        contents["response_A"], contents["response_B"] = contents["response_B"], contents["response_A"]
        prompt_BA = prompt_template.format(**contents)
    # in chatGLM format
    elif isinstance(prompt_template, list):
        prompt_AB = [{"role": prompt["role"], "content": prompt["content"].format(**contents)} for prompt in prompt_template]
        # switch A and B
        contents["response_A"], contents["response_B"] = contents["response_B"], contents["response_A"]
        prompt_BA = [{"role": prompt["role"], "content": prompt["content"].format(**contents)} for prompt in prompt_template]
    return prompt_AB, prompt_BA

def get_single_prompt(eval_type, prompt_template, contents):
    if eval_type == "pairwise":
        return get_pairwise_prompt(prompt_template, contents)
    elif eval_type == "absolute":
        return get_absolute_prompt(prompt_template, contents)
    else:
        raise ValueError("Invalid eval_type.")

def get_all_prompts(eval_type, prompt_template, contents_list):
    prompts = []
    for contents in contents_list:
        if eval_type == "pairwise":
            prompts.extend(get_single_prompt("pairwise", prompt_template, contents))
        elif eval_type == "absolute":
            prompts.append(get_single_prompt("absolute", prompt_template, contents))
        else:
            raise ValueError("Invalid eval_type.")
    return prompts

def get_prompts_labels(eval_type, content_file=None, content_list=None, prompt_template=None, prompt_template_file=None):
    """Get prompt and label for each content."""
    # Load prompt template
    assert prompt_template is not None or prompt_template_file is not None
    if prompt_template is None:
        prompt_template = load_prompt_template(prompt_template_file)
    # if not is_prompt_template_valid(eval_type, prompt_template):
    #     raise ValueError("Prompt template is not valid.")
    
    # Get prompts
    assert content_file is not None or content_list is not None
    if content_list is None:
        content_list = load_file(content_file)
    prompts = get_all_prompts(eval_type, prompt_template, content_list)
    
    # Prepare labels
    if eval_type == "absolute":
        labels = [content["label"] for content in content_list]
    elif eval_type == "pairwise":
        labels = []
        for content in content_list:
            # here label should be 'A' or 'B'
            assert content["label"] in ["A", "B"]
            labels.extend([1,0] if content["label"] == "A" else [0,1])
    
    return prompts, labels
